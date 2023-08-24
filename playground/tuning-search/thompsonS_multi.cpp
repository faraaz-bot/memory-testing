#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "beta_distribution.hpp"

// compile: gcc thompsonS_multi.cpp -lm -lstdc++ -o thompsonS_multi.o
// run: ./thompsonS_multi.o num-samples
// example: ./thompsonS_multi.o 100

double max_outcome;
const int num_groups = 3;

const int num_elem_g0 = 7;
const int num_elem_g1 = 15;
const int num_elem_g2 = 10;
const std::vector<int> num_elem_in_group = {num_elem_g0, num_elem_g1, num_elem_g2};

class param_machine
{

private:
    double mean;
    double deviation;
    std::normal_distribution<double> dist;
    sftrabbit::beta_distribution<double> beta_dist;

    ///
    int success_count = 1;
    int failure_count = 1;
    double posterior = 0;

public:
    param_machine() = default;
    param_machine(double m, double d)
    : mean(m), deviation(d)
    {
        std::random_device rd;
        dist = std::normal_distribution<double>(mean, deviation);
    }

    template <typename URNG>
    double sample(URNG& engine)
    {
        return dist(engine);
    }

    ///
    void inc_success()
    {
        ++success_count;
    }

    void inc_failure()
    {
        ++failure_count;
    }

    int get_success_count()
    {
        return success_count;
    }

    int get_failure_count()
    {
        return failure_count;
    }

    double get_posterior()
    {
        return (double)(success_count) / (double)(success_count + failure_count);
    }

    template <typename URNG>
    double draw_beta_dist_sample(URNG& engine)
    {
        beta_dist.param(success_count, failure_count);
        return beta_dist(engine);
    }
};

class Machine
{
private:
    std::vector<param_machine*> using_machines;

public:
    Machine() = default;
    Machine(param_machine& m1, param_machine& m2, param_machine& m3)
    {
        using_machines = {&m1, &m2, &m3};
    }

    template <typename URNG>
    double evaluate(URNG& engine)
    {
        return using_machines[0]->sample(engine) * using_machines[1]->sample(engine) * using_machines[2]->sample(engine);
    }

    void success()
    {
        using_machines[0]->inc_success();
        using_machines[1]->inc_success();
        using_machines[2]->inc_success();
    }

    void failure()
    {
        using_machines[0]->inc_failure();
        using_machines[1]->inc_failure();
        using_machines[2]->inc_failure();
    }

    void posteriors(double& post0, double& post1, double& post2)
    {
        post0 = using_machines[0]->get_posterior();
        post1 = using_machines[1]->get_posterior();
        post2 = using_machines[2]->get_posterior();
    }
};

Machine machines[num_elem_g0][num_elem_g1][num_elem_g2];

// thompson sampling
// choose the button with largest theta (samling beta_dist) value
template <typename URNG>
int choose_next_sample_id(URNG& engine, std::vector<param_machine>& target_group)
{
    int num_candidates = target_group.size();
    int choosen = 0;
    double max_theta = target_group[0].draw_beta_dist_sample(engine);
    for(int i = 1; i < num_candidates; ++i)
    {
        double theta = target_group[i].draw_beta_dist_sample(engine);
        if(theta > max_theta)
        {
            max_theta = theta;
            choosen = i;
        }
    }

    return choosen;
}

int find_most_probable_machines(std::vector<param_machine>& target_group)
{
    int num_candidates = target_group.size();
    int best_idx = 0;
    double max_prob = target_group[0].get_posterior();
    for(int i = 1; i < num_candidates; ++i)
    {
        double prob = target_group[i].get_posterior();
        if(prob > max_prob)
        {
            best_idx = i;
            max_prob = prob;
        }
    }

    return best_idx;
}

void print_group_summary(std::vector<param_machine>& target_group)
{
    int num_candidates = target_group.size();

    std::cout << "\nGroup: \n";
    for(int i = 0; i < num_candidates; ++i)
    {
        std::cout << "Element " << i
                  << ": success count: " << target_group[i].get_success_count()
                  << ", failure count: " << target_group[i].get_failure_count()
                  << ", posterior: " << target_group[i].get_posterior() << "\n";
        }
}

// update if the reward is success or not
// success: outcome is within 10% of the current largest outcome
template <typename URNG>
void experiment(URNG& engine, int test_counter, int choosen_g1, int choosen_g2, int choosen_g3)
{
    auto& testing_machine = machines[choosen_g1][choosen_g2][choosen_g3];
    double outcome = testing_machine.evaluate(engine);

    // if max_outcome is 0 (first time), we make the reward = success
    double ratio = (max_outcome == 0)? 1 : outcome / max_outcome;
    bool good = ratio >= 0.9;

    if(good)
        testing_machine.success();
    else
        testing_machine.failure();

    double post0, post1, post2;
    testing_machine.posteriors(post0, post1, post2);
    max_outcome = std::max(max_outcome, outcome);

    std::cout << "\ntest: " << test_counter
              << ", params [" << choosen_g1 << ", " << choosen_g2 << ", " << choosen_g3 << "]"
              << ": outcome is " << outcome << ", max outcome is " << max_outcome << ", "
              << (good ? "success" : "failure")
              << ", posterior is [" << post0 << ", " << post1 << ", " << post2 << "]";
}

int main(int argc, char* argv[])
{
    int N = 50;
    if(argc >= 2)
        N = std::stoi(argv[1]);

    // obtain a seed from the system clock:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);  // mt19937 is a standard mersenne_twister_engine

    // best is 1
    std::vector<param_machine> param0(num_elem_in_group[0]);
    param0[0] = param_machine(85.0, 11.0);
    param0[1] = param_machine(94.0, 10.0);
    param0[2] = param_machine(18.0, 8.0);
    param0[3] = param_machine(50.0, 7.0);
    param0[4] = param_machine(35.0, 13.0);
    param0[5] = param_machine(47.0, 5.0);
    param0[6] = param_machine(74.0, 9.0);

    // best is 7
    std::vector<param_machine> param1(num_elem_in_group[1]);
    param1[0] = param_machine(28.0, 11.0);
    param1[1] = param_machine(47.0, 10.0);
    param1[2] = param_machine(82.0, 8.0);
    param1[3] = param_machine(67.0, 7.0);
    param1[4] = param_machine(35.0, 13.0);

    param1[5] = param_machine(18.0, 14.0);
    param1[6] = param_machine(56.0, 9.0);
    param1[7] = param_machine(94.0, 12.0);
    param1[8] = param_machine(68.0, 7.0);
    param1[9] = param_machine(34.0, 13.0);

    param1[10] = param_machine(63.0, 9.0);
    param1[11] = param_machine(74.0, 13.0);
    param1[12] = param_machine(17.0, 8.0);
    param1[13] = param_machine(83.0, 12.0);
    param1[14] = param_machine(46.0, 11.0);

    // best is 3
    std::vector<param_machine> param2(num_elem_in_group[2]);
    param2[0] = param_machine(53.0, 12.0);
    param2[1] = param_machine(24.0, 13.0);
    param2[2] = param_machine(58.0, 10.0);
    param2[3] = param_machine(84.0, 8.0);
    param2[4] = param_machine(68.0, 9.0);

    param2[5] = param_machine(58.0, 8.0);
    param2[6] = param_machine(49.0, 13.0);
    param2[7] = param_machine(75.0, 7.0);
    param2[8] = param_machine(64.0, 7.0);
    param2[9] = param_machine(15.0, 12.0);

    for(int i = 0; i < num_elem_g0; ++i)
        for(int j = 0; j < num_elem_g1; ++j)
            for(int k = 0; k < num_elem_g2; ++k)
            {
                machines[i][j][k] = Machine(param0[i], param1[j], param2[k]);
            }

    max_outcome = 0;
    // test A FEW samples as priors
    for(int i = 0, count = 0; i < num_elem_in_group[0]; i+=3)
        for(int j = 0; j < num_elem_in_group[1]; j+=3)
            for(int k = 0; k < num_elem_in_group[2]; k+=3)
            {
                experiment(generator, count++, i, j, k);
            }

    std::vector<int> best_selections(3);
    // test N times
    for(int i = 0; i < N; ++i)
    {
        // thompson sampling to choose trial button
        int next_id_g1 = choose_next_sample_id(generator, param0);
        int next_id_g2 = choose_next_sample_id(generator, param1);
        int next_id_g3 = choose_next_sample_id(generator, param2);

        // evaluate the sample
        experiment(generator, i, next_id_g1, next_id_g2, next_id_g3);

        // update the posterior: optimization should be [1,7,3]
        best_selections[0] = find_most_probable_machines(param0);
        best_selections[1] = find_most_probable_machines(param1);
        best_selections[2] = find_most_probable_machines(param2);

        std::cout << ", current best is: ["
        << best_selections[0] << ","
        << best_selections[1] << ","
        << best_selections[2] << "]";
    }

    print_group_summary(param0);
    print_group_summary(param1);
    print_group_summary(param2);

    return 0;
}