#include <iostream>
#include <vector>
#include <random>

// compile: gcc bayesian.cpp -lm -lstdc++ -o bayesian.o
// run: ./bayesian.o num-samples
// example: ./bayesian.o 100

std::vector<int> counts;
std::vector<double> alpha;
std::vector<double> beta;

class normal_dist_machine
{
    double mean;
    double deviation;
    std::mt19937 gen;
    std::normal_distribution<double> dist;

public:
    normal_dist_machine() = default;
    normal_dist_machine(double m, double d)
    : mean(m), deviation(d)
    {
        std::random_device rd;
        gen = std::mt19937(rd());
        dist = std::normal_distribution<double>(mean, deviation);
    }

    double sample()
    {
        return dist(gen);
    }
};

// single outcome = [1,100]
void update_posterior_params(int test_counter, int button_id, double outcome)
{
    counts[button_id]++;
    alpha[button_id] += outcome;
    beta[button_id] += 100.0 - outcome;

    auto prob = alpha[button_id] / (alpha[0] + beta[0]);
    std::cout << "test " << test_counter << "\t button " << button_id
              << ": outcome is " << outcome << "\t posterior prob is "
              <<  prob << "\t";
}

int find_most_probable_button(int total_btn)
{
    int max_id = 0;
    double max_prob = alpha[0] / (alpha[0] + beta[0]);

    for(int i = 1; i < total_btn; ++i)
    {
        double prob = alpha[i] / (alpha[i] + beta[i]);
        if(prob > max_prob)
        {
            max_id = i;
            max_prob = prob;
        }
    }

    return max_id;
}

int main(int argc, char* argv[])
{
    int N = 50;
    if(argc >= 2)
        N = std::stoi(argv[1]);

    counts = std::vector<int>(N, 0);
    alpha = std::vector<double>(N, 1.0);
    beta = std::vector<double>(N, 1.0);

    std::vector<normal_dist_machine> buttons(5);
    buttons[0] = normal_dist_machine(80.0, 12.0);
    buttons[1] = normal_dist_machine(50.0, 24.0);
    buttons[2] = normal_dist_machine(83.0, 6.0);
    buttons[3] = normal_dist_machine(70.0, 20.0);
    buttons[4] = normal_dist_machine(60.0, 6.0);

    // test N times
    for(int i = 0; i < N; ++i)
    {
        // from i to button id
        int button_id = i % buttons.size();
        double outcome = buttons[button_id].sample();
        update_posterior_params(i, button_id, outcome);

        // current most probable result:
        int most_probable_button = find_most_probable_button(N);
        std::cout << "most_probable_button is: " << most_probable_button << std::endl;
    }

    return 0;
}