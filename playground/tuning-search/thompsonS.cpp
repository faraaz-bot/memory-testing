#include <iostream>
#include <vector>
#include <random>

// compile: gcc thompsonS.cpp -lm -lstdc++ -o thompsonS.o
// run: ./thompsonS.o num-samples
// example: ./thompsonS.o 100

std::vector<int> counts;
std::vector<double> success; // alpha = success + 1
std::vector<double> failure; // beta = failure + 1

double max_outcome;

const int num_buttons = 5;

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

// thompson sampling
int choose_button()
{
    std::vector<double> theta(num_buttons, 0.0);

    // a random generator
    std::random_device rd;
    std::default_random_engine generator(rd());

    // sample beta distribution of each button
    for(int i = 0; i < num_buttons; ++i)
    {
        std::gamma_distribution<double> gamma(success[i], failure[i]);
        theta[i] = gamma(generator);
    }

    // choose the button with largest theta value
    int choosen = 0;
    double max_theta = theta[0];
    for(int i = 1; i < num_buttons; ++i)
    {
        if(theta[i] > max_theta)
        {
            max_theta = theta[i];
            choosen = i;
        }
    }

    return choosen;
}

// update if the reward is success or not
// success: outcome is within 2% of the current largest outcome
void experiment(int test_counter, int button_id, double outcome)
{
    double ratio = outcome / max_outcome;
    bool good = ratio >= 0.94;

    if(good)
        success[button_id]++;
    else
        failure[button_id]++;

    if(outcome > max_outcome)
        max_outcome = outcome;

    auto prob = success[button_id] / (success[button_id] + failure[button_id]);
    std::cout << "\ntest " << test_counter << "\t button " << button_id
              << ": outcome is " << outcome << "\t " << (good ? "success" : "failure")
              << "\t posterior prob is " << prob << "\t";
}


int find_most_probable_button()
{
    int max_id = 0;
    double max_prob = success[0] / (success[0] + failure[0]);

    for(int i = 1; i < num_buttons; ++i)
    {
        double prob = success[i] / (success[i] + failure[i]);
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
    success = std::vector<double>(N, 1);
    failure = std::vector<double>(N, 1);

    std::vector<normal_dist_machine> buttons(num_buttons);
    buttons[0] = normal_dist_machine(80.0, 12.0);
    buttons[1] = normal_dist_machine(50.0, 24.0);
    buttons[2] = normal_dist_machine(83.0, 6.0);
    buttons[3] = normal_dist_machine(70.0, 20.0);
    buttons[4] = normal_dist_machine(60.0, 6.0);

    max_outcome = 0;

    for(int i = 0; i < num_buttons; ++i)
    {
        int button_id = i;
        double outcome = buttons[i].sample();
        experiment(i, button_id, outcome);
    }

    // test N times
    for(int i = 0; i < N; ++i)
    {
        // thompson sampling to choose trial button
        int button_id = choose_button();
        double outcome = buttons[button_id].sample();

        experiment(i, button_id, outcome);

        // current most probable result:
        int most_probable_button = find_most_probable_button();
        std::cout << "most_probable_button is: " << most_probable_button;
    }

    return 0;
}