#include <iostream>
#include <vector>
#include "gtest/gtest.h"

#include "CLI11.hpp"

TEST(sample_test_case, sample_test)
{
    EXPECT_EQ(1, 1);
}

int random_seed{};
int postarg{};

int main(int argc, char **argv)
{
    std::cout << "argv:\n";
    for(int i = 0; i < argc; ++i) {
        std::cout << "\t" << argv[i] << "\n";
    }

    // Save the first name so that we can use it later:
    std::string argv0 = argv[0];
        
    CLI::App app{"cli11 app"};
    CLI::Option* opt_seed
        = app.add_option("--seed", random_seed, "Random seed; if unset, use an actual random seed");
    
    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }
    
    if(!*opt_seed)
    {
        std::cout << "Seed was not specified, so we can generate a random one.\n";
        // https://xkcd.com/221/
        random_seed = 4;
    }
    
    std::vector<std::string> remaining_args = app.remaining();
    // Google test ignores the first element, so add something there so that it parses all of hte
    // arguments that we want it to parse.:
    remaining_args.insert(remaining_args.begin(), argv0);
    // NB: If we initialize gtest first, then it removes all of its own command-line
    // arguments and sets argc and argv correctly;
    std::vector<char*> carg;
    for(std::string& s : remaining_args)
    {
        carg.push_back(&s[0]);
    }
    carg.push_back(NULL);
    decltype(argc) cargc = carg.size() - 1;
    ::testing::InitGoogleTest(&cargc, carg.data());

    
    app.add_option("--postarg", postarg, "postarg");
    try
    {
        app.parse(cargc, carg.data());
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    
    if(!*opt_seed)
    {
        std::cout << "seed is still not set\n";
    }

    std::cout << "postarg: " << postarg << "\n";
    std::cout << "random seed: " << random_seed << "\n";
   
    return RUN_ALL_TESTS();
}
