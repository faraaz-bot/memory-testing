#include <iostream>
#include <vector>
#include "gtest/gtest.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

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
    
    po::options_description opdesc("asan sample command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("seed", po::value<int>(&random_seed), "random seed");

    po::variables_map vm;
    po::parsed_options parsed =
        po::command_line_parser(argc, argv).options(opdesc).allow_unregistered().run(); 
    po::store(parsed, vm);
    po::notify(vm);

    
    if (!vm.count("seed"))  
    {
        std::cout << "Seed was not specified, so we can generate a random one.\n";
        // https://xkcd.com/221/
        random_seed = 4;
    }
    
    std::cout << "random seed: " << random_seed << "\n";
    
    std::vector<std::string> remaining_args = po::collect_unrecognized(parsed.options,
                                                                       po::include_positional);
    
    std::cout << "remaining args\n";
    for(const auto & var : remaining_args)
        std::cout << "\t" << var << "\n";
    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

    // The command-line parser expects the first argument to be the
    // program name, so put it back:
    remaining_args.insert(remaining_args.begin(), argv0);
    
    std::vector<char*> carg;
    std::transform(std::begin(remaining_args), std::end(remaining_args),
                   std::back_inserter(carg),
                   [](std::string& s){ s.push_back(0); return &s[0]; });
    decltype(argc) cargc = carg.size();
    
    ::testing::InitGoogleTest(&cargc, carg.data());
    
    std::cout << "carg\n";
    for(int i = 0; i < cargc; ++i) {
        // NB: gtest may hanve changed the value of cargc, so we can't
        // just loop over the entire std::vector.
        std::cout << "\t" << carg[i] << "\n";
    }
    std::cout << std::flush;
    
    opdesc.add_options()("postarg", po::value<int>(&postarg), "random seed");
    store(parse_command_line(cargc, (const char **)carg.data(), opdesc), vm);
    po::notify(vm);
   
    std::cout << "postarg: " << postarg << "\n";
    std::cout << "random seed: " << random_seed << "\n";
    
    if (!vm.count("seed"))  
    {
        std::cout << "seed is still not set\n";
    }

    
    return RUN_ALL_TESTS();
}
