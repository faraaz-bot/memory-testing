#include <iostream>
#include <vector>
#include "gtest/gtest.h"

#include "CLI11.hpp"

TEST(sample_test_case, sample_test)
{
    EXPECT_EQ(1, 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
   
    return RUN_ALL_TESTS();
}
