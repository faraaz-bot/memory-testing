cmake_minimum_required( VERSION 3.21 )

project( argv LANGUAGES CXX  )
set( CMAKE_CXX_STANDARD 14 )

find_package(GTest REQUIRED)

add_executable( carg carg.cpp )
target_link_libraries(
  carg
  GTest::gtest_main
)

add_executable( barg barg.cpp )
find_package( Boost COMPONENTS program_options REQUIRED)
target_link_libraries(
  barg
  GTest::gtest_main
  Boost::program_options
)

add_executable( garg garg.cpp )
target_link_libraries(
  garg
  GTest::gtest_main
)
