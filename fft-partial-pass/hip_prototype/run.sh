#!/bin/bash

dir=$(pwd)

make -C $dir/build
# ===================================================================
# cd $dir/build/bin/
# echo "Timing: CS_3D_RC 64x64x64" 
# ./partial-pass-d 1 0 0 
# echo "Accuracy vs Octave: CS_3D_RC 64x64x64" 
# ./partial-pass-d 0 0 0 > $dir/octave/rocfft_output_data_1.txt
# cd $dir/octave
# ./rocfft_to_matlab.sh 0 rocfft_output_data_1.txt
# cd $dir/build/bin/
# ./partial-pass-d 0 1 0 > $dir/octave/rocfft_input_data.txt
# cd $dir/octave
# ./rocfft_to_matlab.sh 1 rocfft_input_data.txt
# # octave -W partial_pass_test.m
# ===================================================================
cd $dir/build/bin/
# echo "Timing: CS_3D_RC 64x64x64 with partial pass" 
# ./partial-pass-d 1 0 1 
echo "Accuracy vs Octave: CS_3D_RC 64x64x64 with partial pass" 
./partial-pass-d 0 0 1 > $dir/octave/rocfft_output_data_2.txt
cd $dir/octave
./rocfft_to_matlab.sh 0 rocfft_output_data_2.txt
cd $dir/build/bin/
./partial-pass-d 0 1 1 > $dir/octave/rocfft_input_data.txt
cd $dir/octave
./rocfft_to_matlab.sh 1 rocfft_input_data.txt
octave -W partial_pass_test.m
# ===================================================================