#!/bin/bash

make -C /home/flteixei/data/Dev/partial_pass/build

# ===================================================================
cd /home/flteixei/data/Dev/partial_pass/build/bin/
echo "Timing: CS_3D_RC 64x64x64" 
./partial-pass-d 1 0 0 
echo "Accuracy vs Octave: CS_3D_RC 64x64x64" 
./partial-pass-d 0 0 0 > /home/flteixei/matlab/rocfft_output_data.txt
cd /home/flteixei/matlab/
./rocfft_to_matlab.sh 0 rocfft_output_data.txt
cd /home/flteixei/data/Dev/partial_pass/build/bin/
./partial-pass-d 0 1 0 > /home/flteixei/matlab/rocfft_input_data.txt
cd /home/flteixei/matlab/
./rocfft_to_matlab.sh 1 rocfft_input_data.txt
octave-cli partial_pass_test.m
# ===================================================================
cd /home/flteixei/data/Dev/partial_pass/build/bin/
echo "Timing: CS_3D_RC 64x64x64 with partial pass" 
./partial-pass-d 1 0 1 
echo "Accuracy vs Octave: CS_3D_RC 64x64x64 with partial pass" 
./partial-pass-d 0 0 1 > /home/flteixei/matlab/rocfft_output_data.txt
cd /home/flteixei/matlab/
./rocfft_to_matlab.sh 0 rocfft_output_data.txt
cd /home/flteixei/data/Dev/partial_pass/build/bin/
./partial-pass-d 0 1 1 > /home/flteixei/matlab/rocfft_input_data.txt
cd /home/flteixei/matlab/
./rocfft_to_matlab.sh 1 rocfft_input_data.txt
octave-cli partial_pass_test.m
# ===================================================================