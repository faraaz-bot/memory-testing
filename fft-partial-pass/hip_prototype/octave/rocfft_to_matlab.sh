#! /bin/bash

#usage /.rocfft_to_matlab.sh $arg $file 
# arg=0 (input) arg=1 (output)

if [ $1 -eq 1 ]; then
    filename="rocfft_input_data.m"
elif [ $1 -eq 0 ]; then
    filename="rocfft_output_data.m"
else
    echo "error"
fi

sed '1i data=[' $2 | sponge $filename
sed 's/(//g' $filename | sponge $filename
sed 's/,/+/g' $filename | sponge $filename

tr '\n' ' ' < $filename | sponge $filename

sed 's/)/i;\n/g' $filename | sponge $filename

sed '$a];' $filename | sponge $filename

if [ $1 -eq 1 ]; then
    sed -i "1s/^/function data = rocfft_input_data()\n/" $filename
elif [ $1 -eq 0 ]; then
    sed -i "1s/^/function data = rocfft_output_data()\n/" $filename
fi

rm $2
