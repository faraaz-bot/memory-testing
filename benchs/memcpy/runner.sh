#!/bin/bash

# Run this from memcpy directory
# Runs membench for varying parameters, graphing data for weak and strong scaling

mkdir strong_scaling # Same N, varying ngpus
mkdir weak_scaling   # N scales alongside ngpus

ntrials = 10
benchmarks = "all"

# Build it!

# Setup python env
# python3 -m venv env
# source ./env/bin/activate
# pip install -r visualizer_py_requirements.txt

# Strong scaling tests - graph per each N value
for n in {512,1024,2048,4096,8192,16384}
do
    mkdir ./strong_scaling/${n}
    for g in {1,2,4,8}
    do
        ./build/membench -n ${n} -g ${g} -t ${ntrials} -r ${benchmarks} --benchmark_format=csv > ./strong_scaling/${n}/output.csv
        # python3 visualizer.py -i ./strong_scaling/${n}/output.csv # Note: this probably dumps png into cwd for now
    done
done

# Weak scaling tests - aggregate everything into one output graph
for n in {512,1024,2048,4096,8192,16384}
do
    for g in {1,2,4,8}
    do
        ./build/membench -n ${n} -g ${g} -t ${ntrials} -r ${benchmarks} --benchmark_format=csv > ./weak_scaling/output-n${n}-g${g}.csv
    done
done
# python3 visualizer.py -i ./weak_scaling/output-n${n}-g${g}.csv # Note: this probably dumps png into cwd for now
