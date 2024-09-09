#!/bin/bash

# Usage: gen_pmlog.sh log_file_name "your_workload"

echo "Starting..."
echo "PMLog file name: $1"

sudo agt_internal -i=5 -pmperiod=50 -pmlogall -pmstopcheck -pmnoesckey -pmoutput=$1 2>&1 &
eval $2
touch terminate.txt
stty sane
