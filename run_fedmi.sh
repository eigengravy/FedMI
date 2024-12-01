#!/bin/bash

# Define all loss options
losses=("ce" "remine_a0.1_b0" "reinfonce_a0.1_b0" "resmile_t10_a0.1_b0" "renwj_t10_a0.1" \
        "retuba_t10_a0.1" "rejs_a0.1_b1" "renwjjs_a0.1_b1" "remine_a0.01_b0" \
        "reinfonce_a0.01_b0" "resmile_t10_a0.01_b0" "renwj_t10_a0.01" "retuba_t10_a0.01" \
        "rejs_a0.01_b1" "renwjjs_a0.01_b1" "remine_a0.001_b0" "reinfonce_a0.001_b0" \
        "resmile_t10_a0.001_b0" "renwj_t10_a0.001" "retuba_t10_a0.001" "rejs_a0.001_b1" \
        "renwjjs_a0.001_b1")

additional_args="$@"

# Loop through each loss function and run the script
for loss in "${losses[@]}"; do
    echo "Running with --loss=$loss $additional_args"
    python fedmi.py --loss "$loss" $additional_args
done

