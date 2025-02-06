#!/bin/bash

# Define arrays for each parameter
models=(
    "yolov10x.pt"
    "yolov10m.pt"
    "yolov10n.pt"
    "yolov5x.pt"
    "yolov5mu.pt"
    "yolov5nu.pt"
)

freeze_values=(0 3 5 10)
epoch_values=(40 50 60)
conf_thres=(0.3 0.5 0.8)
data="CRTL_multi_4_labeled.yaml"


# Counter for total jobs
total_jobs=0

# Perform grid search
for model in "${models[@]}"; do
    for freeze in "${freeze_values[@]}"; do
        for epochs in "${epoch_values[@]}"; do
            for conf in "${conf_thres[@]}"; do
                # Generate timestamp for this specific run
                thistime=${epochs}_${freeze}_${model}_${conf}
                # Construct the qsub command
                qsub -v model=${model},freeze=${freeze},epochs=${epochs},thistime=${thistime},data=${data},confthres=${conf} ./train_on_meta.sh


                # Increment counter
                ((total_jobs++))

                # Optional: Add a small delay to prevent overwhelming the queue system
                sleep 0.1
            done
        done
    done
done

# Print summary to console
echo "Grid search completed!"
echo "Total jobs submitted: $total_jobs"


