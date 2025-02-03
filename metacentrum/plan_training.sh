#!/bin/bash

# Define arrays for each parameter
models=(
    "yolov10x.pt"
    "yolov10m.pt"
    "yolov10s.pt"
    "yolov10n.pt"
    "yolov10b.pt"
    "yolov5x.pt"
    "yolov5mu.pt"
    "yolov10s.pt"
    "yolov5nu.pt"
)

freeze_values=(0 3 5)
epoch_values=(5 10 20 40 60)
conf_thres=(0.3 0.5 0.8)
data="CRTL_multi_labeled.yaml"


# Counter for total jobs
total_jobs=0

# Perform grid search
for model in "${models[@]}"; do
    for freeze in "${freeze_values[@]}"; do
        for epochs in "${epoch_values[@]}"; do
            for conf in "${conf_thres[@]}"; do
                # Generate timestamp for this specific run
                thistime=$(date '+%Y_%m_%d_%H_%M')

                # Construct the qsub command
                qsub -v model=${model},freeze=${freeze},epochs=${epochs},thistime=${thistime},data=${data},confthres=${conf} ./train_on_meta.sh


                # Increment counter
                ((total_jobs++))

                # Optional: Add a small delay to prevent overwhelming the queue system
                sleep 1
            done
        done
    done
done

# Print summary to console
echo "Grid search completed!"
echo "Total jobs submitted: $total_jobs"


