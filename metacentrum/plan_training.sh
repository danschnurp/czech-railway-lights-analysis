qsub -v model="yolov10m.pt",epochs=30,thistime=$(date '+%Y_%m_%d_%H_%M') ./train_on_meta.sh
