
qsub -v model="yolov10x.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh

qsub -v model="yolov10x.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh

qsub -v model="yolov10x.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh


qsub -v model="yolov10x.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled.yaml" ./train_on_meta.sh


# transfer learning

qsub -v model="yolov10x.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=10,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh

qsub -v model="yolov10x.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=20,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh

qsub -v model="yolov10x.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=40,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh


qsub -v model="yolov10x.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10m.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10s.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
qsub -v model="yolov10n.pt",epochs=60,thistime=$(date '+%Y_%m_%d_%H_%M'),data="CzechRailwayTrafficLights_multi_labeled_transfer.yaml" ./train_on_meta.sh
