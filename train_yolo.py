import argparse

from ultralytics import YOLO


parser = argparse.ArgumentParser(description='training yolo')
parser.add_argument('-m', "--model", default="./yolov5mu.pt")
parser.add_argument('-e', '--epochs', type=int, default=10)
args = parser.parse_args()


# Load a model
model = YOLO(args.model)

# # Train the model with 2 GPUs
# results = model.train(data="CzechRailwayTrafficLights.yaml", epochs=100, imgsz=640,
#                       # device="mps"   # this is M1 apple device
#                                                                        )

model.train(data="./CzechRailwayTrafficLights.yaml", epochs=args.epochs, imgsz=1280, device="cuda", batch=8)

model.export()
