

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# # Train the model with 2 GPUs
# results = model.train(data="CzechRailwayTrafficLights.yaml", epochs=100, imgsz=640,
#                       # device="mps"   # this is M1 apple device
#                                                                        )

model.train(data="./CzechRailwayTrafficLights.yaml", epochs=1, imgsz=1280, device="cpu"
                      # device="mps"   # this is M1 apple device
                                                                       )
print("results")
