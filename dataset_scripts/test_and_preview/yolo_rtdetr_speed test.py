from ultralytics import RTDETR, YOLO



# Load a COCO-pretrained RT-DETR-l model
model = YOLO("../../experiment_results/yolo/CRL_extended_v2/60_lr001_0_yolov8n.pt_0.5/weights/best.pt")

# Display model information (optional)
model.info()



# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/reconstructed/czech_railway_light_dataset/val/images/multi_class/15.jpg")

# print(results)

print("-------------------------------")

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("../../experiment_results/yolo/CRL_extended_v2/60_lr001_0_rtdetr-l.pt_0.5/weights/best.pt")

# Display model information (optional)
model.info()


# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/reconstructed/czech_railway_light_dataset/val/images/multi_class/15.jpg")

# print(results)




