from utils.general_utils import print_statistics, get_jpg_files


print("------------------------")
print("traffic_lights_raw predicted by yolo")
all_yolo = print_statistics("../railway_datasets/traffic_lights_raw.json")
print("\ntrue predicted by yolo and checked by human")
all_yolo_human_human = print_statistics("../railway_datasets/annotated_traffic_lights.json")
print("")
print(f"\t \t \t \t \t \t \t \t \t  acc: \t {all_yolo_human_human / all_yolo:0.2f}")
print("------------------------")

print("yolov5m with movement detection:", len([i for i in get_jpg_files("../reconstructed/all_yolov5mu_raw") if i.find("box.jpg") != -1]), "moments")
print("yolov5m with movement detection:", len([i for i in get_jpg_files("../reconstructed/all_yolov5mu_raw") if i.find("roi") != -1]), "object")

print("yolov5m with movement detection filtered by human:", len([i for i in get_jpg_files("../reconstructed/all_yolov5mu") if i.find("roi") != -1]), "object")


print("yolov10m:",len([i for i in get_jpg_files("../reconstructed/DIP_unannontated") if i.find("clean.jpg") != -1 and i.find("yolov10m") != -1]))
print("yolov5m:",len([i for i in get_jpg_files("../reconstructed/DIP_unannontated") if i.find("clean.jpg") != -1 and i.find("yolov5") != -1]))
