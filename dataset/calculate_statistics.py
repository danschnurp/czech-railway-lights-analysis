from general_utils import print_statistics

print("------------------------")
print("traffic_lights_raw predicted by yolo")
all_yolo = print_statistics("../railway_datasets/traffic_lights_raw.json")
print("\ntrue predicted by yolo and checked by human")
all_yolo_human_human = print_statistics("../railway_datasets/annotated_traffic_lights.json")
print("")
print(f"\t \t \t \t \t \t \t \t \t  acc: \t {all_yolo_human_human / all_yolo:0.2f}")
print("------------------------")
