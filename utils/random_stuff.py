import json

from general_utils import normalize_text, remove_annotated_duplicates


def ad_second(temp):
    temp1 = [int(j) - 1 for j in temp[i]]
    temp1.extend([int(j) + 1 for j in temp[i]])
    temp1.extend([int(j) for j in temp[i]])
    return temp1


def load_class(json_path="../railway_datasets/simple_classes/stop.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    d = d["data"]
    dataset = {}
    for i in d:
        try:
            dataset[normalize_text((i["video name"]))].append(i["timestamp in video"])
        except KeyError:
            dataset[normalize_text((i["video name"]))] = [i["timestamp in video"]]
    return dataset


remove_annotated_duplicates("C:/Users/dschn/PycharmProjects/czech-railway-trafic-lights-detection/railway_datasets/simple_classes/stop.json")

# stops = load_class()
# today_results = load_class("../dataset/reconstructed/today_results.json")
#
# print("----- diffs -----")
# print(set(today_results.keys()).difference(
#     set(stops.keys()) ))
# print()
# print(set(stops.keys()).difference(
#     set(today_results.keys()) ))
# print("-----------------")
#
# for i in set(today_results.keys()).intersection(
#     set(stops.keys()) ):
#     print(i)
#     same_times = list(set(ad_second(today_results)).intersection(set(ad_second(stops))))
#     print(same_times)
#     # print([same_times[j-1] for j in range(0, len(same_times), 2) if same_times[j] - same_times[j-1] > 1])
#     print("------------")
#

