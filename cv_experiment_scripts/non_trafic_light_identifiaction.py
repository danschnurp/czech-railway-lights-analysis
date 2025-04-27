import json
import numpy as np
from typing import Dict, List, Union, Tuple

from general_utils import get_times_by_video_name, save_json_unicode, print_statistics


def remove_close_numbers(arr: List[float], threshold: float) -> List[float]:
    """
    Remove numbers from array that are closer than the specified threshold.
    Keeps the first occurrence and removes subsequent close numbers.

    Args:
        arr: Input array of numbers
        threshold: Minimum allowed distance between numbers

    Returns:
        List with close numbers removed
    """
    if not arr:
        return []

    # Sort the array first
    sorted_arr = sorted(arr)
    result = [sorted_arr[0]]  # Keep the first number

    # Compare adjacent numbers
    for i in range(1, len(sorted_arr)):
        if abs(sorted_arr[i] - result[-1]) > threshold:
            result.append(sorted_arr[i])

    # Return numbers in their original order
    return [x for x in arr if x in result]


def compare_arrays(arr1: List[float], arr2: List[float]) -> Dict[str, Union[bool, float, str]]:
    """
    Compare two arrays and return statistics about their differences.

    Args:
        arr1: First array
        arr2: Second array

    Returns:
        Dictionary containing comparison statistics
    """


    differences = np.array(list(set(arr1) - set(arr2)))
    differences = remove_close_numbers(list(differences), 10)
    return {

        "differences len": len(differences),
        "equal_length": True,
        "length": len(arr1),
        "mean_diff": float(np.mean(differences)),
        "max_diff": float(np.max(differences)),
        "min_diff": float(np.min(differences)),
        "std_diff": float(np.std(differences)),
        "differences": differences,
    }


def compare_jsons(json1: Dict[str, List[float]], json2: Dict[str, List[float]]) -> Dict[str, Dict]:
    """
    Compare two JSON objects containing arrays.

    Args:
        json1: First JSON object
        json2: Second JSON object

    Returns:
        Dictionary containing comparison results for each key
    """
    all_keys = set(json1.keys()) | set(json2.keys())
    comparison = {}

    for key in all_keys:
        if key not in json1:
            comparison[key] = {"status": "missing_in_json1"}
        elif key not in json2:
            comparison[key] = {"status": "missing_in_json2"}
        else:
            comparison[key] = {
                "status": "compared",
                "comparison": compare_arrays(json1[key], json2[key])
            }

    return comparison


def format_comparison_results(results: Dict[str, Dict]) -> str:
    """
    Format comparison results into a readable string.

    Args:
        results: Comparison results dictionary

    Returns:
        Formatted string representation of results
    """
    output = []

    for key, data in results.items():
        output.append(f"\nKey: {key}")

        if data["status"] in ["missing_in_json1", "missing_in_json2"]:
            output.append(f"Status: {data['status']}")
            continue

        comp = data["comparison"]

        output.append(
                f"Arrays comparison \n"
                f" - differences len: {comp['differences len']}\n"
                f"  - Mean difference: {comp['mean_diff']:.4f}\n"
                f"  - Max difference: {comp['max_diff']:.4f}\n"
                f"  - Min difference: {comp['min_diff']:.4f}\n"
                f"  - Standard deviation of differences: {comp['std_diff']:.4f}\n\n\n"
                f" - differences: {comp['differences']}\n\n\n"
            )

    return "\n".join(output)


def main():

    traffic_lights_raw = get_times_by_video_name("../railway_datasets/annotated_traffic_lights.json", reverse=True)

    annotated_traffic_lights = get_times_by_video_name("../railway_datasets/traffic_lights_raw.json", reverse=True)

    # Compare the JSONs
    results = compare_jsons(annotated_traffic_lights, traffic_lights_raw)

    # Print formatted results
    print(format_comparison_results(results))
    final_results = {}
    for result in results:
        if results[result]["status"] in ["missing_in_json1", "missing_in_json2"]:
            continue
        final_results[result] = results[result]["comparison"]["differences"]

    save_json_unicode(final_results, filename="./today_results.json")


    print_statistics()

if __name__ == "__main__":
    main()