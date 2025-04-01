import json
import numpy as np
import pandas as pd
from datetime import timedelta


def load_json_data(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return None


def seconds_to_hms(seconds):
    """Convert seconds to hours:minutes:seconds format."""
    return str(timedelta(seconds=seconds)).split('.')[0]


def calculate_statistics(data):
    """Calculate statistics from the video data."""
    # Extract data into lists
    durations = []
    brightness_means = []
    brightness_stds = []
    brightness_temp_vars = []
    histogram_entropies = []

    for title, video in data.items():
        durations.append(float(video['duration']))
        brightness_means.append(float(video['brightness_mean']))
        brightness_stds.append(float(video['brightness_std']))
        brightness_temp_vars.append(float(video['brightness_temporal_variation']))
        histogram_entropies.append(float(video['histogram_entropy']))

    # Calculate statistics
    total_duration = sum(durations)
    avg_duration = np.mean(durations)
    max_duration = max(durations)
    min_duration = min(durations)

    stats = {
        "Total videos": len(data),
        "Total duration": seconds_to_hms(total_duration),
        "Average duration": seconds_to_hms(avg_duration),
        "Longest video": seconds_to_hms(max_duration),
        "Shortest video": seconds_to_hms(min_duration),
        "Brightness statistics": {
            "Average brightness": np.mean(brightness_means),
            "Std deviation": np.std(brightness_means),
            "Min brightness": min(brightness_means),
            "Max brightness": max(brightness_means),
        },
        "Temporal variation statistics": {
            "Average variation": np.mean(brightness_temp_vars),
            "Std deviation": np.std(brightness_temp_vars),
            "Min variation": min(brightness_temp_vars),
            "Max variation": max(brightness_temp_vars),
        },
        "Histogram entropy statistics": {
            "Average entropy": np.mean(histogram_entropies),
            "Std deviation": np.std(histogram_entropies),
            "Min entropy": min(histogram_entropies),
            "Max entropy": max(histogram_entropies),
        }
    }

    return stats


def find_extremes(data):
    """Find videos with extreme values."""
    brightest = max(data.items(), key=lambda x: float(x[1]['brightness_mean']))
    darkest = min(data.items(), key=lambda x: float(x[1]['brightness_mean']))
    most_varied = max(data.items(), key=lambda x: float(x[1]['brightness_temporal_variation']))
    least_varied = min(data.items(), key=lambda x: float(x[1]['brightness_temporal_variation']))
    longest = max(data.items(), key=lambda x: float(x[1]['duration']))
    shortest = min(data.items(), key=lambda x: float(x[1]['duration']))

    extremes = {
        "Brightest video": {
            "title": brightest[0],
            "brightness": float(brightest[1]['brightness_mean'])
        },
        "Darkest video": {
            "title": darkest[0],
            "brightness": float(darkest[1]['brightness_mean'])
        },
        "Most varied video": {
            "title": most_varied[0],
            "variation": float(most_varied[1]['brightness_temporal_variation'])
        },
        "Least varied video": {
            "title": least_varied[0],
            "variation": float(least_varied[1]['brightness_temporal_variation'])
        },
        "Longest video": {
            "title": longest[0],
            "duration": seconds_to_hms(float(longest[1]['duration']))
        },
        "Shortest video": {
            "title": shortest[0],
            "duration": seconds_to_hms(float(shortest[1]['duration']))
        }
    }

    return extremes


def create_dataframe(data):
    """Create a pandas DataFrame from the video data."""
    df = pd.DataFrame()

    for title, video in data.items():
        video_data = {
            'title': title,
            'duration': float(video['duration']),
            'duration_hms': seconds_to_hms(float(video['duration'])),
            'brightness_mean': float(video['brightness_mean']),
            'brightness_std': float(video['brightness_std']),
            'brightness_temporal_variation': float(video['brightness_temporal_variation']),
            'histogram_entropy': float(video['histogram_entropy']),
            'histogram_dynamic_range': float(video['histogram_dynamic_range']),
            'histogram_modality': int(video['histogram_modality'])
        }
        df = pd.concat([df, pd.DataFrame([video_data])], ignore_index=True)

    return df


def main(data_path):
    """Main function."""
    # Load data
    data = load_json_data(data_path)
    if not data:
        return

    # Calculate statistics
    stats = calculate_statistics(data)
    extremes = find_extremes(data)
    with open(data_path.replace(".json", "_statistics.json"), mode="w") as f:
        json.dump(stats, f, indent=2)
    # Print results
    print("\n===== TRAIN VIDEO STATISTICS =====")
    print(f"Total number of videos: {stats['Total videos']}")
    print(f"Total duration: {stats['Total duration']}")
    print(f"Average duration: {stats['Average duration']}")

    print("\n----- BRIGHTNESS STATISTICS -----")
    print(f"Average brightness: {stats['Brightness statistics']['Average brightness']:.2f}")
    print(f"Standard deviation: {stats['Brightness statistics']['Std deviation']:.2f}")
    print(
        f"Range: {stats['Brightness statistics']['Min brightness']:.2f} - {stats['Brightness statistics']['Max brightness']:.2f}")

    print("\n----- TEMPORAL VARIATION STATISTICS -----")
    print(f"Average variation: {stats['Temporal variation statistics']['Average variation']:.2f}")
    print(f"Standard deviation: {stats['Temporal variation statistics']['Std deviation']:.2f}")
    print(
        f"Range: {stats['Temporal variation statistics']['Min variation']:.2f} - {stats['Temporal variation statistics']['Max variation']:.2f}")

    print("\n----- HISTOGRAM ENTROPY STATISTICS -----")
    print(f"Average entropy: {stats['Histogram entropy statistics']['Average entropy']:.2f}")
    print(f"Standard deviation: {stats['Histogram entropy statistics']['Std deviation']:.2f}")
    print(
        f"Range: {stats['Histogram entropy statistics']['Min entropy']:.2f} - {stats['Histogram entropy statistics']['Max entropy']:.2f}")

    print("\n----- EXTREME VALUES -----")
    print(f"Brightest video: {extremes['Brightest video']['title']} ({extremes['Brightest video']['brightness']:.2f})")
    print(f"Darkest video: {extremes['Darkest video']['title']} ({extremes['Darkest video']['brightness']:.2f})")
    print(
        f"Most varied video: {extremes['Most varied video']['title']} ({extremes['Most varied video']['variation']:.2f})")
    print(
        f"Least varied video: {extremes['Least varied video']['title']} ({extremes['Least varied video']['variation']:.2f})")
    print(f"Longest video: {extremes['Longest video']['title']} ({extremes['Longest video']['duration']})")
    print(f"Shortest video: {extremes['Shortest video']['title']} ({extremes['Shortest video']['duration']})")

    # Create and save DataFrame
    df = create_dataframe(data)
    print("\n----- DATA FRAME SAMPLE -----")
    print(df.head())

    # Save DataFrame to CSV
    df.to_csv('train_video_stats.csv', index=False)
    print("\nData saved to 'train_video_stats.csv'")


if __name__ == "__main__":
    main('../railway_datasets/video_expl_analysis/parnici.json')
    # with open("../railway_datasets/video_expl_analysis/parnici_metadata.json", mode="w") as f:
    #     json.dump(all_metadata, f, indent=2)
    main('../railway_datasets/video_expl_analysis/cabview.json')
    main('../railway_datasets/video_expl_analysis/cabview_outliers.json')
