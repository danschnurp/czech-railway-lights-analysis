## Reconstructing the dataset

- use script `reconstruct_dataset.py`
- script parameters:
  - **--nett_name** (default: yolov5mu.pt): Name of the pre-trained neural network model (default: yolov5mu.pt).
  - **--sequences_jsom_path** (default: ./traffic_lights.json): Path to a JSON file containing video sequences information (default: ./traffic_lights.json).
  - **--sequence_seconds_before** (default: 0.001): Number of seconds of inference to include before each timestamp (default: 0.001 seconds).
  - **--sequence_seconds_after** (default: 0.001): Number of seconds of inference to include after each timestamp (default: 0.001 seconds).
    - used for blinking states
  - **--clean_pictures** (default: True): Generate images without markings (original frames) (default: on).
  - **--bounding_box_pictures** (default: True): Generate images with bounding boxes around objects of interest (default: on).
  - **--roi_pictures** (default: True): Generate images containing only regions of interest (default: on).

parser.add_argument('--in-dir', default="/Volumes/zalohy/dip")
parser.add_argument('--out-dir', default="/Volumes/zalohy/dip")

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../colored_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.002)
parser.add_argument('--sequence_seconds_after', type=float, default=0.002)
parser.add_argument('--clean_pictures', default=False)
parser.add_argument('--bounding_box_pictures', default=False)
parser.add_argument('--work_dir', default="/Volumes/zalohy/dip")
parser.add_argument('--roi_pictures', default=True)

