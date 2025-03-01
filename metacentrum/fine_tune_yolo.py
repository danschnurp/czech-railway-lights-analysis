from torch import cuda, version
import argparse
from pathlib import Path
import yaml
import torch



def control_torch():
    """
    test if cuda is available
    """
    print("cuda availability: " + str(cuda.is_available()))
    import gc
    gc.collect()
    if not cuda.is_available():
        return "cpu"
    print("version: " + version.cuda)
    cuda.empty_cache()
    # Storing ID of current CUDA device
    cuda_id = cuda.current_device()
    print(f"ID of current CUDA device:{cuda_id}")
    print(f"Name of current CUDA device:{cuda.get_device_name(cuda_id)}")
    return "cuda"


workdir = "./"

parser = argparse.ArgumentParser(description='YOLO Training and Validation Arguments')
# Dataset parameters
parser.add_argument('--data', type=str, default=f"{workdir}CzechRailwayTrafficLights_multi_labeled.yaml",
                    help='Path to data.yaml file')
parser.add_argument('--img-size', type=int, default=[1920, 1080], help='Training image size (pixels)')
# Model parameters
parser.add_argument('--model', type=str, default="yolov10m.pt", help='Initial weights path')
parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
# Training parameters
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
parser.add_argument('--device', default=control_torch(), help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# Optimization parameters
# architecture https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models yamls
parser.add_argument('--freeze', default=0, help='Freezes the first N layers of the model or specified layers '
                                                'by index')
parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer (SGD, Adam, AdamW)')
parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay coefficient')
# Logging parameters
parser.add_argument('--project', default='runs/train', help='Project name')
parser.add_argument('--name', default='exp', help='Experiment name')
parser.add_argument('--exist-ok', action='store_true', help='Allow existing project')
parser.add_argument('--save-period', type=int, default=100, help='Save checkpoint every x epochs')
# Validation parameters
parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')  # todo
parser.add_argument('--iou-thres', type=float, default=0.1, help='Non-Maximum Suppression IoU Intersection over '
                                                                 'Union threshold')  #
# Augmentation parameters
parser.add_argument('--hsv-h', type=float, default=0, help='HSV-Hue augmentation')
parser.add_argument('--hsv-s', type=float, default=0, help='HSV-Saturation augmentation')
parser.add_argument('--hsv-v', type=float, default=0, help='HSV-Value augmentation')
parser.add_argument('--degrees', type=float, default=0.0, help='Rotation augmentation')  # todo
parser.add_argument('--translate', type=float, default=0.00, help='Translation augmentation')
parser.add_argument('--scale', type=float, default=0.0, help='Scale augmentation')
parser.add_argument('--shear', type=float, default=0.0, help='Shear augmentation')
args = parser.parse_args()

if args.model.find("yolov5") != -1:
    from ultralytics import YOLO as YOLOv10
else:
    from ultralytics import YOLOv10 as YOLOv10


# # Create output directories
# args.save_dir = Path(args.project) / args.name
# args.save_dir.mkdir(parents=True, exist_ok=args.exist_ok)



if args.resume:
        model = YOLOv10(args.model, 2)
else:
        model = YOLOv10(args.model, 2)

# Prepare training arguments
train_args = {
        # Dataset parameters
        'data': args.data,
        'imgsz': args.img_size,
        # Training parameters
        'epochs': args.epochs,
        'batch': args.batch_size,
        'workers': args.workers,
        'device': args.device if args.device else None,
        # Optimization parameters
        'freeze': args.freeze,
        'optimizer': args.optimizer,
        'lr0': args.lr0,  # initial learning rate
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        # Logging parameters
        'val': True,
        'plots': True,  # save plots
        'save': False,  # save checkpoints
        'save_period': args.save_period,
        'project': args.project,
        'name': f"{args.epochs}_{args.model.replace('.', '')}",
        'exist_ok': args.exist_ok,
        # Validation parameters
        'conf': args.conf_thres,
        'iou': args.iou_thres,
        # Augmentation parameters
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
    }

results = model.train(**train_args)







