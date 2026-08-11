import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default="best_human_thermal.pt")
parser.add_argument("--imgsz", type=int, default=640)
args, unknown = parser.parse_known_args()

model = YOLO(args.weights)
export_path = model.export(format="ncnn", imgsz=args.imgsz)
print(f"[export_ncnn.py] Exported to: {export_path}")
print("[export_ncnn.py] Copy this whole folder to the Raspberry Pi "
      "(e.g. via scp) and point pi_app.py at it.")
