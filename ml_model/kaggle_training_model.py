%%writefile train.py

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml


def build_data_yaml(data_root: Path, out_path: Path, class_names):
    """
    Writes the data.yaml file YOLO needs, pointing at the train/test
    image folders inside the dataset you already have.
    """
    data_cfg = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "test/images", 
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)
    print(f"[train.py] Wrote data config to {out_path}")
    return out_path


def sanity_check_dataset(data_root: Path):
    required = [
        data_root / "train" / "images",
        data_root / "train" / "labels",
        data_root / "test" / "images",
        data_root / "test" / "labels",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[train.py] ERROR: expected folders not found:")
        for m in missing:
            print(f"   - {m}")
        print("[train.py] Point --data_root at the folder that directly "
              "contains 'train' and 'test' subfolders.")
        sys.exit(1)

    n_train_imgs = len(list((data_root / "train" / "images").glob("*")))
    n_train_lbls = len(list((data_root / "train" / "labels").glob("*.txt")))
    n_test_imgs = len(list((data_root / "test" / "images").glob("*")))
    print(f"[train.py] Found {n_train_imgs} train images, {n_train_lbls} train labels, "
          f"{n_test_imgs} test images.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for human detection on thermal images")
    parser.add_argument("--data_root", type=str, required=True,
                         help="Path to the 'dataset' folder containing train/ and test/")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Base checkpoint to fine-tune: yolov8n.pt (fast) up to yolov8x.pt (accurate)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="human_thermal",
                         help="Run name; results land in runs/detect/<name>/")
    parser.add_argument("--class_names", type=str, default="person",
                         help="Comma-separated class names, in label-index order. "
                              "Most single-class human-detection sets just use 'person'.")
    parser.add_argument("--patience", type=int, default=15,
                         help="Early stopping patience (epochs with no val improvement)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    sanity_check_dataset(data_root)

    class_names = [c.strip() for c in args.class_names.split(",")]

    data_yaml_path = Path("data.yaml")
    build_data_yaml(data_root, data_yaml_path, class_names)

    from ultralytics import YOLO

    print(f"[train.py] Loading base model {args.model} ...")
    model = YOLO(args.model)

    print("[train.py] Starting training ...")
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        name=args.name,
        plots=True,
    )

    print("[train.py] Running validation on test set ...")
    metrics = model.val(data=str(data_yaml_path), split="val")
    print(metrics)

    best_weights = Path("runs") / "detect" / args.name / "weights" / "best.pt"
    print(f"\n[train.py] DONE. Best weights saved at: {best_weights}")
    print("[train.py] Download this file — you'll load it in app.py for inference.")

    export_path = Path("best_human_thermal.pt")
    if best_weights.exists():
        shutil.copy(best_weights, export_path)
        print(f"[train.py] Also copied to: {export_path.resolve()}")


if __name__ == "__main__":
    main()
