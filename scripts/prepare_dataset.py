from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json

import cv2
import numpy as np
import ijson

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k.zip"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "roadguard_yolo"

IMAGE_ROOT = "bdd100k/bdd100k/images/100k"

ANNOTATIONS = {
    "train": (
        "bdd100k_labels_release/bdd100k/labels/"
        "bdd100k_labels_images_train.json"
    ),
    "val": (
        "bdd100k_labels_release/bdd100k/labels/"
        "bdd100k_labels_images_val.json"
    ),
}

# Fixed class IDs: do not change their order after training starts.
CLASSES = [
    "person",
    "rider",
    "bike",
    "motor",
    "car",
    "truck",
    "bus",
    "train",
    "traffic light",
    "traffic sign",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

LIMITS = {
    "train": 5000,
    "val": 1000,
}


def build_image_index(archive, split):
    prefix = f"{IMAGE_ROOT}/{split}/"
    index = {}

    for name in archive.namelist():
        if not name.startswith(prefix):
            continue

        if Path(name).suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".webp"
        }:
            continue

        filename = Path(name).name

        # Don't silently choose if filenames are duplicated.
        if filename in index:
            raise ValueError(
                f"Duplicate filename in {split}: {filename}"
            )

        index[filename] = name

    return index


def convert_boxes(record, width, height, counters):
    yolo_labels = []

    for obj in record.get("labels", []):
        category = obj.get("category")
        if category not in CLASS_TO_ID:
            counters["excluded_class"] += 1
            continue

        box = obj.get("box2d")
        if not isinstance(box, dict):
            counters["missing_box"] += 1
            continue

        try:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
        except (KeyError, TypeError, ValueError):
            counters["malformed_box"] += 1
            continue

        coords = [x1, y1, x2, y2]
        if not all(np.isfinite(v) for v in coords):
            counters["malformed_box"] += 1
            continue

        # Clip coordinates to the image boundaries.
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))

        if x2 <= x1 or y2 <= y1:
            counters["invalid_box"] += 1
            continue

        # YOLO uses normalized center coordinates and dimensions.
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height

        class_id = CLASS_TO_ID[category]

        yolo_labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
        counters[f"class:{category}"] += 1

    return yolo_labels


def prepare_split(archive, split, image_index):
    image_dir = OUTPUT_DIR / "images" / split
    label_dir = OUTPUT_DIR / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    processed = 0
    annotation_path = ANNOTATIONS[split]

    with archive.open(annotation_path) as file:
        for record in ijson.items(file, "item"):
            if processed >= LIMITS[split]:
                break

            filename = record.get("name")
            archive_image = image_index.get(filename)

            if archive_image is None:
                counters["missing_image"] += 1
                continue

            with archive.open(archive_image) as image_file:
                image_bytes = image_file.read()

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                counters["unreadable_image"] += 1
                continue

            height, width = image.shape[:2]

            labels = convert_boxes(
                record, width, height, counters
            )

            # Keep only images containing at least one selected object.
            if not labels:
                counters["no_selected_objects"] += 1
                continue

            output_image = image_dir / filename
            output_label = label_dir / f"{Path(filename).stem}.txt"

            if not cv2.imwrite(str(output_image), image):
                counters["write_errors"] += 1
                continue

            output_label.write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8"
            )

            processed += 1

            if processed % 500 == 0:
                print(
                    f"{split}: prepared {processed:,}/"
                    f"{LIMITS[split]:,} images"
                )

    print(f"\n{split.upper()} SUMMARY")
    print(f"Images prepared: {processed:,}")
    for key, value in sorted(counters.items()):
        print(f"{key}: {value:,}")

    return processed


def write_yaml():
    yaml_path = OUTPUT_DIR / "data.yaml"
    names = json.dumps(CLASSES, ensure_ascii=False)

    yaml_text = (
        f"path: {OUTPUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {names}\n"
    )

    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\nDataset configuration written to: {yaml_path}")


def main():
    if not ZIP_PATH.exists():
        print(f"ZIP not found: {ZIP_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with ZipFile(ZIP_PATH, "r") as archive:
        for split in ("train", "val"):
            print(f"\nIndexing {split} images...")
            image_index = build_image_index(archive, split)
            print(f"Indexed {len(image_index):,} images")

            print(f"Preparing {split} dataset...")
            count = prepare_split(archive, split, image_index)

            if count == 0:
                raise RuntimeError(
                    f"No images were prepared for {split}."
                )

    write_yaml()
    print("\nDataset preparation finished.")


if __name__ == "__main__":
    main()
