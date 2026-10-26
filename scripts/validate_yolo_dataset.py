from pathlib import Path
import math

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = (
    PROJECT_ROOT / "data" / "processed" / "roadguard_yolo"
)

SPLITS = ("train", "val")
NUM_CLASSES = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def validate_split(split):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    print(f"\n{'=' * 45}")
    print(f"Checking {split.upper()} dataset")
    print(f"{'=' * 45}")

    if not image_dir.is_dir() or not label_dir.is_dir():
        print("ERROR: Image or label directory is missing.")
        return False

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(label_dir.glob("*.txt"))

    image_by_stem = {}
    duplicate_stems = set()

    for image_path in images:
        if image_path.stem in image_by_stem:
            duplicate_stems.add(image_path.stem)
        image_by_stem[image_path.stem] = image_path

    label_by_stem = {p.stem: p for p in labels}

    missing_labels = set(image_by_stem) - set(label_by_stem)
    orphan_labels = set(label_by_stem) - set(image_by_stem)

    errors = []
    checked_boxes = 0
    checked_images = 0

    if duplicate_stems:
        errors.append(
            f"Duplicate image filenames (same stem): "
            f"{len(duplicate_stems)}"
        )

    if missing_labels:
        errors.append(f"Images without label files: {len(missing_labels)}")

    if orphan_labels:
        errors.append(f"Labels without matching images: {len(orphan_labels)}")

    for stem, image_path in image_by_stem.items():
        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(f"Unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        checked_images += 1

        label_path = label_by_stem.get(stem)

        if label_path is None:
            continue

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError as exc:
            errors.append(f"Cannot read {label_path.name}: {exc}")
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    "expected 5 values"
                )
                continue

            try:
                class_value = float(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    "non-numeric value"
                )
                continue

            if (
                not math.isfinite(class_value)
                or not class_value.is_integer()
                or not 0 <= class_value < NUM_CLASSES
            ):
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    f"invalid class ID {parts[0]}"
                )
                continue

            if not all(math.isfinite(v) for v in coords):
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    "coordinates contain NaN or infinity"
                )
                continue

            x_center, y_center, box_width, box_height = coords

            if not all(0 <= v <= 1 for v in coords):
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    "coordinate outside [0, 1]"
                )
                continue

            if box_width <= 0 or box_height <= 0:
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    "box width or height is zero"
                )
                continue

            # Check that the entire box fits within the image.
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            if (
                x1 < -1e-6 or y1 < -1e-6
                or x2 > 1 + 1e-6 or y2 > 1 + 1e-6
            ):
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    "bounding box extends outside image"
                )
                continue

            # Ensure the image dimensions are valid.
            if width <= 0 or height <= 0:
                errors.append(f"Invalid image dimensions: {image_path.name}")
                continue

            checked_boxes += 1

    print(f"Images found: {len(images):,}")
    print(f"Label files found: {len(labels):,}")
    print(f"Readable images checked: {checked_images:,}")
    print(f"Valid boxes checked: {checked_boxes:,}")
    print(f"Errors found: {len(errors):,}")

    for error in errors[:30]:
        print("ERROR:", error)

    if len(errors) > 30:
        print(f"... and {len(errors) - 30:,} more errors")

    if errors:
        print(f"RESULT: {split.upper()} NEEDS ATTENTION")
        return False

    print(f"RESULT: {split.upper()} PASSED")
    return True


def main():
    yaml_path = DATASET_DIR / "data.yaml"

    if not yaml_path.is_file():
        print(f"ERROR: Missing dataset configuration: {yaml_path}")
        return

    results = [
        validate_split(split)
        for split in SPLITS
    ]

    print("\n" + "=" * 45)
    if all(results):
        print("ALL CHECKS PASSED — DATASET READY FOR NEXT STEP")
    else:
        print("VALIDATION FAILED — FIX REPORTED ISSUES FIRST")
    print("=" * 45)


if __name__ == "__main__":
    main()
