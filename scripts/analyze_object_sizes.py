from pathlib import Path
from collections import defaultdict
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roadguard_yolo"
    / "images"
    / "train"
)

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roadguard_yolo"
    / "labels"
    / "train"
)


# YOLO class IDs from data.yaml
CLASS_NAMES = {
    0: "person",
    1: "rider",
    2: "bike",
    3: "motor",
    4: "car",
    5: "truck",
    6: "bus",
    7: "train",
    8: "traffic light",
    9: "traffic sign",
}


def main():

    print("=" * 60)
    print("ROADGUARD: OBJECT SIZE ANALYSIS")
    print("=" * 60)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found: {LABEL_DIR}"
        )

    # Store area ratios for each class
    object_sizes = defaultdict(list)

    image_files = list(IMAGE_DIR.glob("*.jpg"))

    print(f"Images found: {len(image_files)}")
    print("Analyzing bounding boxes...\n")

    for index, image_path in enumerate(image_files, start=1):

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        # Read image dimensions
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        with open(label_path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # YOLO coordinates are normalized.
                # Calculate the fraction of the image occupied by the box.
                area_ratio = box_width * box_height

                object_sizes[class_id].append(area_ratio)

        if index % 500 == 0:
            print(f"Processed {index}/{len(image_files)} images")

    print("\n" + "=" * 60)
    print("OBJECT SIZE RESULTS")
    print("=" * 60)

    for class_id in sorted(object_sizes):

        sizes = object_sizes[class_id]

        if not sizes:
            continue

        sizes.sort()

        total = len(sizes)

        small = sum(size < 0.01 for size in sizes)
        medium = sum(0.01 <= size < 0.10 for size in sizes)
        large = sum(size >= 0.10 for size in sizes)

        small_pct = small / total * 100
        medium_pct = medium / total * 100
        large_pct = large / total * 100

        average = sum(sizes) / total

        print(
            f"\n{CLASS_NAMES.get(class_id, f'class_{class_id}')}"
        )

        print(f"  Total objects: {total:,}")
        print(f"  Average area:  {average * 100:.3f}% of image")
        print(f"  Small:         {small:,} ({small_pct:.1f}%)")
        print(f"  Medium:        {medium:,} ({medium_pct:.1f}%)")
        print(f"  Large:         {large:,} ({large_pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()