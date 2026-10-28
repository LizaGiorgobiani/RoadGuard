from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import ijson

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k.zip"

IMAGE_PREFIX = "bdd100k/bdd100k/images/100k/"
ANNOTATION_FILES = {
    "train": (
        "bdd100k_labels_release/bdd100k/labels/"
        "bdd100k_labels_images_train.json"
    ),
    "val": (
        "bdd100k_labels_release/bdd100k/labels/"
        "bdd100k_labels_images_val.json"
    ),
}


def validate_annotations():
    if not ZIP_PATH.exists():
        print(f"ZIP not found: {ZIP_PATH}")
        return

    with ZipFile(ZIP_PATH, "r") as archive:
        archive_names = archive.namelist()

        # Only index the 100k object-detection image component.
        image_paths = {}
        for name in archive_names:
            if not name.startswith(IMAGE_PREFIX):
                continue

            if Path(name).suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".webp"
            }:
                continue

            filename = Path(name).name
            image_paths.setdefault(filename, []).append(name)

        print("=" * 60)
        print("ROADGUARD: ANNOTATION VALIDATION")
        print("=" * 60)
        print(f"100k image files indexed: {sum(map(len, image_paths.values())):,}")
        print(f"Unique image filenames: {len(image_paths):,}")

        for split, annotation_path in ANNOTATION_FILES.items():
            print(f"\n--- {split.upper()} ---")

            if annotation_path not in archive_names:
                print("Annotation file missing from archive.")
                continue

            record_count = 0
            matched_count = 0
            missing_count = 0
            duplicate_filename_count = 0
            class_counts = Counter()
            invalid_boxes = 0
            missing_boxes = 0
            examples_missing = []

            with archive.open(annotation_path) as file:
                for record in ijson.items(file, "item"):
                    record_count += 1
                    filename = record.get("name")

                    matches = image_paths.get(filename, [])
                    if matches:
                        matched_count += 1
                        if len(matches) > 1:
                            duplicate_filename_count += 1
                    else:
                        missing_count += 1
                        if len(examples_missing) < 5:
                            examples_missing.append(filename)

                    for label in record.get("labels", []):
                        category = label.get("category", "UNKNOWN")
                        class_counts[category] += 1

                        box = label.get("box2d")
                        if not isinstance(box, dict):
                            missing_boxes += 1
                            continue

                        try:
                            x1 = float(box["x1"])
                            y1 = float(box["y1"])
                            x2 = float(box["x2"])
                            y2 = float(box["y2"])

                            if x2 <= x1 or y2 <= y1:
                                invalid_boxes += 1
                        except (KeyError, TypeError, ValueError):
                            invalid_boxes += 1

            print(f"Annotation records: {record_count:,}")
            print(f"Matched image records: {matched_count:,}")
            print(f"Unmatched image records: {missing_count:,}")
            print(f"Records with duplicate filename matches: "
                  f"{duplicate_filename_count:,}")
            print(f"Missing bounding boxes: {missing_boxes:,}")
            print(f"Invalid bounding boxes: {invalid_boxes:,}")

            if examples_missing:
                print("Examples of unmatched filenames:")
                for filename in examples_missing:
                    print(f"  {filename}")

            print("\nObject instances by category:")
            for category, count in class_counts.most_common():
                print(f"  {category:<25} {count:>10,}")

    print("\nValidation complete. No files were extracted or modified.")


if __name__ == "__main__":
    validate_annotations()
