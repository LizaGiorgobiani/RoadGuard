
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k.zip"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def inspect_dataset():
    if not ZIP_PATH.exists():
        print(f"ZIP not found: {ZIP_PATH}")
        return

    folder_counts = Counter()
    annotation_files = []

    with ZipFile(ZIP_PATH, "r") as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue

            path = Path(item.filename)

            if path.suffix.lower() in IMAGE_EXTENSIONS:
                # Count images by their immediate parent directory.
                folder_counts[path.parent.as_posix()] += 1

            if path.suffix.lower() == ".json":
                annotation_files.append(
                    (item.filename, item.file_size)
                )

    print("=" * 60)
    print("ROADGUARD: DATASET STRUCTURE")
    print("=" * 60)

    print("\nImage counts by directory:")
    for folder, count in sorted(folder_counts.items()):
        print(f"{count:>8,} images | {folder}")

    print("\nAnnotation files:")
    for filename, size in annotation_files:
        print(f"{size / (1024 ** 2):>8.2f} MB | {filename}")

    print("\nInspection complete. Nothing was extracted.")

if __name__ == "__main__":
    inspect_dataset()
