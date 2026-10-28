from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k.zip"

ANNOTATIONS = [
    "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
    "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
]

def inspect_annotations():
    if not ZIP_PATH.exists():
        print(f"ZIP not found: {ZIP_PATH}")
        return

    with ZipFile(ZIP_PATH, "r") as archive:
        available_files = set(archive.namelist())

        for annotation_path in ANNOTATIONS:
            print("\n" + "=" * 60)
            print(annotation_path)
            print("=" * 60)

            if annotation_path not in available_files:
                print("Annotation file not found in ZIP.")
                continue

            with archive.open(annotation_path) as file:
                preview = file.read(4000).decode("utf-8", errors="replace")

            print(preview[:3000])

    print("\nInspection complete. Nothing was extracted.")

if __name__ == "__main__":
    inspect_annotations()
