from pathlib import Path
from zipfile import ZipFile, BadZipFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent


ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k.zip"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ANNOTATION_EXTENSIONS = {".json", ".txt", ".xml"}

def inspect_zip(zip_path: Path) -> None:
    """Inspect a dataset ZIP without extracting its contents."""

    print("=" * 60)
    print("ROADGUARD: BDD100K ZIP INSPECTION")
    print("=" * 60)


    if not zip_path.exists():
        print(f"\nZIP not found: {zip_path}")
        print("Check the filename and make sure the download is complete.")
        return

    print(f"\nZIP path: {zip_path}")
    print(f"ZIP size: {zip_path.stat().st_size / (1024 ** 3):.2f} GB")

    try:
        with ZipFile(zip_path, "r") as archive:
            files = [
                item for item in archive.infolist()
                if not item.is_dir()
            ]

            image_files = [
                item for item in files
                if Path(item.filename).suffix.lower()
                in IMAGE_EXTENSIONS
            ]

            annotation_files = [
                item for item in files
                if Path(item.filename).suffix.lower()
                in ANNOTATION_EXTENSIONS
            ]

            print(f"\nTotal files: {len(files):,}")
            print(f"Image files: {len(image_files):,}")
            print(f"Possible annotation files: {len(annotation_files):,}")

            print("\nFirst 30 archive entries:")
            for item in files[:30]:
                print(f"  {item.filename}")

            print("\nPossible annotation filenames:")
            for item in annotation_files[:20]:
                print(f"  {item.filename}")

            if len(annotation_files) > 20:
                print("  ... more annotation files omitted")

            print("\nLargest 10 files by uncompressed size:")
            largest_files = sorted(
                files,
                key=lambda item: item.file_size,
                reverse=True
            )[:10]

            for item in largest_files:
                size_mb = item.file_size / (1024 ** 2)
                print(f"  {size_mb:8.2f} MB  {item.filename}")

            print("\nInspection finished. Nothing was extracted.")

    except BadZipFile:
        print("\nERROR: This is not a valid ZIP archive.")
        print("The download may be incomplete or corrupted.")
    except OSError as error:
        print(f"\nCould not read the ZIP: {error}")


if __name__ == "__main__":
    inspect_zip(ZIP_PATH)