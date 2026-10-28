from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEST_MODEL = (
    PROJECT_ROOT
    / "models"
    / "best.pt"
)

VAL_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roadguard_yolo"
    / "images"
    / "val"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "roadguard_evaluation"
    / "predictions"
)


def main():
    if not BEST_MODEL.exists():
        raise FileNotFoundError(
            f"Best model not found: {BEST_MODEL}"
        )

    if not VAL_IMAGES.exists():
        raise FileNotFoundError(
            f"Validation images not found: {VAL_IMAGES}"
        )

    # Load the trained RoadGuard model
    model = YOLO(str(BEST_MODEL))

    print("=" * 60)
    print("ROADGUARD: VISUAL ERROR ANALYSIS")
    print("=" * 60)

    print(f"Model: {BEST_MODEL}")
    print(f"Validation images: {VAL_IMAGES}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Run predictions on validation images
    results = model.predict(
        source=str(VAL_IMAGES),
        imgsz=416,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(PROJECT_ROOT / "runs"),
        name="roadguard_evaluation/predictions",
        exist_ok=True,
    )

    print("\nVisual prediction analysis complete.")
    print(f"Predictions saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()