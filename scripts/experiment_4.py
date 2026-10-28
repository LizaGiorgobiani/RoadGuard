from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roadguard_yolo"
    / "data.yaml"
)

BASE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "best.pt"
)


def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    if not BASE_MODEL.exists():
        raise FileNotFoundError(
            f"Base model not found: {BASE_MODEL}"
        )

    # Start from our best 416x416 model.
    model = YOLO(str(BASE_MODEL))

    print("=" * 60)
    print("ROADGUARD: EXPERIMENT 4")
    print("Targeted augmentation for small objects")
    print("=" * 60)

    model.train(
        data=str(DATASET_YAML),
        imgsz=416,
        batch=4,
        epochs=10,
        patience=5,
        device="cpu",
        workers=0,
        cache=False,
        mosaic=0.5,
        scale=0.3,
        close_mosaic=3,
        seed=42,
        plots=True,
        project=str(PROJECT_ROOT / "runs"),
        name="roadguard_experiment_4",
    )


if __name__ == "__main__":
    main()
