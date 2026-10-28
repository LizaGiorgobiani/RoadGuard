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
    / "runs"
    / "roadguard_experiment_2"
    / "weights"
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

        # Keep resolution consistent with Experiment 2.
        imgsz=416,
        batch=4,

        # Final training experiment.
        epochs=10,
        patience=5,

        # CPU settings.
        device="cpu",
        workers=0,
        cache=False,

        # Augmentation changes.
        # Less mosaic: fewer four-image composites.
        mosaic=0.5,

        # More conservative random scaling.
        scale=0.3,

        # Disable mosaic in the final 3 epochs.
        close_mosaic=3,

        # Reproducibility and output.
        seed=42,
        plots=True,
        project=str(PROJECT_ROOT / "runs"),
        name="roadguard_experiment_4",
    )


if __name__ == "__main__":
    main()
