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

    # Load the trained model from Experiment 2
    model = YOLO(str(BASE_MODEL))

    print("=" * 60)
    print("ROADGUARD: EXPERIMENT 3")
    print("Higher input resolution")
    print("=" * 60)

    model.train(
        data=str(DATASET_YAML),

        # Experiment 3 variable
        imgsz=640,
        batch=2,
        epochs=10,
        device="cpu",
        workers=0,
        cache=False,
        seed=42,
        plots=True,
        project=str(PROJECT_ROOT / "runs"),
        name="roadguard_experiment_3",
    )


if __name__ == "__main__":
    main()