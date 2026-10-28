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

LAST_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "roadguard_experiment_2"
    / "weights"
    / "last.pt"
)

def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    if not LAST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Training checkpoint not found: {LAST_CHECKPOINT}"
        )

    # Load a lightweight pretrained detection model.
    # model = YOLO("yolo26n.pt")
    model = YOLO(str(LAST_CHECKPOINT))

    #first model - not good 5 epochs was nto enough

    """model.train(
        data=str(DATASET_YAML),
        epochs=5,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        cache=False,
        pretrained=True,
        seed=42,
        plots=True,
        project=str(PROJECT_ROOT / "runs"),
        name="roadguard_cpu_baseline",
    ) """

    #second model
    """ model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        cache=False,
        pretrained=True,
        seed=42,
        plots=True,

        #stop if validation performance stops improving
        patience=10,

        project=str(PROJECT_ROOT / "runs"),
        name="roadguard_experiment_2",
        
    )"""

    #training was stopped due to unseen circumstances so the training needs to be continued this way
    model.train(resume=True)

if __name__ == "__main__":
    main()
