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

BEST_MODEL = (
    PROJECT_ROOT
    / "models"
    / "best.pt"
)



def main():
    # Check that the dataset configuration exists
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    # Check that the trained model exists
    if not BEST_MODEL.exists():
        raise FileNotFoundError(
            f"Best model not found: {BEST_MODEL}"
        )

    # Load our trained RoadGuard model
    model = YOLO(str(BEST_MODEL))

    print("=" * 60)
    print("ROADGUARD: MODEL EVALUATION")
    print("=" * 60)

    # Evaluate on the validation dataset
    results = model.val(
        data=str(DATASET_YAML),
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(PROJECT_ROOT / "runs"),
        name="roadguard_evaluation",
    )

    print("\n" + "=" * 60)
    print("ROADGUARD: EVALUATION RESULTS")
    print("=" * 60)

    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    print(f"mAP50:      {results.box.map50:.4f}")
    print(f"mAP50-95:   {results.box.map:.4f}")

    print("\nPer-class mAP50:")
    for class_id, score in enumerate(results.box.ap50):
        class_name = model.names[class_id]
        print(f"  {class_name:<20} {score:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()