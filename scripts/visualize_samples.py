from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "roadguard_yolo"

IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

CLASSES = [
    "person", "rider", "bike", "motor", "car",
    "truck", "bus", "train", "traffic light", "traffic sign",
]

label_files = list(LABEL_DIR.glob("*.txt"))

if not label_files:
    raise RuntimeError(f"No label files found in {LABEL_DIR}")

samples = random.sample(label_files, min(12, len(label_files)))

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for ax, label_file in zip(axes, samples):
    image_path = IMAGE_DIR / f"{label_file.stem}.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        ax.set_title(f"Cannot read {image_path.name}")
        ax.axis("off")
        continue

    height, width = image.shape[:2]

    for line in label_file.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        x_center, y_center, box_width, box_height = map(
            float, values[1:]
        )

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if 0 <= class_id < len(CLASSES):
            name = CLASSES[class_id]
        else:
            name = f"Unknown {class_id}"

        cv2.putText(
            image,
            name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(label_file.stem)
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
