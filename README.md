# RoadGuard

RoadGuard is a Computer Vision object detection project designed to detect common road-scene objects from images. The project uses the BDD100K dataset and a fine-tuned YOLO26n object detection model.

The system detects 10 road-scene object classes:

- person
- rider
- bike
- motor
- car
- truck
- bus
- train
- traffic light
- traffic sign

The project includes the complete data preparation pipeline, model training experiments, evaluation, analysis, and a Streamlit-based demonstration application.

---

## 1. Project Structure

```text
RoadGuard/
│
├── data/
│   ├── raw/
│   │   └── bdd100k.zip
│   │
│   └── processed/
│       └── roadguard_yolo/
│           ├── images/
│           │   ├── train/
│           │   └── val/
│           ├── labels/
│           │   ├── train/
│           │   └── val/
│           └── data.yaml
│
├── models/
│   └── roadguard_best.pt
│
├── runs/
│   ├── roadguard_cpu_baseline/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   └── ...
│   │
│   ├── roadguard_experiment_2/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   └── ...
│   │
│   ├── roadguard_experiment_3/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   └── ...
│   │
│   ├── roadguard_experiment_4/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   └── ...
│   │
│   └── roadguard_evaluation/
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       └── ...
│
├── scripts/
│   ├── inspect_zip.py
│   ├── validate_annotations.py
│   ├── prepare_dataset.py
│   ├── analyze_object_sizes.py
│   ├── evaluate_model.py
│   └── roadguard_demo.py
│
├── reports/
│   ├── metrics/
│   └── figures/
│
├── README.md
└── requirements.txt
```

---

## 2. Dataset

RoadGuard uses the BDD100K dataset for road-scene object detection.

The original dataset contains 100,000 images. The project uses a selected subset for model development:

- Training: 5,000 images
- Validation: 1,000 images
- Image resolution used during training: 416 × 416

The original BDD100K annotations are provided as JSON files containing image metadata and object bounding boxes.

The relevant annotation files are:

```text
bdd100k_labels_images_train.json
bdd100k_labels_images_val.json
```

Each annotated object contains information such as:

- object category
- bounding box coordinates
- occlusion status
- truncation status
- additional attributes for traffic lights

---

## 3. Dataset Processing

The dataset preparation pipeline performs the following steps:

1. Inspects the BDD100K ZIP archive.
2. Identifies the image files and annotation files.
3. Validates the correspondence between annotation records and image files.
4. Checks bounding box validity.
5. Selects the required object-detection classes.
6. Converts the annotations into YOLO format.
7. Creates separate training and validation datasets.
8. Generates the YOLO dataset configuration file.

The final dataset configuration is located at:

```text
data/processed/roadguard_yolo/data.yaml
```

The dataset uses the following class mapping:

```yaml
names:
  - person
  - rider
  - bike
  - motor
  - car
  - truck
  - bus
  - train
  - traffic light
  - traffic sign
```

### Dataset validation

The prepared dataset was validated before training.

Training dataset:

- Images: 5,000
- Label files: 5,000
- Readable images: 5,000
- Valid bounding boxes: 92,407
- Errors: 0

Validation dataset:

- Images: 1,000
- Label files: 1,000
- Readable images: 1,000
- Valid bounding boxes: 18,393
- Errors: 0

Both datasets passed all validation checks.

---

## 4. Model

RoadGuard uses **YOLO26n**, a lightweight YOLO object detection model from Ultralytics.

The nano-sized model was selected because training and experimentation were performed on a CPU-only machine. A lightweight architecture makes the training and inference requirements more manageable.

The final trained model is stored in:

```text
models/roadguard_best.pt
```

The model contains approximately:

```text
2,391,922 parameters
5.4 GFLOPs
```

---

## 5. Training

Training was performed using:

```text
Device: CPU
Image size: 416 × 416
Batch size: 4
Workers: 0
Cache: False
```

The initial baseline was trained for 5 epochs. Since the results indicated that additional training was necessary, the experiment was continued using a longer training configuration.

The training process was interrupted once because the computer was restarted. Training was resumed from the saved checkpoint using Ultralytics' resume functionality.

The final experiment was trained for a total of 50 epochs.

Training outputs and experiment results are stored under:

```text
runs/
```

---

## 6. Experiments

Several experiments were performed to investigate how different training configurations affected the model.

### Baseline

The first baseline experiment used:

```text
Epochs: 5
Image size: 416 × 416
Batch size: 4
Device: CPU
```

The baseline achieved:

```text
Precision:  0.545
Recall:     0.197
mAP50:      0.196
mAP50-95:   0.107
```

The relatively low recall and mAP indicated that additional training was necessary.

### Experiment 2

The second experiment extended training to 50 epochs while keeping the CPU-friendly configuration.

The resulting model achieved:

```text
Precision:  0.5364
Recall:     0.2539
mAP50:      0.2596
mAP50-95:   0.1370
```

This represented an improvement over the 5-epoch baseline.

### Experiment 3

A further experiment was performed with a modified training configuration.

The resulting evaluation was:

```text
Precision:  0.5386
Recall:     0.2622
mAP50:      0.2576
mAP50-95:   0.1353
```

Although recall increased slightly, the mAP50 and mAP50-95 values were slightly lower than Experiment 2.

### Experiment 4

The final experiment investigated another training configuration with the goal of improving detection performance, particularly for difficult road objects.

The resulting validation metrics were:

```text
Precision:  0.506
Recall:     0.267
mAP50:      0.256
mAP50-95:   0.133
```

Although recall increased, the overall mAP values did not improve compared with Experiment 2.

### Experiment selection

Experiment 2 was selected as the final model because it achieved the strongest overall mAP50 and mAP50-95 results among the tested configurations.

The final model used for the RoadGuard application is:

```text
models/roadguard_best.pt
```

---

## 7. Object Size Analysis

An object-size analysis was performed to understand why some classes were more difficult to detect.

The analysis showed that many of the road-scene objects are small within the image.

For example, a large proportion of traffic lights and traffic signs occupy relatively small regions of the image. Small objects are generally more difficult for an object detector to localize and classify accurately.

This analysis helped explain the lower performance observed for some classes, especially:

- bike
- motor
- traffic light
- traffic sign
- rider

In contrast, cars were detected substantially better because they are frequent in the dataset and generally provide more visual information for the detector.

---

## 8. Model Evaluation

The final selected model was evaluated on the 1,000-image validation dataset.

### Overall results

| Metric | Result |
|---|---:|
| Precision | 0.5364 |
| Recall | 0.2539 |
| mAP50 | 0.2596 |
| mAP50-95 | 0.1370 |

### Per-class mAP50

| Class | mAP50 |
|---|---:|
| person | 0.3039 |
| rider | 0.2383 |
| bike | 0.1313 |
| motor | 0.1701 |
| car | 0.5810 |
| truck | 0.3518 |
| bus | 0.2988 |
| train | 0.0000 |
| traffic light | 0.2445 |
| traffic sign | 0.2761 |

The model performs best on the `car` class, with a mAP50 of 0.5810.

Performance is weaker on small or less frequently represented objects. The `train` class has only one validation instance, so its zero mAP50 should not be interpreted as a reliable estimate of general train-detection performance.

---

## 9. Confusion Matrix

A normalized confusion matrix was generated during evaluation to analyze class-level prediction errors.

The confusion matrix helps identify which classes are being confused with one another and provides additional information beyond the aggregate mAP metrics.

In particular, traffic lights and traffic signs remain challenging classes, which is consistent with the object-size analysis.

The generated confusion-matrix figures are stored in:

```text
reports/figures/
```

and the evaluation outputs are also available under:

```text
runs/roadguard_evaluation/
```

---

## 10. Optimization

Optimization was considered in terms of improving model performance while taking into account the available CPU-only hardware.

The project investigated different training configurations rather than simply increasing model complexity.

The experiments showed that longer training improved the baseline performance, while the later experimental configurations did not produce a further improvement in the overall mAP metrics.

Therefore, the configuration from Experiment 2 was retained as the final model.

This decision prioritizes measured validation performance rather than assuming that a more complex or later experiment is automatically better.

---

## 11. Practical Application

RoadGuard includes a Streamlit web application that allows users to upload a road image and run object detection using the trained model.

The application:

1. Loads `models/roadguard_best.pt`.
2. Accepts an uploaded image.
3. Runs YOLO object detection.
4. Displays the detected objects.
5. Draws bounding boxes around detected objects.
6. Displays the predicted class and confidence information.

The demo is implemented in:

```text
scripts/roadguard_demo.py
```

---

## 12. Running the Project

### Requirements

The project uses Python 3.11.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the model evaluation

From the project root:

```bash
python scripts/evaluate_model.py
```

The evaluation results will be saved under:

```text
runs/roadguard_evaluation/
```

### Run the RoadGuard demo

From the project root:

```bash
streamlit run scripts/roadguard_demo.py
```

A browser window will open with the RoadGuard interface.

Upload a road-scene image to see the model's predictions and bounding boxes.

---

## 13. Reproducing the Dataset Preparation

The dataset preparation scripts are located in:

```text
scripts/
```

The main preparation script is:

```bash
python scripts/prepare_dataset.py
```

The resulting YOLO dataset is created under:

```text
data/processed/roadguard_yolo/
```

The dataset configuration is:

```text
data/processed/roadguard_yolo/data.yaml
```

The annotation validation script can be run with:

```bash
python scripts/validate_annotations.py
```

The object-size analysis can be run with:

```bash
python scripts/analyze_object_sizes.py
```

---

## 14. Results

The final RoadGuard model demonstrates the ability to detect multiple road-scene object categories from images.

The strongest detection performance was obtained for cars, while smaller objects such as traffic lights, traffic signs, bikes, and motorcycles were more challenging.

The experiments demonstrate that additional training improved the initial baseline, but simply changing the training configuration did not consistently improve the final mAP.

The project therefore uses the best-performing experimentally validated model rather than selecting a model based only on the number of epochs or the latest experiment.

---

## 15. Limitations

The main limitations of the current system are:

- Training was performed using CPU-only hardware.
- Only 5,000 training images and 1,000 validation images were used.
- Small objects are more difficult to detect.
- Traffic lights and traffic signs can occasionally be missed or confused.
- Some classes have very few examples, particularly `train`.
- The current model prioritizes a lightweight architecture suitable for the available hardware.

These limitations provide opportunities for future improvement through a larger training subset, higher-resolution training, more extensive hyperparameter tuning, stronger augmentation strategies, or a larger detection model.

---

## 16. Technologies

- Python 3.11
- Ultralytics YOLO26n
- PyTorch
- OpenCV
- Pillow
- NumPy
- Streamlit
- Matplotlib
- Pandas
- YAML

---

## 17. Final Model

The final trained RoadGuard model is available at:

```text
models/roadguard_best.pt
```

This file is used by the evaluation and demonstration components of the project.

---

## 18. Project Status

The RoadGuard project includes:

- Dataset inspection
- Dataset validation
- Dataset preparation
- YOLO annotation conversion
- Object-size analysis
- Model fine-tuning
- Multiple training experiments
- Model evaluation
- Confusion-matrix analysis
- Model selection
- Streamlit demonstration application
- Final trained model
- Project documentation