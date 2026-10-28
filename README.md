# RoadGuard

RoadGuard is a Computer Vision object-detection system for analyzing road scenes. The project uses the BDD100K dataset and a fine-tuned YOLO26n model to detect common road objects such as cars, pedestrians, traffic lights, and traffic signs.

## Project Overview

The project follows an end-to-end Computer Vision workflow:

1. Dataset inspection and preparation
2. Model training / fine-tuning
3. Experiments and comparison
4. Model evaluation and error analysis
5. Optimization experiments
6. Practical application through an interactive demo

The project uses:

- **Python 3.11**
- **Ultralytics YOLO**
- **PyTorch**
- **OpenCV**
- **Pillow**
- **Streamlit**
- **BDD100K dataset**

The model is trained using CPU-only hardware.

---

## Project Structure

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
│           │
│           ├── labels/
│           │   ├── train/
│           │   └── val/
│           │
│           └── data.yaml
│
├── runs/
│   ├── roadguard_cpu_baseline/
│   ├── roadguard_experiment_2/
│   ├── roadguard_experiment_3/
│   └── roadguard_experiment_4/
│
├── scripts/
│   ├── inspect_zip.py
│   ├── validate_annotations.py
│   ├── prepare_dataset.py
│   ├── analyze_object_sizes.py
│   ├── evaluate_model.py
│   ├── experiment_4.py
│   └── roadguard_demo.py
│
└── README.md
```

---

## Dataset

The project uses the **BDD100K** dataset for road-scene object detection.

The prepared dataset contains:

- **5,000 training images**
- **1,000 validation images**
- YOLO-format bounding-box annotations

The final detection classes are:

```text
person
rider
bike
motor
car
truck
bus
train
traffic light
traffic sign
```

The dataset was inspected before extraction and the annotations were validated after preparation.

The final validation check reported:

```text
TRAIN
Images found: 5,000
Label files found: 5,000
Readable images checked: 5,000
Valid boxes checked: 92,407
Errors found: 0

VAL
Images found: 1,000
Label files found: 1,000
Readable images checked: 1,000
Valid boxes checked: 18,393
Errors found: 0
```

---

## Model

The project uses **YOLO26n**, a lightweight object-detection model from Ultralytics.

The final selected model is:

```text
runs/roadguard_experiment_2/weights/best.pt
```

The model was selected based on overall validation performance after comparing several configurations.

### Final model evaluation

```text
Precision:   0.5364
Recall:      0.2539
mAP50:       0.2596
mAP50-95:    0.1370
```

Per-class mAP50:

```text
person          0.3039
rider           0.2383
bike            0.1313
motor           0.1701
car             0.5810
truck           0.3518
bus             0.2988
train           0.0000
traffic light   0.2445
traffic sign    0.2761
```

The relatively weaker performance on small objects, especially traffic lights and traffic signs, was investigated during the experiment and error-analysis stages.

---

# Experiments

Several training configurations were tested.

## Experiment 1 — Baseline

A lightweight YOLO26n model was trained at:

```text
Image size: 416 × 416
Epochs: 5
Batch size: 4
Device: CPU
```

The initial model showed that additional training was necessary.

## Experiment 2 — Main Training Experiment

The model was trained for up to 50 epochs with early-stopping patience.

The final selected checkpoint was:

```text
runs/roadguard_experiment_2/weights/best.pt
```

This configuration achieved the best overall performance among the tested configurations.

## Experiment 3 — Higher Resolution

A higher image resolution was tested to investigate whether additional image detail would improve detection of small objects.

The overall performance did not improve compared with the 416 × 416 configuration, so the higher-resolution configuration was not selected as the final model.

## Experiment 4 — Targeted Augmentation

A targeted augmentation strategy was tested with reduced mosaic and more conservative scaling.

This experiment slightly improved traffic-light and traffic-sign mAP50, but decreased the overall validation performance.

Therefore, Experiment 2 remained the final model.

---

# Evaluation

The model was evaluated using:

- Precision
- Recall
- mAP50
- mAP50-95
- Per-class mAP50
- Confusion matrix
- Visual inspection of predictions
- Object-size analysis

The evaluation showed that the model performs best on relatively common classes such as cars, while smaller and less frequent objects are more difficult to detect.

Visual inspection was also used to identify cases where traffic signs and traffic lights were incorrectly detected or missed.

---

# Running the RoadGuard Demo

The project includes an interactive Streamlit application that allows the trained model to be tested on new images.

The application performs the following process:

```text
Upload road image
       ↓
RoadGuard YOLO model
       ↓
Object detection
       ↓
Bounding boxes + confidence scores
       ↓
Annotated image + detection table
```

## 1. Activate the virtual environment

From the project root:

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

If the environment is already active, this step can be skipped.

---

## 2. Install dependencies

Install the required packages:

```bash
pip install ultralytics streamlit pillow opencv-python
```

If the packages have already been installed, this step can be skipped.

---

## 3. Check the trained model

The demo expects the final trained model at:

```text
runs/roadguard_experiment_2/weights/best.pt
```

Make sure this file exists before starting the application.

---

## 4. Start the demo

From the project root, run:

```bash
streamlit run scripts/roadguard_demo.py
```

Streamlit will display a local address, normally:

```text
http://localhost:8501
```

Open the displayed address in a web browser.

---

## 5. Use the application

Inside the RoadGuard application:

1. Click **Browse files**.
2. Select a `.jpg`, `.jpeg`, or `.png` road image.
3. The original image will be displayed.
4. Click **Detect Objects**.
5. RoadGuard will run the trained YOLO model.
6. The annotated image will be displayed with bounding boxes.
7. A detection table will show the detected object classes and confidence scores.

Example output:

```text
Detected Objects

Object              Confidence
--------------------------------
car                 91.2%
car                 87.5%
traffic light       76.1%
traffic sign        68.4%
person              84.7%
```

For the best demonstration, use road images that were not included in the training or validation datasets.

---

# Reproducing the Training

The prepared dataset configuration is located at:

```text
data/processed/roadguard_yolo/data.yaml
```

The final model can be reproduced by running the corresponding training script with the dataset configuration.

The trained model itself is stored at:

```text
runs/roadguard_experiment_2/weights/best.pt
```

---

# Evaluation Script

The final model can be evaluated using:

```bash
python scripts/evaluate_model.py
```

The evaluation results are saved under:

```text
runs/roadguard_evaluation/
```

---

# Dataset Validation

Before training, the prepared dataset was checked for:

- Missing images
- Missing label files
- Unreadable images
- Invalid bounding boxes
- Annotation consistency

The final dataset validation completed with:

```text
ALL CHECKS PASSED — DATASET READY FOR NEXT STEP
```

---

# Limitations

The model is a lightweight CPU-trained object detector and therefore does not achieve perfect detection performance.

The main limitations observed during evaluation were:

- Small-object detection is difficult.
- Traffic lights can occasionally be missed or incorrectly classified.
- Traffic signs can occasionally be missed or incorrectly classified.
- Rare classes such as `train` have insufficient validation examples for reliable performance measurement.
- CPU-only training limits the amount of experimentation and training time possible.

These limitations were considered during the experiment and optimization stages.

---

# Final Result

The final RoadGuard system consists of:

```text
BDD100K
   ↓
Dataset inspection
   ↓
Dataset preparation and validation
   ↓
YOLO26n fine-tuning
   ↓
Experiments
   ↓
Evaluation + error analysis
   ↓
Optimization experiments
   ↓
Final model
   ↓
Streamlit practical demonstration
```

The final selected model is:

```text
runs/roadguard_experiment_2/weights/best.pt
```

RoadGuard can be used to detect road-scene objects in previously unseen images and visually present the resulting bounding boxes and confidence scores.