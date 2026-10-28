# RoadGuard

RoadGuard is a Computer Vision object detection project designed to detect common road-scene objects from images. The project uses the BDD100K dataset and a lightweight YOLO26n model fine-tuned for road-scene object detection.

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

The project includes dataset inspection and preparation, annotation validation, model training, multiple experiments, model evaluation, object-size analysis, visualization, and a Streamlit-based demonstration application.

---

## 📄 Final Report

The complete project report is available here:

**[RoadGuard Final Report](reports/RoadGuard_Final_Report.md)**

The final report contains the project's:

- Dataset preparation and analysis
- Data validation
- Model architecture and training methodology
- Experimental configurations
- Model evaluation
- Performance comparison
- Confusion matrix analysis
- Object-size analysis
- Optimization experiments
- Limitations
- Conclusions and future improvements

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
│   └── best.pt
│
├── runs/
│   ├── roadguard_cpu_baseline/
│   ├── roadguard_experiment_2/
│   ├── roadguard_experiment_3/
│   ├── roadguard_experiment_4/
│   └── roadguard_evaluation/
│
├── scripts/
│   ├── analyze_object_sizes.py
│   ├── evaluate_model.py
│   ├── experiment_4.py
│   ├── experiment_improvement.py
│   ├── inspect_annotations.py
│   ├── inspect_dataset.py
│   ├── inspect_zip.py
│   ├── prepare_dataset.py
│   ├── roadguard_demo.py
│   ├── train_model.py
│   ├── validate_annotations.py
│   ├── validate_yolo_dataset.py
│   ├── visualize_results.py
│   ├── visualize_samples.py
│   └── yolo26n.pt
│
├── reports/
│   ├── metrics/
│   ├── figures/
│   └── RoadGuard_Final_Report.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 2. Project Overview

RoadGuard follows a complete Computer Vision development workflow:

1. Dataset inspection
2. Dataset preparation and cleaning
3. Annotation validation
4. YOLO dataset creation
5. Model training
6. Experimental comparison
7. Model evaluation
8. Error and object-size analysis
9. Model optimization experiments
10. Practical application through a web-based demonstration

The project was developed using CPU-only hardware, so a lightweight YOLO model and CPU-friendly training configuration were used.

---

## 3. Dataset

RoadGuard uses the **BDD100K** dataset for road-scene object detection.

The original BDD100K dataset contains 100,000 images. For this project, a subset was prepared for model development:

- **Training images:** 5,000
- **Validation images:** 1,000
- **Training image size:** 416 × 416

The processed dataset is stored in:

```text
data/processed/roadguard_yolo/
```

The YOLO dataset configuration is:

```text
data/processed/roadguard_yolo/data.yaml
```

### Dataset Classes

The project uses the following 10 classes:

```text
0  person
1  rider
2  bike
3  motor
4  car
5  truck
6  bus
7  train
8  traffic light
9  traffic sign
```

---

## 4. Dataset Preparation

The dataset preparation pipeline was used to inspect the original dataset, validate annotations, select the required classes, and convert the annotations into YOLO format.

The preparation process includes:

- Inspecting the dataset archive
- Inspecting dataset images
- Inspecting annotations
- Matching annotation records with image files
- Checking bounding boxes
- Selecting the required object classes
- Converting annotations into YOLO format
- Creating training and validation directories
- Validating the resulting YOLO dataset
- Visualizing sample annotations

The main preparation script is:

```text
scripts/prepare_dataset.py
```

Run it from the project root:

```bash
python scripts/prepare_dataset.py
```

The processed dataset is generated under:

```text
data/processed/roadguard_yolo/
```

---

## 5. Dataset Validation

Several scripts are included to verify that the dataset is correctly prepared before training.

### Inspect ZIP archive

```bash
python scripts/inspect_zip.py
```

### Inspect dataset

```bash
python scripts/inspect_dataset.py
```

### Inspect annotations

```bash
python scripts/inspect_annotations.py
```

### Validate annotations

```bash
python scripts/validate_annotations.py
```

### Validate YOLO dataset

```bash
python scripts/validate_yolo_dataset.py
```

The final prepared dataset passed the validation checks.

### Training dataset

- Images found: **5,000**
- Label files found: **5,000**
- Readable images checked: **5,000**
- Valid boxes checked: **92,407**
- Errors found: **0**

### Validation dataset

- Images found: **1,000**
- Label files found: **1,000**
- Readable images checked: **1,000**
- Valid boxes checked: **18,393**
- Errors found: **0**

---

## 6. Dataset Visualization

Sample annotations can be visualized using:

```bash
python scripts/visualize_samples.py
```

The visualization displays the bounding boxes associated with the objects in the images.

These visualizations were used to manually verify that the generated YOLO annotations corresponded correctly to the objects in the images.

---

## 7. Model

RoadGuard uses **YOLO26n**, a lightweight object detection model from Ultralytics.

The nano-sized model was selected because the project was trained using CPU-only hardware.

The final trained model is stored in:

```text
models/best.pt
```

The pretrained YOLO26n model used during development is located in:

```text
scripts/yolo26n.pt
```

---

## 8. Model Training

The main training script is:

```text
scripts/train_model.py
```

Training was performed using a CPU-friendly configuration.

The main configuration included:

```text
Device: CPU
Image size: 416 × 416
Batch size: 4
Workers: 0
Cache: False
```

The initial baseline was trained for 5 epochs. The model was subsequently trained for a longer run using the saved checkpoint.

The training process was interrupted once because the computer was restarted. Training was resumed from the saved checkpoint rather than starting the experiment from the beginning.

The completed long training run reached a total of **50 training epochs**.

---

## 9. Training Runs

The completed training runs are stored in:

```text
runs/
```

The directory contains the outputs from the different experiments:

```text
runs/
├── roadguard_cpu_baseline/
├── roadguard_experiment_2/
├── roadguard_experiment_3/
├── roadguard_experiment_4/
└── roadguard_evaluation/
```

The final trained model intended for practical use is:

```text
models/best.pt
```

---

## 10. Experiments

Multiple training configurations were evaluated to determine whether model performance could be improved.

The experiment scripts include:

```text
scripts/experiment_improvement.py
scripts/experiment_4.py
```

The experiments were evaluated using the same validation dataset and compared using Precision, Recall, mAP50, and mAP50-95.

### Experiment Results

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline | 0.5447 | 0.1966 | 0.1963 | 0.1075 |
| Experiment 2 | 0.5364 | 0.2539 | **0.2596** | **0.1370** |
| Experiment 3 | 0.5386 | 0.2622 | 0.2576 | 0.1353 |
| Experiment 4 | 0.5057 | **0.2669** | 0.2556 | 0.1330 |

Experiment 2 achieved the highest overall mAP50 and mAP50-95 and was therefore selected as the strongest overall experiment.

Experiment 4 achieved the highest Recall, but its Precision and overall mAP were lower.

Detailed numerical results are stored in:

```text
reports/metrics/
```

---

## 11. Object Size Analysis

An object-size analysis was performed to better understand the distribution of objects in the training dataset.

The analysis can be run using:

```bash
python scripts/analyze_object_sizes.py
```

The analysis calculates:

- Number of objects per class
- Average bounding-box area
- Percentage of small objects
- Percentage of medium objects
- Percentage of large objects

The analysis showed that many objects in the dataset occupy relatively small regions of the image.

This is particularly relevant for road-scene objects such as traffic lights and traffic signs, which can be difficult to detect when they occupy only a small portion of an image.

---

## 12. Model Evaluation

The final model can be evaluated using:

```text
scripts/evaluate_model.py
```

Run:

```bash
python scripts/evaluate_model.py
```

The evaluation uses the validation dataset defined in:

```text
data/processed/roadguard_yolo/data.yaml
```

The evaluation results are saved under:

```text
runs/roadguard_evaluation/
```

The evaluation includes:

- Precision
- Recall
- mAP50
- mAP50-95
- Per-class mAP50
- Confusion matrices
- Additional evaluation plots

---

## 13. Final Evaluation Results

The selected final model was evaluated on the 1,000-image validation dataset.

### Overall Performance

| Metric | Result |
|---|---:|
| Precision | **0.5364** |
| Recall | **0.2539** |
| mAP50 | **0.2596** |
| mAP50-95 | **0.1370** |

### Per-Class mAP50

| Class | mAP50 |
|---|---:|
| person | 0.3039 |
| rider | 0.2383 |
| bike | 0.1313 |
| motor | 0.1701 |
| car | **0.5810** |
| truck | 0.3518 |
| bus | 0.2988 |
| train | 0.0000 |
| traffic light | 0.2445 |
| traffic sign | 0.2761 |

The strongest performance was obtained for the `car` class, with a mAP50 of **0.5810**.

Smaller and less frequent object categories were more challenging for the detector.

The `train` class contains only one validation instance, so its 0.0000 mAP50 should not be considered a reliable measure of general train-detection performance.

---

## 14. Confusion Matrix Analysis

A normalized confusion matrix was generated during model evaluation.

The confusion matrix was used to analyze prediction errors between the different object categories and the background class.

The analysis showed that some classes are more difficult to distinguish than others, with traffic lights and traffic signs remaining challenging.

Selected evaluation figures are stored in:

```text
reports/figures/
```

---

## 15. Optimization

Model optimization was investigated through additional training experiments.

The goal was to determine whether changes to the training configuration could improve:

- Detection accuracy
- Recall
- mAP
- Performance on difficult object classes

The experiments showed that extending training beyond the initial baseline improved the model, while the later configurations did not produce a further improvement in the overall mAP metrics.

The comparison was:

| Experiment | mAP50 | mAP50-95 |
|---|---:|---:|
| Baseline | 0.1963 | 0.1075 |
| Experiment 2 | **0.2596** | **0.1370** |
| Experiment 3 | 0.2576 | 0.1353 |
| Experiment 4 | 0.2556 | 0.1330 |

Therefore, the final model was selected based on its measured validation performance rather than simply selecting the latest experiment.

The selected model is:

```text
models/best.pt
```

---

## 16. Practical Application — Demo

RoadGuard includes a Streamlit web application for demonstrating the trained object detector.

The application is located at:

```text
scripts/roadguard_demo.py
```

The application loads the trained model from:

```text
models/best.pt
```

The demo allows a user to:

1. Upload a road-scene image.
2. Run the trained YOLO model.
3. Detect objects in the image.
4. Display bounding boxes.
5. Display predicted object classes.
6. Display confidence scores.

---

## 17. Running the Demo

### Step 1 — Create a virtual environment

From the project root:

```bash
python -m venv .venv
```

### Step 2 — Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Start the Streamlit application

```bash
streamlit run scripts/roadguard_demo.py
```

The Streamlit application will open in the browser.

Upload a road-scene image to run RoadGuard and visualize the detected objects.

---

## 18. Running Evaluation

To evaluate the final model:

```bash
python scripts/evaluate_model.py
```

The evaluation script loads:

```text
models/best.pt
```

and evaluates it using:

```text
data/processed/roadguard_yolo/data.yaml
```

The generated evaluation results are saved to:

```text
runs/roadguard_evaluation/
```

---

## 19. Reproducing the Workflow

The main workflow can be reproduced from the project root.

### Inspect the original dataset

```bash
python scripts/inspect_zip.py
python scripts/inspect_dataset.py
python scripts/inspect_annotations.py
```

### Prepare the dataset

```bash
python scripts/prepare_dataset.py
```

### Validate the dataset

```bash
python scripts/validate_annotations.py
python scripts/validate_yolo_dataset.py
```

### Visualize samples

```bash
python scripts/visualize_samples.py
```

### Analyze object sizes

```bash
python scripts/analyze_object_sizes.py
```

### Train the model

```bash
python scripts/train_model.py
```

### Run experiments

```bash
python scripts/experiment_improvement.py
python scripts/experiment_4.py
```

### Evaluate the final model

```bash
python scripts/evaluate_model.py
```

### Visualize results

```bash
python scripts/visualize_results.py
```

### Run the practical demo

```bash
streamlit run scripts/roadguard_demo.py
```

---

## 20. Reports and Figures

All project documentation and supporting analysis are organized under:

```text
reports/
├── metrics/
├── figures/
└── RoadGuard_Final_Report.pdf
```

### Metrics

The `reports/metrics/` directory contains the numerical evaluation results and experiment comparisons.

### Figures

The `reports/figures/` directory contains selected figures generated during the project, including:

- Dataset visualizations
- Training results
- Confusion matrices
- Normalized confusion matrix
- Object-size analysis
- Model prediction examples

### Final Report

The complete academic/project report is available directly in the repository:

**[📄 RoadGuard Final Report](reports/RoadGuard_Final_Report.md)**

The report provides the complete documentation of the project, including the methodology, dataset preparation, model development, experiments, evaluation, optimization analysis, limitations, and conclusions.

---

## 21. Technologies

The project uses:

- **Python 3.11**
- **Ultralytics YOLO26n**
- **PyTorch**
- **OpenCV**
- **Pillow**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Streamlit**

---

## 22. Limitations

The current RoadGuard model has several limitations:

- Training was performed using CPU-only hardware.
- Only 5,000 training images and 1,000 validation images were used.
- Small objects are more difficult to detect.
- Traffic lights and traffic signs can occasionally be missed or confused.
- Some object classes have relatively few examples.
- The `train` class has very limited representation in the validation set.
- The lightweight YOLO26n architecture has limited capacity compared with larger object-detection models.

The results indicate that additional improvements could potentially be achieved through a larger training subset, higher input resolution, additional data augmentation, more extensive hyperparameter tuning, or a larger detection architecture.

---

## 23. Future Improvements

Potential future improvements include:

- Training on a larger portion of BDD100K.
- Increasing input image resolution.
- Improving data augmentation.
- Addressing class imbalance.
- Testing larger YOLO architectures.
- Performing broader hyperparameter optimization.
- Applying techniques specifically designed for small-object detection.
- Evaluating the model on an independent test dataset.
- Improving traffic-light and traffic-sign detection.

---

## 24. AI Tool Usage

AI tools, including ChatGPT, were used as supporting tools during development for planning, debugging, code organization, documentation, and interpretation of Machine Learning results.

Generated suggestions were independently verified by running the project code, validating the dataset, inspecting model predictions, and evaluating the trained models using the project's evaluation scripts.

The final technical decisions were based on the observed dataset characteristics, experimental results, and available hardware rather than relying solely on AI-generated recommendations.

---

## 25. Conclusion

RoadGuard provides a complete end-to-end Computer Vision object detection pipeline, covering dataset preparation, validation, model training, experimental comparison, evaluation, optimization analysis, and practical demonstration.

The final selected model achieved:

- **Precision:** 0.5364
- **Recall:** 0.2539
- **mAP50:** 0.2596
- **mAP50-95:** 0.1370

on the 1,000-image validation dataset.

The project demonstrates the capabilities and limitations of lightweight object detection under CPU-only resource constraints and provides a functional demonstration application for testing the trained model on road-scene images.

For the complete methodology, experiments, evaluation, analysis, and conclusions, see the **[RoadGuard Final Report](reports/RoadGuard_Final_Report.md)**.