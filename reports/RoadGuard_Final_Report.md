# RoadGuard — Computer Vision Object Detection System

  

## 1. Introduction

  

RoadGuard is a Computer Vision-based object detection system designed to identify common objects in road-scene images. The project was developed as an end-to-end Machine Learning workflow, covering dataset preparation, model training and fine-tuning, experimentation, evaluation, optimization, and practical deployment.

  

The system uses the BDD100K dataset and a lightweight YOLO26n object detection model. The final system is capable of detecting ten classes of road-scene objects: person, rider, bike, motor, car, truck, bus, train, traffic light, and traffic sign.

  

The project was developed using CPU-only hardware. Because of this resource limitation, a lightweight model and relatively small image resolution were selected to make training feasible while still producing a complete working Computer Vision pipeline.

  

The main objective was not only to train an object detector, but to investigate the complete development process and understand the strengths and limitations of the resulting system.

  

---

  

# 2. Data Processing

  

## 2.1 Dataset

  

The project uses the BDD100K dataset, which is designed for road-scene analysis and object detection. A subset of the dataset was selected for this project.

  

The final dataset contains:

  

- 5,000 training images

- 1,000 validation images

- 10 object classes

- YOLO-format annotations

- 92,407 valid training bounding boxes

- 18,393 valid validation bounding boxes

  

The processed dataset is stored in:

  

`data/processed/roadguard_yolo/`

  

The dataset configuration is defined in `data.yaml`.

  

The ten classes are:

  

| ID | Class |

|---:|---|

| 0 | person |

| 1 | rider |

| 2 | bike |

| 3 | motor |

| 4 | car |

| 5 | truck |

| 6 | bus |

| 7 | train |

| 8 | traffic light |

| 9 | traffic sign |

  

## 2.2 Dataset Preparation

  

The original data was inspected before model training. The preparation process included examining the dataset structure and annotations, selecting the required data, converting the annotations into YOLO format, and organizing the images and labels into training and validation directories.

  

The prepared dataset was then validated before training.

  

The validation produced the following results:

  

| Check | Training | Validation |

|---|---:|---:|

| Images found | 5,000 | 1,000 |

| Label files found | 5,000 | 1,000 |

| Readable images | 5,000 | 1,000 |

| Valid boxes | 92,407 | 18,393 |

| Errors | 0 | 0 |

  

All dataset checks passed.

  

Sample images were also visualized with their bounding boxes. This manual inspection was useful for confirming that the generated annotations were correctly aligned with the objects in the images.

  

## 2.3 Object Size Analysis

  

An additional object-size analysis was performed to understand the distribution of objects in the training data.

  

The analysis showed that a large proportion of the objects are small relative to the image dimensions. This is particularly important for road-scene detection because traffic lights, traffic signs, bikes, and distant vehicles can occupy very small regions of an image.

  

For example, the analysis showed that 99.9% of the detected train objects and 98.2% of truck objects in the analyzed training annotations fell into the small-object category according to the project's area thresholds. The analysis therefore provided useful context when interpreting the model's difficulties with certain object classes.

  

The detailed object-size analysis is included in the project figures.

  

---

  

# 3. Model Creation and Adaptation

  

## 3.1 Model Selection

  

YOLO26n was selected as the base object detection model.

  

The nano version was chosen primarily because the available training hardware was CPU-only. A larger model would require substantially more computational resources and would increase training time.

  

The final fused model contains approximately:

  

- 2.39 million parameters

- 5.4 GFLOPs

  

This makes it substantially more suitable for a resource-constrained environment than a larger object detection architecture.

  

The pretrained model was fine-tuned on the prepared RoadGuard dataset.

  

## 3.2 Training Configuration

  

The main CPU-friendly configuration used:

  

| Parameter | Value |

|---|---|

| Device | CPU |

| Image size | 416 × 416 |

| Batch size | 4 |

| Workers | 0 |

| Cache | False |

  

The initial baseline experiment used five epochs. Since this was insufficient for the model to reach stable performance, a longer training experiment was conducted.

  

The longer experiment was trained for a total of 50 epochs. During development, the computer was restarted while training was in progress. The experiment was resumed from the saved `last.pt` checkpoint rather than restarting training from the beginning.

  

This allowed the training process to continue while preserving the progress already made.

  

---

  

# 4. Experiments

  

Multiple experiments were conducted to determine whether changes to the training configuration could improve model performance.

  

The experiments were evaluated using the same 1,000-image validation dataset. The main metrics used for comparison were Precision, Recall, mAP50, and mAP50-95.

  

## 4.1 Experiment Results

  

| Experiment | Precision | Recall | mAP50 | mAP50-95 |

|---|---:|---:|---:|---:|

| Baseline | 0.5447 | 0.1966 | 0.1963 | 0.1075 |

| Experiment 2 | 0.5364 | 0.2539 |  **0.2596**  |  **0.1370**  |

| Experiment 3 | 0.5386 | 0.2622 | 0.2576 | 0.1353 |

| Experiment 4 | 0.5057 |  **0.2669**  | 0.2556 | 0.1330 |

  

## 4.2 Experiment Analysis

  

The baseline achieved a Precision of 0.5447, but its Recall was only 0.1966. Its mAP50 was 0.1963 and mAP50-95 was 0.1075.

  

Experiment 2 produced a substantial improvement in the overall detection metrics. Its mAP50 increased from 0.1963 to 0.2596, while mAP50-95 increased from 0.1075 to 0.1370. Recall also increased from 0.1966 to 0.2539.

  

Experiment 3 achieved a slightly higher Recall of 0.2622, but its mAP50 and mAP50-95 were slightly lower than Experiment 2.

  

Experiment 4 achieved the highest Recall at 0.2669. However, its Precision decreased to 0.5057, and its mAP50 and mAP50-95 also decreased compared with Experiment 2.

  

Therefore, the experiments demonstrated an important trade-off between Recall and the overall detection metrics. The experiment with the highest Recall was not necessarily the experiment with the best overall performance.

  

Based on mAP50 and mAP50-95, Experiment 2 was selected as the strongest overall configuration.

  

---

  

# 5. Model Evaluation

  

The selected model was evaluated on the 1,000-image validation dataset.

  

The final overall results were:

  

| Metric | Result |

|---|---:|

| Precision |  **0.5364**  |

| Recall |  **0.2539**  |

| mAP50 |  **0.2596**  |

| mAP50-95 |  **0.1370**  |

  

## 5.1 Per-Class Performance

  

The mAP50 values for individual classes were:

  

| Class | mAP50 |

|---|---:|

| person | 0.3039 |

| rider | 0.2383 |

| bike | 0.1313 |

| motor | 0.1701 |

| car |  **0.5810**  |

| truck | 0.3518 |

| bus | 0.2988 |

| train | 0.0000 |

| traffic light | 0.2445 |

| traffic sign | 0.2761 |

  

The strongest performance was achieved for the `car` class, with a mAP50 of 0.5810.

  

The model also performed reasonably better on trucks, buses, and people compared with several of the smaller object categories.

  

The weakest results were observed for train, bike, and motor. The train result should be interpreted cautiously because there was only one train instance in the validation dataset.

  

Traffic lights and traffic signs also remained challenging. This is consistent with the object-size analysis and with manual inspection of the model's predictions, where these objects were sometimes missed or incorrectly classified.

  

## 5.2 Confusion Matrix Analysis

  

A confusion matrix and normalized confusion matrix were generated to investigate classification errors.

  

The confusion matrix provides information beyond the overall mAP values by showing which classes are confused with one another.

  

The normalized confusion matrix was particularly useful for identifying the relative distribution of classification errors between classes.

  

The analysis indicated that the model was considerably stronger for common road objects such as cars, while smaller or less frequent classes were more difficult.

  

The confusion-matrix figures are included in the project report.

  

## 5.3 Prediction Analysis

  

Visual inspection of predictions showed that the model was capable of correctly detecting many of the major road-scene objects.

  

Cars and people were generally detected successfully, while traffic lights and traffic signs were more likely to be missed or incorrectly detected.

  

This qualitative analysis was important because numerical metrics alone do not fully describe the behavior of an object detector.

  

---

  

# 6. Optimization

  

Optimization was investigated through multiple experiments rather than assuming that a single training configuration would be optimal.

  

The main objective was to improve detection performance while keeping the model practical for the available CPU hardware.

  

The experiments demonstrated that longer training and alternative configurations could substantially improve performance over the initial five-epoch baseline. However, the later experiments did not improve the overall mAP beyond Experiment 2.

  

The comparison was:

  

| Experiment | mAP50 | mAP50-95 |

|---|---:|---:|

| Baseline | 0.1963 | 0.1075 |

| Experiment 2 |  **0.2596**  |  **0.1370**  |

| Experiment 3 | 0.2576 | 0.1353 |

| Experiment 4 | 0.2556 | 0.1330 |

  

Experiment 2 therefore remained the selected model.

  

This is an important result of the optimization stage: optimization does not necessarily mean that every subsequent experiment must improve the model. The experiments were used to test alternative approaches and determine which configuration provided the best measured result.

  

Further optimization would likely require additional computational resources and more extensive experimentation.

  

---

  

# 7. Practical Application

  

The final RoadGuard model was integrated into a Streamlit application.

  

The demonstration application is located at:

  

`scripts/roadguard_demo.py`

  

The application loads the trained model from:

  

`models/roadguard_best.pt`

  

The user can upload a new road-scene image and run the trained object detector. The application displays the detected objects using bounding boxes and provides the predicted class information and confidence values.

  

This demonstrates that the trained model is not limited to evaluation on the validation dataset and can be used as part of a simple practical application.

  

The project therefore includes a complete path from dataset preparation to a usable Computer Vision application.

  

---

  

# 8. Project Structure

  

The final repository is organized as follows:

  

```text

RoadGuard/

│

├── data/

│ ├── raw/

│ └── processed/

│ └── roadguard_yolo/

│ ├── images/

│ │ ├── train/

│ │ └── val/

│ ├── labels/

│ │ ├── train/

│ │ └── val/

│ └── data.yaml

│

├── models/

│ └── roadguard_best.pt

│

├── runs/

│ ├── roadguard_cpu_baseline/

│ ├── roadguard_experiment_2/

│ ├── roadguard_experiment_3/

│ ├── roadguard_experiment_4/

│ └── roadguard_evaluation/

│

├── scripts/

│ ├── analyze_object_sizes.py

│ ├── evaluate_model.py

│ ├── experiment_4.py

│ ├── experiment_improvement.py

│ ├── inspect_annotations.py

│ ├── inspect_dataset.py

│ ├── inspect_zip.py

│ ├── prepare_dataset.py

│ ├── roadguard_demo.py

│ ├── train_model.py

│ ├── validate_annotations.py

│ ├── validate_yolo_dataset.py

│ ├── visualize_results.py

│ ├── visualize_samples.py

│ └── yolo26n.pt

│

├── reports/

│ ├── metrics/

│ └── figures/

│

├── .gitignore

├── README.md

└── requirements.txt

```

  

---

  

# 9. Technologies

  

The project was implemented using:

  

- Python 3.11

- Ultralytics YOLO26n

- PyTorch

- OpenCV

- Pillow

- NumPy

- Pandas

- Matplotlib

- Streamlit

  

The use of Ultralytics provides the object detection training and evaluation framework, while PyTorch serves as the underlying machine learning framework.

  

---

  

# 10. Strengths and Limitations

  

## Strengths

  

The main strengths of the RoadGuard solution are:

  

- Complete end-to-end Computer Vision workflow.

- Proper dataset validation before training.

- 5,000 training images and 1,000 validation images.

- Multiple training experiments instead of relying on a single configuration.

- Quantitative evaluation using several object-detection metrics.

- Per-class performance analysis.

- Confusion-matrix analysis.

- Object-size analysis.

- Practical demonstration through a Streamlit application.

- Lightweight model suitable for CPU-based inference.

- Final trained model included separately for practical use.

  

## Limitations

  

The model also has several limitations.

  

First, training was performed using CPU-only hardware. This significantly limited the number and complexity of experiments that could be performed within the available time.

  

Second, the model was trained using a relatively small subset of the full BDD100K dataset. A larger and more diverse training set could potentially improve generalization.

  

Third, small objects remain difficult to detect. Traffic lights and traffic signs can occupy very small areas of road-scene images, making accurate detection challenging.

  

Fourth, some classes have significantly fewer examples than others. This class imbalance can contribute to differences in per-class performance.

  

Finally, the overall Recall and mAP values show that the current detector still misses a meaningful number of objects. The system should therefore be viewed as a working Computer Vision prototype rather than a production-ready autonomous driving perception system.

  

---

  

# 11. Future Improvements

  

If additional time and computational resources were available, several improvements could be investigated.

  

### Larger training dataset

  

Increasing the number of training images would provide the model with more examples of difficult road-scene situations.

  

### Higher image resolution

  

Increasing the input resolution could help the model detect small objects such as traffic lights and traffic signs.

  

### Improved data augmentation

  

Additional augmentation techniques could be tested to improve robustness to different lighting conditions, camera perspectives, object scales, and road environments.

  

### Class balancing

  

The training data could be analyzed and balanced more carefully to improve performance on underrepresented classes.

  

### Larger model

  

A larger YOLO architecture could be evaluated if GPU resources were available. This could provide greater model capacity at the cost of increased computation and inference time.

  

### More extensive hyperparameter optimization

  

A broader search over learning rate, batch size, image size, augmentation, and other training parameters could potentially produce better results.

  

---

  

# 12. Conclusion

  

RoadGuard demonstrates a complete Computer Vision object-detection workflow, beginning with raw road-scene data and ending with a working application.

  

The dataset was prepared and validated successfully, with zero validation errors across the 5,000-image training set and 1,000-image validation set. A lightweight YOLO26n model was then fine-tuned using CPU-compatible training settings.

  

Four experiments were conducted and compared using Precision, Recall, mAP50, and mAP50-95. Experiment 2 achieved the strongest overall performance, with a Precision of 0.5364, Recall of 0.2539, mAP50 of 0.2596, and mAP50-95 of 0.1370.

  

The experiments also demonstrated that the highest Recall does not necessarily correspond to the best overall detector. Experiment 4 achieved the highest Recall, but its lower Precision and mAP values made Experiment 2 the stronger overall choice.

  

The evaluation and qualitative analysis showed that the model performs particularly well on common objects such as cars, while small and less frequent objects remain challenging. The object-size analysis and confusion matrix provided additional evidence for understanding these limitations.

  

Finally, the trained model was integrated into a Streamlit application, allowing users to upload new road-scene images and visualize the model's detections.

  

Overall, the project satisfies the complete Computer Vision workflow from data processing and model adaptation through experimentation, evaluation, optimization, and practical application, while clearly identifying the limitations of the current solution and possible directions for future improvement.

  

# 13. AI Tools Usage

  

AI tools, including ChatGPT, were used during the development of the RoadGuard project as a supporting tool for planning, debugging, code organization, documentation, and interpretation of Machine Learning results.

  

The generated suggestions were not treated as automatically correct. Dataset validation, model training, evaluation results, experiment comparisons, and the behavior of the final application were independently checked by running the corresponding code and inspecting the resulting outputs.

  

The final technical decisions were based on the observed dataset characteristics, experimental results, and available CPU hardware rather than on AI-generated recommendations alone.
