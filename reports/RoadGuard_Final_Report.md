# RoadGuard — Final Technical Report

## 1. Problem Definition

RoadGuard is a computer vision object detection project designed to identify common objects in road scenes. The main goal is to develop a model that can automatically detect road users and traffic-related objects from images.

The system detects 10 object classes:

- Person
- Rider
- Bike
- Motor
- Car
- Truck
- Bus
- Train
- Traffic light
- Traffic sign

The project follows an end-to-end object detection workflow, starting with dataset preparation and annotation validation and continuing through model training, experimentation, evaluation, and a practical demonstration application.

The problem is challenging because road scenes contain objects at very different scales. Cars and people may occupy a relatively large portion of an image, while traffic lights, traffic signs, bikes, and other objects can be much smaller. Objects can also be partially occluded or appear in visually complex environments.

---

## 2. Dataset, Model, and Approach

### 2.1 Dataset Selection

The **BDD100K dataset** was selected because it is specifically designed for autonomous-driving and road-scene understanding tasks. It contains a large variety of real-world driving scenes and includes many of the object categories required by RoadGuard.

For this project, a subset of the dataset was prepared because training was performed using CPU-only hardware.

The prepared dataset contains:

- **5,000 training images**
- **1,000 validation images**
- **10 object classes**
- YOLO-format bounding-box annotations
- Input resolution of **416 × 416**

The dataset was converted into a YOLO-compatible structure containing separate image and label directories for training and validation.

Before training, the annotations and images were systematically validated. The validation process found no corrupt images or invalid bounding boxes in the prepared dataset.

### 2.2 Model Selection

The project uses **YOLO26n**, the nano-sized version of the YOLO26 object detection architecture available through Ultralytics.

A lightweight model was selected primarily because the project had to be developed and trained using a CPU rather than a dedicated GPU. A larger model would require substantially more computational resources and training time.

YOLO was selected because it provides an efficient end-to-end object detection approach capable of predicting both object locations and object classes in a single model.

### 2.3 Training Approach

The model was trained using a CPU-friendly configuration:

- Image size: **416 × 416**
- Batch size: **4**
- Device: **CPU**
- Workers: **0**
- Dataset caching: disabled

An initial baseline experiment was performed for 5 epochs. A longer training experiment was then performed and continued from a saved checkpoint after the computer was restarted. This allowed training to continue without losing the previously completed training progress.

Multiple experiments were subsequently performed to determine whether changes to the training configuration could improve the model.

---

## 3. Experiments and Results

Four main experiments were evaluated using the same validation dataset.

The models were compared using:

- Precision
- Recall
- mAP50
- mAP50-95

### 3.1 Experiment Comparison

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline | 0.5447 | 0.1966 | 0.1963 | 0.1075 |
| Experiment 2 | **0.5364** | 0.2539 | **0.2596** | **0.1370** |
| Experiment 3 | 0.5386 | 0.2622 | 0.2576 | 0.1353 |
| Experiment 4 | 0.5057 | **0.2669** | 0.2556 | 0.1330 |

The baseline achieved a Precision of 0.5447, but its Recall and mAP values were relatively low.

Experiment 2 produced a substantial improvement over the baseline:

- mAP50 increased from **0.1963 to 0.2596**
- mAP50-95 increased from **0.1075 to 0.1370**
- Recall increased from **0.1966 to 0.2539**

Experiment 3 produced slightly higher Recall than Experiment 2, but its mAP50 and mAP50-95 were slightly lower.

Experiment 4 achieved the highest Recall at **0.2669**, but its Precision decreased to **0.5057**, and its mAP values were also lower than Experiment 2.

Therefore, **Experiment 2 was selected as the strongest overall experiment**, since it achieved the highest mAP50 and mAP50-95.

### 3.2 Final Model Performance

The selected model achieved the following validation results:

| Metric | Result |
|---|---:|
| Precision | **0.5364** |
| Recall | **0.2539** |
| mAP50 | **0.2596** |
| mAP50-95 | **0.1370** |

The per-class mAP50 results were:

| Class | mAP50 |
|---|---:|
| Person | 0.3039 |
| Rider | 0.2383 |
| Bike | 0.1313 |
| Motor | 0.1701 |
| Car | **0.5810** |
| Truck | 0.3518 |
| Bus | 0.2988 |
| Train | 0.0000 |
| Traffic light | 0.2445 |
| Traffic sign | 0.2761 |

The model performed best on the **car** class, with a mAP50 of **0.5810**.

The lower results for several other classes are related to factors such as object size, class frequency, and the visual difficulty of detecting certain objects.

The `train` class had only one validation instance, so its mAP50 of 0.0000 is not a reliable indication of general train-detection performance.

---

## 4. Challenges and Solutions

Several challenges were encountered during development.

### 4.1 CPU-Only Training

The model was trained using a CPU, which significantly increased training time compared with GPU-based training.

To address this, a lightweight YOLO26n model was selected and the training configuration was adjusted to be more suitable for CPU execution. A relatively small input resolution and batch size were also used.

### 4.2 Training Interruption

During the longer training experiment, the computer had to be restarted. Rather than starting the training process from the beginning, the saved `last.pt` checkpoint was loaded and training was resumed.

This allowed the experiment to continue from the previously saved training state.

### 4.3 Small Objects

Object-size analysis showed that a large proportion of objects in the dataset are small.

For example, many of the objects belonging to classes such as bikes, trucks, and traffic-related categories occupy a small percentage of the image area. Small objects are generally more difficult for an object detector to localize accurately.

This was also reflected in the evaluation results, where traffic lights and traffic signs were more difficult to detect consistently.

### 4.4 Traffic Signs and Traffic Lights

During visual inspection of predictions, the model generally detected road objects correctly but occasionally missed or confused traffic signs and traffic lights.

These classes are challenging because they can appear at small sizes, at different distances, and against complex backgrounds.

Object-size analysis and confusion-matrix analysis were used to investigate these errors.

### 4.5 Experiment Selection

Not every additional experiment improved the model.

Although later experiments increased Recall slightly, they did not improve the overall mAP metrics. Therefore, the final model was selected based on the overall validation performance rather than simply selecting the latest experiment.

---

## 5. Strengths and Weaknesses

### 5.1 Strengths

The main strengths of the RoadGuard solution are:

- It provides a complete object detection pipeline from dataset preparation to deployment.
- The dataset was validated before training.
- The model detects 10 different road-scene object classes.
- Multiple training experiments were conducted rather than relying on a single configuration.
- The experiments were evaluated using consistent validation data.
- The final model performs particularly well on the car class.
- The project includes a practical demonstration application.
- The trained model is lightweight enough to run on CPU hardware.

The project also includes supporting analysis such as object-size analysis and confusion-matrix evaluation, which helped identify the main sources of model errors.

### 5.2 Weaknesses

The main weakness is the relatively low overall Recall and mAP compared with what would be expected from a highly optimized production-level road-object detector.

The final Recall of **0.2539** indicates that the model misses a considerable number of objects.

The model also performs considerably better on some classes than others. In particular, small and less frequent objects are more difficult to detect.

Additional limitations include:

- Only a subset of BDD100K was used.
- Training was performed on CPU hardware.
- The model uses a lightweight architecture.
- Input resolution was limited to 416 × 416.
- Some classes have substantially fewer examples than others.
- The validation set contains very few examples of some categories.
- Traffic signs and traffic lights remain challenging.
- The experiments were limited by the available development time and computational resources.

---

## 6. Future Improvements

With additional time and computational resources, several improvements could be investigated.

### 6.1 Larger Training Dataset

A larger portion of BDD100K could be used for training. More training examples would provide greater variation in road conditions, object appearances, distances, and lighting conditions.

### 6.2 GPU-Based Training

Training on a GPU would significantly reduce training time and make it possible to perform more extensive experimentation.

This would allow testing:

- More epochs
- Larger batch sizes
- Higher image resolutions
- More model architectures
- More hyperparameter configurations

### 6.3 Higher Image Resolution

Increasing the input resolution could help detect small objects such as traffic signs and traffic lights.

However, higher resolution would also increase computational requirements.

### 6.4 Larger Model Architecture

A larger YOLO architecture could be tested to determine whether additional model capacity improves detection performance.

The trade-off would be increased inference time and computational requirements.

### 6.5 Improved Small-Object Detection

Since the object-size analysis showed that many objects are small, future work could focus specifically on small-object detection.

Possible approaches include:

- Higher-resolution training
- More appropriate augmentation
- Multi-scale training
- Increasing the representation of small objects in the training data
- Testing architectures or configurations better suited to small-object detection

### 6.6 Class Imbalance

Some object classes have substantially more training examples than others. Additional data balancing techniques could be investigated to improve performance on less represented classes.

### 6.7 More Extensive Hyperparameter Optimization

More systematic experimentation could be performed with:

- Learning rate
- Batch size
- Image size
- Augmentation parameters
- Optimizer configuration
- Confidence thresholds
- IoU thresholds

This would make it possible to determine whether the model can achieve better performance with a more carefully optimized configuration.

---

## 7. Conclusion

RoadGuard successfully implements a complete road-scene object detection pipeline using BDD100K and YOLO26n.

The project included dataset preparation, annotation validation, model training, multiple experiments, quantitative evaluation, object-size analysis, error analysis, and a practical demonstration application.

Four experiments were conducted. The second experiment achieved the strongest overall validation performance, with:

- **Precision:** 0.5364
- **Recall:** 0.2539
- **mAP50:** 0.2596
- **mAP50-95:** 0.1370

The results demonstrate that the lightweight model can detect common road objects, particularly cars, but also show clear limitations when detecting smaller and less frequent objects.

The experiments also demonstrated that additional training does not automatically result in better performance. Experiment 4 achieved slightly higher Recall than Experiment 2, but its overall mAP and Precision were lower. Therefore, Experiment 2 was selected based on the overall evaluation metrics.

The final RoadGuard model is stored as:

```text
models/best.pt
```

A Streamlit-based demonstration is also included, allowing the trained model to be tested on new road-scene images.

Overall, the project demonstrates the complete process of developing and evaluating a practical computer vision object detector under limited computational resources, while identifying clear areas for future improvement.