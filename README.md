# MedVisionAI

## Breast Ultrasound Classification Using Deep Learning and Transfer Learning

MedVisionAI is a deep learning computer vision project exploring binary breast ultrasound image classification using convolutional neural networks and transfer learning.

The project evolved through two major experimental versions:

* **Version 1 — BreastMNIST:** Initial experimentation using the MedMNIST benchmark dataset. (Yang, J., et al. "MedMNIST v2 - A large-scale lightweight benchmark 
for 2D and 3D biomedical image classification." Scientific Data, 2023.)

* **Version 2 — BUSI:** A more realistic breast ultrasound classification experiment using the Breast Ultrasound Images Dataset (BUSI).

The project was developed to explore the complete machine learning workflow:

```text
Dataset
   ↓
Data Exploration
   ↓
Preprocessing
   ↓
Data Augmentation
   ↓
Model Development
   ↓
Transfer Learning
   ↓
Training Optimization
   ↓
Evaluation
   ↓
Explainability
   ↓
Deployment
```

The project currently focuses on binary classification:

```text
0 → Benign
1 → Malignant
```

> ⚠️ **Important:** MedVisionAI is an educational and research project. It is not a medical diagnostic system and must not be used for clinical decision-making.

---

# Live Demo

The MedVisionAI application is deployed using Streamlit Community Cloud.

🔗 **Live Demo:** https://medvisionai-web.streamlit.app

The application allows users to:

* Upload a breast ultrasound image
* Receive a benign or malignant prediction
* View the model's confidence score
* Generate a Grad-CAM visualization showing regions that influenced the prediction

---

# Project Evolution

## Version 1 — BreastMNIST

The first version of MedVisionAI used the **BreastMNIST** dataset from the MedMNIST collection.

The purpose of Version 1 was to:

* Build the initial data pipeline
* Experiment with medical image classification
* Develop a baseline CNN
* Explore transfer learning with ResNet18
* Implement training and evaluation pipelines
* Experiment with grayscale adaptation of pretrained models

The initial model achieved strong performance on the BreastMNIST benchmark dataset.

However, BreastMNIST consists of small 28 × 28 pixel images. This made it useful for experimentation but less representative of a realistic high-resolution ultrasound classification problem.

---

## Version 2 — BUSI

The second version transitioned to the **Breast Ultrasound Images Dataset (BUSI)**.

This change was motivated by the desire to experiment with larger, more realistic breast ultrasound images.

Version 2 introduced:

* A larger image resolution
* A custom BUSI dataset loader
* A new train-validation-test split
* Transfer learning using ResNet18
* Data augmentation
* Class-weighted loss
* Learning-rate scheduling
* Early stopping
* Model checkpointing
* Grad-CAM explainability
* Streamlit deployment

The transition from BreastMNIST to BUSI represents the main development progression of the project.

---

# Dataset

## BUSI — Breast Ultrasound Images Dataset

The current version of MedVisionAI uses the Breast Ultrasound Images Dataset (BUSI).

The dataset contains breast ultrasound images categorized into:

* Normal
* Benign
* Malignant

For this project, the classification task was simplified to binary classification:

| Class     | Label |
| --------- | ----: |
| Benign    |     0 |
| Malignant |     1 |

Normal images were excluded from the current binary classification experiment.

The dataset contains approximately 780 ultrasound images, originally collected from 600 women between the ages of 25 and 75.

Dataset source:

https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset

---

## Dataset Characteristics

* Breast ultrasound images
* Grayscale medical images
* PNG format
* Binary classification task
* Small-scale medical imaging dataset
* Class imbalance between benign and malignant samples

The dataset contains both ultrasound images and segmentation masks. Segmentation mask files were excluded from the classification dataset.

---

# Dataset Splitting

BUSI contains many near-duplicate images: several ultrasound frames of the same lesion, sometimes with different caliper marks or annotations. A plain random split puts near-copies of training images into the test set, which inflates test scores.

`scripts/find_duplicates.py` groups images by lesion using a perceptual hash (difference hash, 16×16, Hamming distance ≤ 55, checked visually) and writes `data/busi_groups.json`:

```text
647 images → 509 lesion groups
115 groups contain near-duplicates (253 images)
14 images appear under both benign and malignant
```

The 14 conflicting images have an unknown true label and are excluded from training and evaluation.

The remaining 633 images are divided with `StratifiedGroupKFold` (seed 42), so every image of a lesion lands in the same split and each split keeps the benign / malignant ratio:

```text
~70% Training    (451 images)
~15% Validation  (91 images)
~15% Testing     (91 images)
```

To regenerate the groups:

```bash
python -m scripts.find_duplicates
```

The training set uses data augmentation, while the validation and test sets use deterministic preprocessing.

```text
Training Dataset
        ↓
Resize
        ↓
Random Rotation
        ↓
Random Translation
        ↓
Tensor Conversion
        ↓
Normalization
```

Validation and test datasets:

```text
Validation/Test Dataset
        ↓
Resize
        ↓
Tensor Conversion
        ↓
Normalization
```

---

# Project Pipeline

```text
                BUSI Dataset
                     ↓
           Dataset Exploration
                     ↓
              Data Cleaning
                     ↓
          Remove Segmentation Masks
                     ↓
           Binary Classification
          Benign vs Malignant
                     ↓
              Dataset Splitting
            70% / 15% / 15%
                     ↓
              Image Preprocessing
                     ↓
              Data Augmentation
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
    CNN Baseline          ResNet18 Transfer
                              Learning
                                  ↓
                       Architecture Adaptation
                                  ↓
                       Grayscale Input Support
                                  ↓
                       Binary Classification Head
                                  ↓
                              Training
                                  ↓
                     Weighted Cross Entropy Loss
                                  ↓
                         Adam Optimizer
                                  ↓
                     Learning Rate Scheduling
                                  ↓
                          Early Stopping
                                  ↓
                       Best Model Checkpoint
                                  ↓
                            Evaluation
                                  ↓
            Accuracy | Precision | Recall | F1
                                  ↓
                            Grad-CAM
                                  ↓
                             Deployment
```

---

# Model Development

Two main approaches were explored.

## 1. Custom CNN Baseline

A custom convolutional neural network was developed as an initial baseline.

The purpose of the CNN was to:

* Establish an initial performance benchmark
* Test the classification pipeline
* Understand the difficulty of the dataset
* Compare training from scratch with transfer learning

The CNN baseline provided a reference point for evaluating the benefits of pretrained models.

---

## 2. ResNet18 Transfer Learning

The primary model uses ResNet18.

Instead of training a deep neural network entirely from scratch, the model uses visual features learned from ImageNet and adapts the architecture for grayscale breast ultrasound classification.

Transfer learning was selected because medical imaging datasets are often relatively small compared with datasets such as ImageNet.

Potential advantages include:

* Faster convergence
* Reuse of learned visual features
* Improved feature extraction
* Reduced need for extremely large datasets

---

# ResNet18 Architecture Adaptation

The original ResNet18 architecture expects RGB images.

```text
Original Input:

RGB Image
   ↓
3 Channels
   ↓
ResNet18
   ↓
1000 ImageNet Classes
```

The MedVisionAI model was adapted for grayscale ultrasound images:

```text
Grayscale Ultrasound Image
   ↓
1 Channel
   ↓
Modified ResNet18
   ↓
Feature Extraction
   ↓
2-Class Classifier
   ↓
Benign / Malignant
```

---

## Grayscale Input Modification

The original ResNet18 first convolutional layer expects three input channels:

```python
Conv2d(
    in_channels=3,
    out_channels=64
)
```

The model was modified to accept a single grayscale channel:

```python
Conv2d(
    in_channels=1,
    out_channels=64
)
```

To initialize the grayscale convolution using the pretrained RGB filters, the RGB weights were averaged across the channel dimension to produce a single-channel initialization.

```python
new_conv.weight[:] = old_conv.weight.mean(
    dim=1,
    keepdim=True
)
```

This produces a single-channel convolutional layer initialized using the pretrained RGB filters.

---

## Classification Head

The original ResNet18 model was designed for ImageNet classification:

```text
1000 Classes
```

The final classification layer was replaced with a binary classifier:

```python
self.model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(
        self.model.fc.in_features,
        2
    )
)
```

The output classes are:

```text
0 → Benign
1 → Malignant
```

---

# Transfer Learning Strategy

The model initially freezes the pretrained ResNet18 parameters.

Selected deeper layers are then fine-tuned:

```text
Frozen:
- Initial layers
- Layer 1

Trainable:
- Layer 2
- Layer 3
- Layer 4
- Classification Head
```

This allows the model to preserve general visual features while adapting deeper feature representations to the breast ultrasound classification task.

---

# Image Preprocessing

The original BUSI images have significantly larger dimensions than the 224 × 224 input size expected by ResNet18.

Images are resized to:

```text
224 × 224 pixels
```

The images are then converted to tensors and normalized.

The current preprocessing pipeline is:

```text
Original Ultrasound Image
        ↓
Grayscale Conversion
        ↓
Resize to 224 × 224
        ↓
Tensor Conversion
        ↓
Normalization
        ↓
ResNet18
```

---

# Data Augmentation

To reduce overfitting and improve robustness, the training dataset uses augmentation.

## Random Rotation

```python
transforms.RandomRotation(
    degrees=5
)
```

This introduces small changes in orientation.

---

## Random Translation

```python
transforms.RandomAffine(
    degrees=0,
    translate=(0.05, 0.05)
)
```

This introduces small positional changes.

The validation and test datasets are not augmented.

---

# Training Strategy

The training pipeline includes several optimization techniques.

## Weighted Cross Entropy Loss

The BUSI dataset contains class imbalance.

To reduce bias toward the majority class, weighted cross-entropy loss was used:

```python
CrossEntropyLoss(
    weight=class_weights
)
```

This increases the contribution of underrepresented classes during training.

---

## Optimizer

The model is trained using AdamW (weight decay 1e-4).

The new classifier head uses a 10x higher learning rate than the pretrained backbone.

---

## Learning Rate Scheduling

A cosine annealing scheduler gradually lowers the learning rate over the training run.

---

## Early Stopping

Early stopping is used to reduce overfitting.

Training stops when validation performance fails to improve for a predefined number of epochs.

The checkpoint with the best validation F1 score is saved.

```text
Training
   ↓
Validation F1 Improves
   ↓
Save Best Checkpoint
   ↓
Validation Stops Improving
   ↓
Early Stopping
   ↓
Evaluate Best Checkpoint on Test Set
```

---

# Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall (sensitivity) and specificity
* F1 Score
* ROC AUC and PR AUC
* Calibration (Brier score, Expected Calibration Error, reliability diagram)
* Confusion Matrix
* 95% bootstrap confidence intervals for every metric

The F1 score is particularly important because the dataset contains class imbalance and because both precision and recall are important for the classification task.

---

# BUSI Model Performance

The final reported BUSI results should be kept as a separate experimental result from the BreastMNIST results.

The current BUSI model (retrained October 2026 on a lesion-grouped 70 / 15 / 15 split, seed 42; see Dataset Splitting) achieved the following on the held-out test set of 91 images (62 benign, 29 malignant):

| Metric                       |  Score | 95% CI (bootstrap) |
| ---------------------------- | -----: | :----------------: |
| Accuracy                     | 83.52% |   75.8% – 90.1%    |
| Precision (PPV)              | 73.33% |   56.5% – 88.0%    |
| Recall (Sensitivity)         | 75.86% |   58.8% – 90.9%    |
| Specificity                  | 87.10% |   77.9% – 94.6%    |
| F1 Score                     | 74.58% |   60.0% – 85.3%    |
| ROC AUC                      |  0.889 |   0.801 – 0.956    |
| PR AUC (Average Precision)   |  0.865 |   0.756 – 0.944    |
| Brier Score ↓                |  0.124 |   0.086 – 0.167    |
| Expected Calibration Error ↓ |  0.095 |   0.081 – 0.192    |

Confidence intervals come from 2,000 bootstrap resamples of the test set. Their width shows how much a ~90-image test set limits the precision of any single number. For example, the true accuracy could plausibly be anywhere from about 76% to 90%.

### Effect of the leak-free split

The previous model was trained on a plain stratified split. It scored 86.7% accuracy, 96.9% recall and 0.955 ROC AUC, partly because near-duplicate frames of the same lesion appeared in both training and test. With the lesions grouped, ROC AUC drops to 0.889 and recall to 75.9%. The lower numbers are a more honest estimate of how the model handles lesions it has never seen.

![ROC and Precision-Recall Curves](results/busi/roc_pr_curves.png)

<p>
  <img src="results/busi/confusion_matrix.png" width="45%" alt="Confusion Matrix">
  <img src="results/busi/calibration.png" width="45%" alt="Reliability Diagram">
</p>

### Calibration

`scripts/evaluate_busi.py` applies **temperature scaling** (Guo et al., 2017). It divides the model's logits by a single scalar T, fitted on the validation split by minimising negative log-likelihood. The test set is never used for fitting. T is saved under `"calibration"` in `models/checkpoints/resnet18_busi_metrics.json`, and the web app divides by the same T before showing a confidence score. Because one T rescales both logits equally, the predicted class never changes. Accuracy, recall and ROC AUC are therefore identical before and after.

| Test set (91 images)         | Uncalibrated | Temperature scaled (T = 1.04) |
| ---------------------------- | -----------: | ----------------------------: |
| Brier Score ↓                |        0.124 |                         0.124 |
| Expected Calibration Error ↓ |        0.095 |                         0.106 |
| Negative log-likelihood ↓    |        0.375 |                         0.375 |

**Temperature scaling did not help this model.** The fitted temperature is essentially 1, which means the retrained model's overall confidence level is already about right on the validation set. The calibration error that remains is not uniform over- or under-confidence, so a single scalar cannot remove it. The reliability diagram shows the problem is concentrated in the 0.5–0.7 range: there, the model says "60% malignant" for lesions that turn out malignant only about 20% of the time. Elsewhere, the curve is close to the diagonal. The small change in ECE (0.095 → 0.106) is well inside the bootstrap interval, and ECE with 10 bins on 91 images is itself noisy.

In practice, treat mid-range confidence scores in the web app (roughly 50–70%) as "uncertain" rather than as probabilities. Calibrators that can move the decision boundary, such as Platt scaling (a temperature plus a bias), were not adopted. With only 91 validation images, extra parameters risk fitting noise.

Confusion Matrix:

```text
[[54  8]
 [ 7 22]]
```

Interpretation:

```text
54 benign images were correctly classified.

22 malignant images were correctly classified.

8 benign images were incorrectly classified as malignant.

7 malignant images were incorrectly classified as benign.
```

Training writes these metrics to `models/checkpoints/resnet18_busi_metrics.json`, which the web app reads for its sidebar.

> These results should be interpreted in the context of the relatively small dataset and the specific train-validation-test split used in this experiment.

---

# BUSI Cross-Validation Results

A single ~90-image test set is noisy, so the model was also evaluated with lesion-grouped 5-fold cross-validation (`python -m scripts.cross_validate`). Five models were trained, and each of the 633 usable images was tested exactly once.

| Metric      | Mean ± SD (5 folds) | Pooled (633 images) | 95% CI (bootstrap) |
| ----------- | :-----------------: | :-----------------: | :----------------: |
| Accuracy    |   84.0% ± 7.3%      |        84.0%        |   81.2% – 87.0%    |
| Precision   |   75.4% ± 11.3%     |        74.6%        |   69.0% – 80.5%    |
| Recall      |   76.5% ± 9.8%      |        76.5%        |   70.6% – 82.0%    |
| Specificity |   87.6% ± 6.9%      |        87.6%        |   84.6% – 90.7%    |
| F1 Score    |   75.8% ± 9.9%      |        75.5%        |   70.9% – 80.0%    |
| ROC AUC     |   0.896 ± 0.084     |        0.912        |   0.889 – 0.934    |

Per fold:

| Fold | Accuracy | Recall | Specificity |   F1  | ROC AUC |
| :--: | :------: | :----: | :---------: | :---: | :-----: |
|  1   |   87.4%  |  85.4% |    88.4%    | 81.4% |  0.943  |
|  2   |   85.0%  |  70.7% |    91.9%    | 75.3% |  0.926  |
|  3   |   88.2%  |  78.0% |    93.0%    | 81.0% |  0.908  |
|  4   |   69.8%  |  61.0% |    74.1%    | 56.8% |  0.733  |
|  5   |   89.7%  |  87.5% |    90.7%    | 84.3% |  0.971  |

> Fold 4 is a model-selection failure, not a harder fold. Its epoch-1 model happened to score the best validation F1 on only 85 validation images, so early stopping kept an almost untrained model. Without fold 4, accuracy is 86–90% and ROC AUC 0.91–0.97. Selecting checkpoints by validation loss or ROC AUC, with a minimum number of epochs, would make training more robust.

---

# BreastMNIST Version 1 Results

The original BreastMNIST experiment achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 88.46% |
| Precision | 91.38% |
| Recall    | 92.98% |
| F1 Score  | 92.17% |

Confusion Matrix:

```text
[[32 10]
 [ 8 106]]
```

These results belong to the original BreastMNIST experiment and should not be presented as the BUSI model's results.

---

# Why the Results Should Not Be Directly Compared

The BreastMNIST and BUSI experiments use different datasets.

They differ in:

* Dataset size
* Image resolution
* Data distribution
* Class composition
* Dataset splitting
* Image characteristics

Therefore, the performance metrics should not be interpreted as a direct benchmark between the two models.

The purpose of the two versions is to demonstrate the evolution of the project:

```text
Version 1:
BreastMNIST
   ↓
Initial Experimentation
   ↓
Model Development

Version 2:
BUSI
   ↓
More Realistic Ultrasound Dataset
   ↓
Custom Dataset Pipeline
   ↓
Further Transfer Learning Experiments
```

---

# Explainable AI — Grad-CAM

MedVisionAI integrates Gradient-weighted Class Activation Mapping (Grad-CAM).

Grad-CAM generates a heatmap showing regions of an image that contributed to the model's prediction.

```text
Ultrasound Image
        ↓
ResNet18
        ↓
Prediction
        ↓
Gradient Analysis
        ↓
Grad-CAM Heatmap
        ↓
Important Image Regions
```

Grad-CAM is useful because classification accuracy alone does not explain why a model made a prediction.

---

## Correct Classification Example

![Correct Grad-CAM](screenshots/gradcam_correct.png)

Example:

```text
Prediction: Malignant
Confidence: 97.11%
True Label: Malignant
```

---

## Misclassification Example

![Wrong Grad-CAM](screenshots/gradcam_wrong_prediction.png)

Example:

```text
Prediction: Malignant
Confidence: 76.51%
True Label: Benign
```

This demonstrates that the model can make incorrect predictions, even with relatively high confidence.

This is one reason model explainability and external validation are important in medical AI research.

---

# Deployment

The application is deployed using Streamlit Community Cloud.

```text
GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
MedVisionAI Web Application
        ↓
Image Upload
        ↓
Preprocessing
        ↓
ResNet18 Inference
        ↓
Prediction
        ↓
Confidence Score
        ↓
Optional Grad-CAM Visualization
```

The application provides:

* Image upload
* Benign/malignant classification
* Confidence score
* Optional Grad-CAM visualization
* Research and educational disclaimer

---

# Project Structure

```text
MedVisionAI/
│
├── models/
│   └── checkpoints/
│       ├── resnet18_breastmnist_final_best_f1.pth
│       └── resnet18_busi_final.pth
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── scripts/
│   ├── predict.py
│   ├── evaluate_busi.py
│   ├── evaluate_resnet.py
│   └── __init__.py
│
├── src/
│   │
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── resnet_dataloader.py
│   │   └── busi_dataset.py
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   └── resnet.py
│   │
│   ├── train/
│   │   ├── train.py
│   │   └── train_resnet.py
│   │
│   ├── explainability/
│   │   └── gradcam_utils.py
│   │
│   └── __init__.py
│
├── app.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Development Environment

| Component    | Details                         |
| ------------ | -------------------------------- |
| Python       | 3.12                            |
| PyTorch      | PyTorch                         |
| Torchvision  | Torchvision                     |
| Scikit-learn | Scikit-learn                    |
| Streamlit    | Streamlit                       |
| Hardware     | Apple Silicon                   |
| Acceleration | Apple Metal Performance Shaders |

Training was performed using Apple's Metal Performance Shaders acceleration:

```text
Using device: mps
```

This enabled GPU acceleration on Apple Silicon hardware.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/DunatosCharles/MedVisionAI.git

cd MedVisionAI
```

---

## Create a Virtual Environment

```bash
python -m venv clean_test
```

Activate it:

### macOS / Linux

```bash
source clean_test/bin/activate
```

---

## Install Dependencies

For the web app only:

```bash
pip install -r requirements.txt
```

For training, evaluation and notebooks:

```bash
pip install -r requirements-train.txt
```

---

# Training

To train the ResNet18 model:

```bash
python -m src.train.train_resnet
```

Options include `--epochs`, `--lr`, `--batch-size`, `--patience`, `--seed` and `--full-finetune`.

The best checkpoint (by validation F1) is evaluated on the test set and copied to `models/checkpoints/resnet18_busi_final.pth`, the checkpoint the web app serves.

To run lesion-grouped 5-fold cross-validation (trains five models, about 45 minutes on an Apple Silicon GPU):

```bash
python -m scripts.cross_validate
```

Results are written to `results/busi_cv/cv_metrics.json`. Fold checkpoints go to `models/checkpoints/cv/` and do not replace the web app's model.

The training process includes:

* Dataset loading
* Data augmentation
* Weighted loss
* Model training
* Learning-rate scheduling
* Validation
* Early stopping
* Checkpoint saving

---

# Evaluation

To evaluate the BUSI model:

```bash
python -m scripts.evaluate_busi
```

The evaluation script calculates:

```text
Accuracy, Precision, Recall, Specificity, F1 Score
ROC AUC and PR AUC
Brier Score and Expected Calibration Error
95% bootstrap confidence intervals for every metric
Confusion Matrix
```

It writes the following to `results/busi/`:

* `test_metrics.json`
* `roc_pr_curves.png`
* `confusion_matrix.png`
* `calibration.png` (reliability curves before and after temperature scaling)

It also fits the calibration temperature on the validation split and saves it to `models/checkpoints/resnet18_busi_metrics.json`, where the web app reads it. Retraining rewrites that file, so re-run this script after every training run.

Use `--checkpoint` to evaluate a different model, or `--bootstrap` to change the number of resamples.

---

# Inference

The trained model can be used to classify new ultrasound images.

```bash
python -m scripts.predict path/to/image.png
```

The inference pipeline:

```text
Input Image
     ↓
Grayscale Conversion
     ↓
Resize
     ↓
Normalization
     ↓
ResNet18
     ↓
Softmax Probabilities
     ↓
Prediction
     ↓
Confidence Score
```

---

# Training Curves

The project includes training visualizations such as:

### Training Loss

![Training Loss](screenshots/training_loss.png)

### Accuracy

![Accuracy Curve](screenshots/accuracy_curve.png)

### Validation F1 Score

![F1 Score](screenshots/f1_curve.png)

These visualizations help analyze:

* Convergence
* Overfitting
* Validation performance
* Learning-rate effects

---

# Limitations

Several limitations should be considered.

## Dataset Size

The BUSI dataset is relatively small compared with large-scale computer vision datasets.

This increases the risk of:

* Overfitting
* High variance between train/test splits
* Unstable performance estimates

---

## Dataset Splitting

A single ~90-image test set gives noisy estimates. Lesion-grouped 5-fold cross-validation (`python -m scripts.cross_validate`) trains five models so every usable image is tested exactly once, and reports mean ± standard deviation across folds plus bootstrap confidence intervals.

Near-duplicate grouping relies on image similarity, not patient identifiers, which BUSI does not provide. Different lesions from the same patient may still appear in different splits.

---

## Generalization

The model may not generalize reliably to:

* Different ultrasound machines
* Different hospitals
* Different patient populations
* Different image acquisition protocols
* Different image preprocessing pipelines

---

## Medical Use

The model is not clinically validated.

It should not be used to:

* Diagnose patients
* Replace radiologists
* Make treatment decisions
* Provide medical advice

---

# Future Work

## 1. Cross-Validation

Lesion-grouped 5-fold cross-validation is implemented in `scripts/cross_validate.py`:

```text
Dataset
   ↓
K-Fold Split
   ↓
Train Model Multiple Times
   ↓
Evaluate Each Fold
   ↓
Calculate Mean ± Standard Deviation
```

---

## 2. Patient-Level Data Splitting

Splits are now grouped by lesion using image similarity. If patient identifiers become available, grouping by patient would remove the remaining risk of leakage between different lesions of the same patient.

---

## 3. Larger Architectures

Future models could include:

* ResNet50
* EfficientNet
* ConvNeXt
* Vision Transformers

These models could be compared against ResNet18.

---

## 4. Hyperparameter Optimization

Future experiments could investigate:

* Learning rates
* Batch sizes
* Optimizers
* Weight decay
* Dropout rates
* Fine-tuning depth

---

## 5. Advanced Augmentation

Possible future techniques include:

* Contrast adjustment
* Gaussian noise
* Random cropping
* MixUp
* CutMix

However, augmentation strategies should be carefully evaluated for medical images to ensure that they do not create unrealistic samples.

---

## 6. Segmentation-Based Classification

The BUSI dataset includes segmentation masks.

A future experiment could investigate whether explicitly using lesion segmentation improves classification performance.

Possible pipeline:

```text
Ultrasound Image
        ↓
Lesion Segmentation
        ↓
Region of Interest Extraction
        ↓
Classification
        ↓
Benign / Malignant
```

---

## 7. External Validation

A major future improvement would be evaluating the model on an independent external dataset.

This would provide a better estimate of generalization to unseen data distributions.

---

## 8. Model Calibration

Calibration is now measured, and temperature scaling is fitted and applied (see the Calibration section above). For the current model, temperature scaling has almost no effect (T = 1.04). The remaining error is the over-prediction of malignant for mid-range scores. Methods that can bend the calibration curve, such as Platt scaling, isotonic regression or recalibration with class-prior correction, may help, but need more held-out data than a single split provides.

A model predicting:

```text
95% confidence
```

should ideally be correct approximately 95% of the time under similar conditions.

Calibration methods could include:

* Reliability diagrams
* Expected Calibration Error
* Temperature scaling

---

# Research Direction

The project is currently evolving from a benchmark-oriented experiment toward a more realistic medical imaging research workflow.

The main development progression is:

```text
BreastMNIST
    ↓
Initial CNN Experiments
    ↓
ResNet18 Transfer Learning
    ↓
Grayscale Architecture Adaptation
    ↓
BUSI Dataset
    ↓
Custom Dataset Pipeline
    ↓
Fine-Tuning Experiments
    ↓
Grad-CAM Explainability
    ↓
Web Deployment
    ↓
Future External Validation
```

The main objective is not simply to maximize a single test score.

Instead, the project aims to explore:

* How pretrained computer vision models can be adapted to medical imaging
* How dataset choice affects model performance
* How class imbalance affects training
* How model architecture and fine-tuning strategies influence generalization
* How explainability methods can help analyze model predictions
* How to build reproducible medical AI experiments

---

# Disclaimer

MedVisionAI is developed for educational and research purposes only.

It is not a medical diagnostic system.

The predictions generated by this model should not replace:

* Medical professionals
* Radiologists
* Clinical evaluation
* Professional diagnostic procedures

The model has not been clinically validated and should not be used for real-world medical decision-making.

---

# Author

## Dunatos Charles

Machine Learning and Computer Vision Developer

### Interests

* Artificial Intelligence
* Deep Learning
* Computer Vision
* Medical AI
* Machine Learning Research
* Explainable AI

---

# Project Summary

MedVisionAI demonstrates the development of a complete deep learning medical imaging pipeline.

The project evolved from an initial BreastMNIST experiment into a more advanced BUSI-based experiment involving:

```text
Dataset Exploration
        ↓
Data Preprocessing
        ↓
CNN Baseline
        ↓
Transfer Learning
        ↓
ResNet18 Adaptation
        ↓
Data Augmentation
        ↓
Class-Imbalanced Training
        ↓
Learning-Rate Scheduling
        ↓
Early Stopping
        ↓
Model Evaluation
        ↓
Grad-CAM Explainability
        ↓
Streamlit Deployment
```

The project serves as a foundation for future research into:

* Medical image classification
* Transfer learning
* Explainable AI
* Cross-dataset generalization
* External validation
* More robust medical computer vision systems

---

## Research Feedback

This project is actively being developed and improved.

Feedback regarding the methodology, dataset selection, model architecture, evaluation strategy, explainability, and potential future experiments would be highly valuable.

In particular, areas of interest include:

* Improving the reliability of evaluation on small medical datasets
* Designing better validation strategies
* Avoiding data leakage
* Improving generalization across datasets
* Comparing different transfer learning approaches
* Using segmentation information from the BUSI dataset
* Evaluating explainability methods such as Grad-CAM