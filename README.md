# Brain Tumor Classification using Deep Learning 🧠

A convolutional neural network that classifies brain MRI scans using transfer learning with ResNet18.

## Overview
This project fine-tunes a pretrained ResNet18 model to classify brain tumor MRI images. It includes data augmentation for better generalization and full evaluation via classification report and confusion matrix.

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch, Torchvision
- **Model:** ResNet18 (transfer learning), custom classifier head
- **Evaluation:** scikit-learn (classification report, confusion matrix)
- **Environment:** Google Colab (GPU - Tesla T4)

## Approach
- Loaded and preprocessed MRI image dataset using `ImageFolder`
- Applied data augmentation: random horizontal flip, random rotation, color jitter
- Fine-tuned ResNet18 with a custom head: `Linear → ReLU → Dropout(0.4) → Linear`
- Trained for 25 epochs using Adam optimizer and CrossEntropyLoss
- Evaluated using confusion matrix and per-class precision/recall/F1

## Results
- Training accuracy improved from **56% → 95.4%** over 25 epochs
- Final training loss: **0.17**
- Generated confusion matrix and classification report for detailed evaluation

## How to Run
1. Open the notebook in Google Colab
2. Mount Google Drive and set the dataset path
3. Run all cells — trains the model and saves it as `brain_tumor_model.pth`

## Future Improvements
- Add a validation/test split (currently trained on full dataset)
- Try other architectures (ResNet50, EfficientNet) for comparison
- Deploy as a simple web app for interactive predictions
