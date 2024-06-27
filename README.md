# Brain_Tumor_Detection_CNN
A project for early detection and classification of brain tumors using Convolutional Neural Networks (CNNs) and MRI images.

## Overview

The early detection and classification of brain tumors are crucial in the field of medical imaging as it can help in selecting the most appropriate treatment method to save patients' lives. Brain tumors can be cancerous or noncancerous, and they can cause pressure inside the skull to increase, which can lead to brain damage and be life-threatening. The use of Magnetic Resonance Imaging (MRI) has become an important tool in the diagnosis of brain tumors, allowing for the detection, location identification, and classification of tumors based on malignancy, grade, and type. In recent years, deep learning approaches such as Convolutional Neural Networks (CNNs) have been applied to improve the accuracy and efficiency of brain tumor detection and classification. This experimental work in the diagnosis of brain tumors using a multi-task CNN-based approach has the potential to improve the early detection and classification of brain tumors, which can lead to more successful treatment outcomes and ultimately save lives.

## Dataset

The dataset is sourced from Kaggle and includes MRI images categorized into four classes:
- Pituitary
- Meningioma
- No Tumor
- Glioma

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place it in the `Data/Training` directory, organized into subfolders by category.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DataDreamWeaver/BrainTumorDetectionCNN.git
   cd BrainTumorDetectionCNN
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset:
   - Download the dataset from Kaggle and place your MRI images in the `Data/Training` directory, organized into subfolders by category.

2. Run the preprocessing and training script:
   ```bash
   python train.py
   ```

## Evaluation

The script computes and displays the precision, recall, and F1 scores for each class and their averages.

## Models

The project includes multiple CNN models for comparison:
1. **AlexNet1**
2. **AlexNet2**
3. **ResNet1**
4. **ResNet2**
5. **Pretrained ResNet**

## Results

The project evaluates models using precision, recall, and F1 scores. Confusion matrices and ROC curves are also provided for further analysis.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
