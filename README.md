# Brain_Tumor_Detection_CNN
A project for early detection and classification of brain tumors using Convolutional Neural Networks (CNNs) and MRI images.


## 1. Overview

The early detection and classification of brain tumors is crucial in medical imaging, aiding in the selection of appropriate treatment methods to save patients' lives. Brain tumors, whether cancerous or noncancerous, can cause increased pressure inside the skull, potentially leading to brain damage and life-threatening conditions. Magnetic Resonance Imaging (MRI) plays a pivotal role in diagnosing brain tumors, facilitating their detection, localization, and classification based on malignancy, grade, and type. Recent advancements in deep learning, specifically Convolutional Neural Networks (CNNs), have significantly improved the accuracy and efficiency of brain tumor detection and classification. This experimental work leverages a multi-task CNN-based approach to enhance early detection and classification, aiming to improve treatment outcomes and save lives.

## 2. Dataset

The dataset used in this project is sourced from MRI images categorized into four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary

The dataset is organized into separate folders for each category. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place it in the Data/Training directory, organized into subfolders by category.

## 3. Installation

1. **Clone the repository**:
   ```bash
   [https://github.com/NikkieHooman/Brain_Tumor_Detection_CNN.git]
   cd BrainTumorDetectionCNN
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 4. Data Preprocessing

The following preprocessing steps were applied to each image:
- **Cropping**: Cropped the part of the image containing the brain.
- **Resizing**: Resized all images to 50x50 pixels.
- **Normalization**: Scaled pixel values to the range 0-1.

## 5. Data Augmentation

Given the small dataset size, data augmentation was applied to increase the number of examples and address class imbalance. Techniques such as rotation, width and height shift, shear, and horizontal/vertical flip were used, resulting in an augmented dataset with 2065 images.

## 6. Neural Network Architectures

### AlexNet1
- 32, 64, and 128 filters in three convolution layers
- 3x3 kernel size
- 2x2 max pooling layers
- 128 neurons in the dense layers
- ReLU activation functions

### AlexNet2
- 64, 128, and 256 filters in three convolution layers
- 3x3 kernel size
- 3x3 max pooling layers
- 256 neurons in the dense layers
- ReLU activation functions

### ResNet1
- 32 filters in the initial convolution layers
- 64, 64, and 32 filters in the bypassed convolution layers
- 3x3 and 1x1 kernel sizes
- 2x2 max pooling layers
- 256 neurons in the dense layer

### ResNet2
- 64 filters in all convolution layers
- 3x3 kernel size
- 3x3 max pooling layers
- 256 neurons in the dense layer

### Pretrained ResNet
- Used ResNet50 pretrained on ImageNet
- Fine-tuned on the brain tumor dataset

## 7. Model Training and Evaluation

The models were trained using 10-fold cross-validation. The performance was evaluated based on precision, recall, F1-score, and confusion matrices.

### Evaluation Metrics
- **Precision**: Proportion of true positive predictions among all positive predictions.
- **Recall**: Proportion of true positive predictions among all actual positive cases.
- **F1-score**: Harmonic mean of precision and recall, providing a balanced evaluation metric.

## 8. Results

The results indicate that the ResNet2 model outperforms other models, achieving the highest precision and lowest validation loss during testing.

### Confusion Matrices
Confusion matrices were plotted to visualize the performance of each model in distinguishing between different classes.

### ROC Curves
ROC curves were plotted for all models to compare their performance in terms of true positive and false positive rates.

## 9. Exceptional Work

In addition to the basic models, a pretrained ResNet50 model was used and fine-tuned on the dataset, showing competitive performance compared to the custom architectures.

## 10. Conclusion

The project demonstrates the effectiveness of CNNs, particularly ResNet2, in accurately detecting and classifying brain tumors from MRI images. The use of data augmentation and transfer learning further enhances model performance.

## 11. Future Work

Future improvements could include:
- Increasing the dataset size for better generalization
- Exploring advanced augmentation techniques
- Implementing other state-of-the-art architectures

