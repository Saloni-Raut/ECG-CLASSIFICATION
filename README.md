# ECG-CLASSIFICATION
This project classifies ECG signals into six heart condition categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.
Here’s a breakdown of what the code is doing and how you can create the **README** content based on it:

---

# ECG Arrhythmia Classification using CNN

## Project Overview

This project builds a Convolutional Neural Network (CNN) to classify ECG signals into various heart conditions, such as arrhythmia, normal, and other related classes. The model uses image-based data generated from ECG signals and is trained using TensorFlow/Keras. The classification helps detect different types of arrhythmias based on ECG waveforms, aiding in early diagnosis of heart conditions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training the Model](#training-the-model)
5. [Performance Evaluation](#performance-evaluation)
6. [Confusion Matrix](#confusion-matrix)
7. [Saving the Model](#saving-the-model)
8. [Model Inference](#model-inference)
9. [Requirements](#requirements)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Results](#results)

## Dataset

The dataset used in this project comes from the **MIT-BIH ECG Arrhythmia Dataset**, which contains images of ECG signals classified into six categories:

1. Arrhythmia
2. Left Bundle Branch Block
3. Normal
4. Premature Ventricular Contraction
5. Right Bundle Branch Block
6. Ventricular Fibrillation

**Data directories:**

- Training data: `/content/drive/My Drive/MIT_BIH_ECGDatasets/train`
- Testing data: `/content/drive/My Drive/MIT_BIH_ECGDatasets/test`

The data is loaded using the `ImageDataGenerator` for image augmentation and rescaling.

## Model Architecture

We implement a Convolutional Neural Network (CNN) model with the following architecture:

- **Conv2D** layer with 32 filters of size (3x3), followed by ReLU activation and MaxPooling2D.
- Another **Conv2D** layer with 32 filters of size (3x3) and MaxPooling2D.
- **Flatten** layer to convert 2D features into 1D.
- **Dense** layer with 128 units and ReLU activation.
- Output **Dense** layer with 6 units and softmax activation for multi-class classification.

The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

## Training the Model

The model is trained on the preprocessed image data using the following parameters:

- **Epochs**: 12
- **Batch size**: 32
- **Image Size**: 64x64 pixels
- **Class Mode**: Categorical (for multi-class classification)

To train the model:
```python
model.fit(x_train, epochs=12, validation_data=x_test)
```

## Performance Evaluation

After training, the model is evaluated on the test data to calculate the accuracy and loss. The performance is also assessed using the **ROC Curve** and **AUC** (Area Under the Curve) for each class.

```python
test_loss, test_accuracy = model.evaluate(x_test)
```

To visualize ROC curves for all the classes:
```python
fpr, tpr, roc_auc = roc_curve()
```

## Confusion Matrix

A confusion matrix is used to evaluate the classification results. It compares the predicted class labels with the actual labels to give insights into the model’s performance across all categories.

```python
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
```

## Saving the Model

The trained model is saved as an HDF5 file to be used later for inference.

```python
model.save('ECG_IBM.h5')
```

## Model Inference

You can use the saved model to classify new ECG signals. Here's an example of loading the model and predicting the class of a new ECG image:

```python
model = load_model('/content/ECG_IBM.h5')
img = image.load_img('/path/to/ecg_image.png', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
```

The predicted class can be mapped to its corresponding label:
```python
index = ['Arrhythmia', 'Left Bundle Branch Block', 'Normal', 'Premature Ventricular Contraction', 'Right Bundle Branch Block', 'Ventricular Fibrillation']
result = index[np.argmax(pred)]
print(result)
```

## Requirements

The following dependencies are needed:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Pandas (optional for dataframes)
  
You can install the necessary packages using:
```bash
pip install tensorflow numpy scikit-learn seaborn matplotlib pandas
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecg-arrhythmia-classification.git
   cd ecg-arrhythmia-classification
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train and test the model, follow these steps:

1. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Run the training script:
   ```python
   python train.py
   ```

3. Test the model:
   ```python
   python evaluate.py
   ```

4. Make predictions with a new image:
   ```python
   python predict.py --image /path/to/image.png
   ```

## Results

- **Test Accuracy**: Achieved test accuracy of **XX%** after training the model for 12 epochs.
- **ROC Curves**: Visualized ROC curves with AUC for each class.
- **Confusion Matrix**: Displayed the confusion matrix for better understanding of the classification performance across all classes.

## References
- TensorFlow and Keras documentation

---

This README content covers all key parts of the code, including dataset, model training, evaluation, and usage instructions for inference.
