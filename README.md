# Architectural Heritage Elements Image Classification

## Table of Contents

- [Introduction](#introduction)
- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Building](#model-building)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Optimization](#optimization)
- [Expected Outcome](#expected-outcome)
- [Conclusion](#conclusion)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact Information](#contact-information)

## Introduction

The Architectural Heritage Elements Dataset (AHE) is an image dataset for developing deep learning algorithms and specific techniques in the classification of architectural heritage images. This dataset consists of 10,235 images classified into ten categories.

## Objective

The objective of this project is multi-fold:

1. **CNN Model Training**: Build and train a Convolutional Neural Network (CNN) to classify images from the Architectural Heritage Elements Image64 Dataset.
2. **Deconvolution Visualization**: Use deconvolution techniques to visualize what the trained CNN model sees when it processes an input image.
3. **Image Generation for Specific Classes**: Manipulate the trained model to generate an image corresponding to a specific class.

## Dataset

The dataset used in this project is the Architectural Heritage Elements Image64 Dataset. It consists of 10,235 images classified into ten categories:

- Altar: 829 images
- Apse: 514 images
- Bell tower: 1059 images
- Column: 1919 images
- Dome (inner): 616 images
- Dome (outer): 1177 images
- Flying buttress: 407 images
- Gargoyle (and Chimera): 1571 images
- Stained glass: 1033 images
- Vault: 1110 images

## Methodology

### Data Preprocessing

The images are preprocessed for the model, including resizing the images and normalizing the pixel values.

### Model Building

A CNN model is built using PyTorch. The model includes several convolutional and pooling layers, followed by dense layers.

### Training

The model is trained on a portion of the dataset. Various strategies like data augmentation and dropout are used to improve the model’s performance and prevent overfitting.

### Evaluation

The model’s performance is evaluated on a separate test set. Metrics such as accuracy, precision, recall, and F1-score are used for this purpose.

### Optimization

Based on the evaluation results, the model is further optimized by tuning hyperparameters and modifying the architecture.

## Expected Outcome

At the end of this project, we expect to have a working CNN model that can classify images of architectural elements with reasonable accuracy. Additionally, we aim to gain practical experience in handling image data, building and training CNN models, and optimizing these models for better performance.

## Conclusion

This project demonstrates the application of deep learning techniques to classify architectural heritage elements. By visualizing the feature maps and generating images for specific classes, we gain insights into how the model interprets the input images and which features it finds crucial for classification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The creators of the Architectural Heritage Elements Image64 Dataset.
- The PyTorch community for providing excellent resources and support.

## Contact Information

For any questions or inquiries, please contact:

- Email: pouya.8226@gmail.come
