# Medical Appointment No Shows Prediction

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)
10. [Contact Information](#contact-information)

## Introduction

This project aims to predict whether patients will show up for their medical appointments using the Medical Appointment No Shows dataset. The dataset includes various features such as patient demographics, appointment details, and whether the patient received an SMS reminder. The goal is to implement and optimize a multi-layer perceptron (MLP) model using PyTorch to achieve accurate predictions.

## Dataset Description

The dataset contains the following features:

- **PatientId**: Unique identifier for each patient
- **AppointmentID**: Unique identifier for each appointment
- **Gender**: Gender of the patient (M/F)
- **ScheduledDay**: The day the patient set up their appointment
- **AppointmentDay**: The day of the actual appointment
- **Age**: Age of the patient
- **Neighbourhood**: The location of the hospital
- **Scholarship**: Whether the patient is enrolled in the Brasilian welfare program Bolsa Família (0 or 1)
- **Hipertension**: Whether the patient has hypertension (0 or 1)
- **Diabetes**: Whether the patient has diabetes (0 or 1)
- **Alcoholism**: Whether the patient is an alcoholic (0 or 1)
- **SMS_received**: Whether the patient received an SMS reminder (0 or 1)
- **No-show**: Whether the patient showed up for their appointment (No/Yes)

## Project Structure

The project is structured as follows:

1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature engineering.
2. **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships between features.
3. **Model Implementation**: Building and training an MLP model using PyTorch.
4. **Hyperparameter Tuning**: Experimenting with different activation functions, optimizers, and regularization techniques.
5. **Model Evaluation**: Evaluating model performance using metrics such as accuracy, precision, recall, and F1-score.
6. **Cross-Validation**: Performing cross-validation to ensure robust model evaluation.
7. **Visualization**: Visualizing training progress and feature importance.

## Installation

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch
- imbalanced-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch imbalanced-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/medical-appointment-no-shows.git
```

2. Navigate to the project directory:

```bash
cd medical-appointment-no-shows
```

3. Run the Jupyter notebook or Python script to preprocess the data, train the model, and evaluate its performance.

## Results

### Implementation of MLP Model using PyTorch

The implementation of the Multi-Layer Perceptron (MLP) model was done using PyTorch, a popular open-source machine learning library. The model was designed with an input layer, multiple hidden layers, and an output layer.

#### Model Architecture

The model architecture is as follows:

- **Input Size**: The input size is equal to the number of features in the dataset, which is determined by the number of inputs.
- **Hidden Size**: The hidden size, which is the number of neurons in the hidden layer, was set to 128. This can be adjusted as needed.
- **Output Size**: The output size is 1, corresponding to our binary classification problem.
- **Number of Hidden Layers**: The model was designed with 2 hidden layers.

#### Loss Function and Optimizer

The MSE loss function was used as the criterion for the model. The Adam optimizer was used for optimization, with a learning rate of 0.001 and a weight decay of 0.0001.

#### Training the Model

The model was trained for 30 epochs. In each epoch, the model was set to training mode, and the gradients were zeroed at the start of each mini-batch. The forward pass was performed to get the outputs of the model, and the loss was calculated by comparing these outputs with the targets. The backward pass and optimization step were then performed. The running loss was updated after each mini-batch. At the end of each epoch, the average loss and accuracy for the epoch were calculated and printed.

### A Review of Model Architectures and Techniques

In the process of implementing the MLP model for this project, various techniques were investigated to enhance the model’s performance. These techniques included advanced activation functions, initialization methods, and regularization techniques.

#### Activation Functions

The activation function plays a crucial role in a neural network by determining whether a neuron should be activated or not. The initial model was implemented with the default activation function (ReLU). To investigate the impact of different activation functions on the model’s performance, the Tanh activation function was also tested. The results were as follows:

- **ReLU**: Mean Squared Error (MSE): 0.2008, Mean Absolute Error (MAE): 0.4053, Accuracy: 0.7220
- **Tanh**: Mean Squared Error (MSE): 0.2008, Mean Absolute Error (MAE): 0.4052, Accuracy: 0.7220

The results show that the choice of activation function did not significantly impact the model’s performance in this case.

#### Optimization Algorithms

The optimization algorithm is responsible for updating the weights of the neural network to minimize the loss function. The initial model used the Adam optimizer. To investigate the impact of different optimization algorithms on the model’s performance, the Stochastic Gradient Descent (SGD) optimizer was also tested. The results were as follows:

- **Adam**: Mean Squared Error (MSE): 0.2008, Mean Absolute Error (MAE): 0.4053, Accuracy: 0.7220
- **SGD**: Mean Squared Error (MSE): 0.2006, Mean Absolute Error (MAE): 0.4025, Accuracy: 0.7220

The results show that the choice of optimization algorithm did not significantly impact the model’s performance in this case. All tests were conducted with a batch size of 64 and for 30 epochs.

### Regularization Techniques and Hyperparameter Tuning

To prevent overfitting and improve the performance of our MLP model, we implemented dropout and L2 regularization techniques and performed hyperparameter tuning.

#### Dropout and L2 Regularization

Dropout is a regularization technique that prevents overfitting by randomly dropping out neurons during training. In our model, we implemented dropout in the hidden layers with a dropout rate of 0.2 for the input layer and 0.5 for the hidden layers. L2 regularization, also known as weight decay, was implemented through the Adam optimizer with a weight decay of 0.0001. This adds a penalty term to the loss function, encouraging the model to learn smaller weights and resulting in a simpler model.

#### Hyperparameter Tuning

We performed hyperparameter tuning to find the optimal values for the learning rate, batch size, and activation function. The hyperparameters were tuned by training the model with different combinations of these parameters and comparing the resulting model performance. The hyperparameters were tuned as follows:

- **Learning Rate**: We tested learning rates of 0.001, 0.01, and 0.1.
- **Batch Size**: We tested batch sizes of 32, 64, and 128.
- **Activation Function**: We tested the ReLU, LeakyReLU, and ELU activation functions.

For each combination of hyperparameters, we trained the model and recorded the Mean Squared Error (MSE), Mean Absolute Error (MAE), and Accuracy. The results were then plotted to visually compare the performance of each model.

### Model Training and Evaluation

The MLP model was trained on the training data and validated using the validation set. The performance of the model was evaluated using appropriate metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Accuracy.

#### Model Training

The model was trained using the training data. The training process involved feeding the data to the model, calculating the loss using the MSE loss function, and updating the model parameters using the Adam optimizer. This process was repeated for a specified number of epochs.

#### Model Evaluation

The performance of the model was evaluated on the validation set. The evaluation metrics used were:

- **Mean Squared Error (MSE)**: This is the average of the squared differences between the predicted and actual values. It is a popular metric for regression problems. The MSE for our model was 0.2008.
- **Mean Absolute Error (MAE)**: This is the average of the absolute differences between the predicted and actual values. It gives an idea of how wrong the predictions were. The MAE for our model was 0.4053.
- **Accuracy**: This is the proportion of correct predictions over total predictions. The accuracy of our model was 0.7220.

#### Cross-Validation

To ensure a robust evaluation of the model, cross-validation was performed. In cross-validation, the data is split into ‘k’ subsets, and the model is trained ‘k’ times, each time using a different subset as the validation set and the remaining data as the training set. The cross-validation results were consistent with the evaluation results, confirming the reliability of the model.

**Cross-Validation Results**:

- **Mean Squared Error (MSE)**: 0.2007
- **Mean Absolute Error (MAE)**: 0.4013
- **Accuracy**: 0.7220

### Hyperparameter Tuning Results

The hyperparameter tuning process involved experimenting with different learning rates, batch sizes, and activation functions. The results of the hyperparameter tuning are summarized below:

- **Learning Rate**: Tested values were 0.001, 0.01, and 0.1.
- **Batch Size**: Tested values were 32, 64, and 128.
- **Activation Functions**: Tested functions were ReLU, LeakyReLU, and ELU.

The best performing model was achieved with a learning rate of 0.001, a batch size of 64, and the ReLU activation function.

### Final Model Evaluation

The final model was evaluated on the test set, and the following metrics were obtained:

- **Mean Squared Error (MSE)**: 0.2008
- **Mean Absolute Error (MAE)**: 0.4053
- **Accuracy**: 0.7220

## Conclusion

This project demonstrates the application of MLP models using PyTorch to predict medical appointment no-shows. Through data preprocessing, model training, hyperparameter tuning, and evaluation, we achieved promising results. The choice of activation function and optimization algorithm did not significantly impact the model’s performance in this case. Future work could explore more advanced architectures and additional features to further improve prediction accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments).
- Special thanks to the contributors of the libraries and tools used in this project.

## Contact Information

For any questions or inquiries, please contact:

- **Email**: pouya.8226@gmail.come
