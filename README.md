# Facial Verification using a Siamese Neural Network

Hello! This is the facial verification/recognition repository you came across. I used one-shot learning using a siamese neural network to prevent training a model, which can be time consuming and consume resources. 

## Model Architecture

The neural network itself is a pretty simple one. Here's the architecture:

- ConvLayer - 64 3x3 filters with strides of 2
- MaxPool2D layer - 2x2 "filter"
- Dropout with 20% Probability
- ConvLayer - 128 2x2 filters (no padding/strides)
- MaxPool2D layer - 2x2 "filter"
- Flatten Layer
- Fully Connected Layer with 128 logistic regression units (layer that outputs encodings) 

