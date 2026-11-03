# Deep Learning Lab Programs

**Course:** Deep Learning Laboratory – 7th Semester B.E. (CSE-AIML), 2022 Scheme  
**University:** Visvesvaraya Technological University (VTU), Belagavi  
**College:** AMC Engineering College, Bangalore

This repository contains all 8 programs of the Deep Learning lab, using Python, TensorFlow/Keras and scikit-learn.

## Contents

| No. | Program | Dataset | File |
|-----|---------|---------|------|
| Program 1 | (Word Embeddings) | Sample corpus | [Program1_Word_Embeddings.py](Program1_Word_Embeddings.py) |
| Program 2 | (Deep Neural Network) | Iris | [Program2_Deep_Neural_Network.py](Program2_Deep_Neural_Network.py) |
| Program 3 | (CNN Image Classification) | MNIST | [Program3_CNN_Image_Classification.py](Program3_CNN_Image_Classification.py) |
| Program 4 | (Autoencoder) | MNIST | [Program4_Autoencoder.py](Program4_Autoencoder.py) |
| Program 5 | (Text Classification) | IMDB reviews | [Program5_Text_Classification.py](Program5_Text_Classification.py) |
| Program 6 | (Time Series Forecasting) | Synthetic sine wave | [Program6_Time_Series_Forecasting.py](Program6_Time_Series_Forecasting.py) |
| Program 7 | (Pretrained Model / Transfer Learning) | CIFAR-10 | [Program7_Pretrained_Model_Transfer_Learning.py](Program7_Pretrained_Model_Transfer_Learning.py) |
| Program 8 | (Sentiment Analysis) | Sample text reviews | [Program8_Sentiment_Analysis.py](Program8_Sentiment_Analysis.py) |

## Program Details and Expected Output

### Program 1 (Word Embeddings)
Trains a Word2Vec model (gensim) on a small corpus, with 50-dimensional vectors and a context window of 2.  
**Output:** the 50-value embedding vector for `learning`, followed by the list of words most similar to it with their cosine similarity scores. Exact numbers differ on every run because training is random.

### Program 2 (Deep Neural Network)
A feed-forward network (16-12-8-3 neurons, ReLU and Softmax) classifies the 3 Iris flower species.  
**Output:** model summary, training logs for 50 epochs, test accuracy (about 100%), and predicted classes compared with actual classes.

### Program 3 (CNN Image Classification)
A CNN with 2 convolution and pooling blocks, a dense layer and dropout classifies MNIST digits.  
**Output:** training logs for 10 epochs, test accuracy (about 99%), and predicted vs actual classes for the first 10 test images (`[7 2 1 0 4 1 4 9 5 9]`).

### Program 4 (Autoencoder)
A single-layer autoencoder compresses each 784-pixel MNIST image into 32 values and reconstructs it.  
**Output:** training loss for 5 epochs (falls from about 0.27 to 0.12), a plot of original vs reconstructed digits, and the sizes `Original size: (10000, 784)` and `Compressed size: (10000, 32)`.

### Program 5 (Text Classification)
An Embedding + GlobalAveragePooling network classifies IMDB movie reviews as positive or negative.  
**Output:** training and validation logs for 5 epochs, test accuracy (about 84–88%), and a Positive/Negative label with its probability for the first 5 test reviews.

### Program 6 (Time Series Forecasting)
An LSTM learns a noisy sine wave using a sliding window of 20 time steps and forecasts the held-out 20% of the data.  
**Output:** 3 plots (the synthetic series, training vs validation loss, and actual vs predicted values), with the loss falling to about 0.002.

### Program 7 (Pretrained Model / Transfer Learning)
A frozen MobileNetV2 pre-trained on ImageNet is used as a feature extractor, with a new classifier head for the 10 CIFAR-10 classes.  
**Output:** training logs for 5 epochs, training, validation and test accuracy (about 50%), the actual vs predicted class of one test image, and an accuracy graph. This is slow on CPU, so use a GPU runtime in Google Colab.

### Program 8 (Sentiment Analysis)
TF-IDF features and Logistic Regression classify text reviews as positive or negative.  
**Output:**
```
Test Accuracy: 1.0

Review: The movie was amazing and I loved it
Sentiment: Positive

Review: The product was awful and disappointing
Sentiment: Negative

Review: I am very happy with this purchase
Sentiment: Positive
```
A classification report with precision, recall and F1 is also printed.

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/DL-Lab-Programs.git
cd DL-Lab-Programs

# 2. Install the requirements
pip install -r requirements.txt

# 3. Run any program
python Program1_Word_Embeddings.py
```

The programs also run on **Google Colab** or **Jupyter Notebook**. Programs 3, 4, 5 and 7 download their datasets automatically on the first run, so an internet connection is needed. Plots open in a window; close each one to let the program continue.

## Tools and Libraries
Python, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras, Gensim

## Author Details

- **Author Name:** POTAKARI SMART
- **USN:** 1AM23CI107
- **Branch:** CSE - AIML
- **College:** AMC Engineering College, Bengaluru
- **Subject:** Deep Learning Laboratory
