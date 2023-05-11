import os
import sys
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

import logging

# Define evaluation metrics
def metric_FPR(y_true,y_pred):    
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    FPR=FP/(FP+TN+K.epsilon())  # Calculate False Positive Rate
    return FPR

def metric_FNR(y_true,y_pred):  
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    FNR=FN/(FN+TP+K.epsilon())  # Calculate False Negative Rate
    return FNR

def metric_F1score(y_true,y_pred):    
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP+K.epsilon())  # Calculate Precision
    recall=TP/(TP+FN+K.epsilon())     # Calculate Recall
    F1score=2*precision*recall/(precision+recall+K.epsilon())  # Calculate F1 Score
    return F1score

# Set the logging level to ERROR
logging.basicConfig(level=logging.ERROR)
# Disable TensorFlow logging output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Get command line arguments and convert them into a floating point vector
args = sys.argv[1:]
input_data = np.array([float(arg) for arg in args])
# Convert the vector to a numpy array
input_data_one = np.array(input_data)
# Convert the numpy array to a tensor
input_data_one = np.reshape(input_data_one, (1,len(input_data_one)))
# Load the pre-trained DNN model
trust_model = load_model('./model/premodel1025.h5', compile=True,custom_objects={'metric_F1score': metric_F1score,'metric_FPR':metric_FPR,'metric_FNR':metric_FNR})

# Make predictions on the input data            
trust_pred = trust_model.predict(input_data_one)
trust_pred=np.round(trust_pred,3)
# Get the predicted value from the matrix
trust_value = trust_pred[0][0]
# Print the predicted result and return it to Matlab
print(trust_value)
