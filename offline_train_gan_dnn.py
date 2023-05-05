import keras
import tensorflow as tf
from keras import layers
import numpy as np
import keras.backend as K
import pandas as pd 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Defining metrics FPR, FNR, and F1-score:
def metric_FPR(y_true,y_pred):    
     # False Positive Rate
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    #precision=TP/(TP+FP+K.epsilon())
    FPR=FP/(FP+TN+K.epsilon())
    return FPR

def metric_FNR(y_true,y_pred): 
    # False Negative Rate 
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    # recall=TP/(TP+FN+K.epsilon())
    FNR=FN/(FN+TP+K.epsilon())
    return FNR

def metric_F1score(y_true,y_pred):  
    # F1 score  
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP+K.epsilon())
    recall=TP/(TP+FN+K.epsilon())
    F1score=2*precision*recall/(precision+recall+K.epsilon())
    return F1score

#Defining generator model:
G_input = keras.Input(shape=(4,)) 
x = layers.Dense(15,activation='relu')(G_input)
x = layers.Dense(15,activation='relu')(x)
x = layers.Dense(4,activation='tanh')(x) 
G = keras.models.Model(G_input,x)
G.summary()
#Defining discriminator model:
D_input = keras.Input(shape=(4,))
x = layers.Dense(15,activation='relu')(D_input)
x = layers.Dense(15,activation='relu')(x)
x = layers.Dropout(0.2, input_shape=(4,))(x)
x = layers.Dense(1,activation='sigmoid')(x)
D = keras.models.Model(D_input,x)
D.compile(loss='binary_crossentropy',optimizer='rmsprop')
D.summary()
#Setting discriminator to be non-trainable and creating the generator model:
D.trainable = False 
gan_input = keras.Input(shape=(4,))
gan_output = D(G(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy',optimizer='rmsprop')
gan.summary()
#Loading and normalizing the data:
td = pd.read_csv('real_dataset_1025.csv',usecols=[1, 2, 3,4])
real_point = (td-np.nanmin(td))/(np.nanmax(td)-np.nanmin(td)) 
num_rows, num_cols = real_point.shape
#Training the generator and discriminator:
epochs = 1000
d_loss=[0]*epochs 
a_loss=[0]*epochs 
inputnum=num_rows 
for step in range(epochs):
    random_input = np.random.normal(size=(inputnum,4)) 
    gen_point = G.predict(random_input) 
    combined_point = np.concatenate([real_point,gen_point])
    labels = np.concatenate([np.zeros((inputnum,1)),np.ones((inputnum,1))])

    labels += 0.05*np.random.random(labels.shape)  
    d_loss[step] = D.train_on_batch(combined_point, labels) 
    random_input = np.random.normal(size=(inputnum, 4))
    mis_targets = np.zeros((inputnum,1))
    a_loss[step] = gan.train_on_batch(random_input, mis_targets)
    
    # visualize training process
    if step%20==0:
        print('discriminator loss:', d_loss[step])
        print('adversarial loss:', a_loss[step])
        # try:
        #     ax.lines.remove(points[0])
        # except Exception:
        #     pass
        # gen_point = G.predict(random_input)
        # points = ax.plot(gen_point[:, 0], gen_point[:, 1],gen_point[:, 2],'ro')
        # plt.pause(0.01)
        # if step==20:
        #     plt.pause(10)
        
# generate 10000 data for DNN training
random_input = np.random.normal(size=(10000, 4)) 

# load 1000 real data for DNN training
td = pd.read_csv('realtrainingdata_1025.csv',usecols=[1, 2, 3,4])
realdata1000 = (td-np.nanmin(td))/(np.nanmax(td)-np.nanmin(td))
PreInput=G.predict(random_input)
traindata = np.concatenate([PreInput,realdata1000])
trainlabel = np.concatenate([np.zeros((10000,1)),np.zeros((500,1)),np.ones((500,1))])

# DNN model 
PreTrust_input = keras.Input(shape=(4,))
x = layers.Dense(15,activation='relu')(PreTrust_input)
x = layers.Dense(15,activation='relu')(x)
x = layers.Dropout(0.5, input_shape=(4,))(x)
x = layers.Dense(1,activation='sigmoid')(x)
PreTrust = keras.models.Model(PreTrust_input,x)
PreTrust.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy',metric_FPR,metric_FNR,metric_F1score])
#PreTrust.compile(loss='mean_squared_error',optimizer='rmsprop',metrics='accuracy')
PreTrust.summary()

history=PreTrust.fit(traindata,trainlabel,epochs=100)
print(history)
# load 1000 real data for DNN testing
testdata = pd.read_csv('testdata_1025.csv',usecols=[1, 2, 3,4])
tdata = (testdata-np.nanmin(testdata))/(np.nanmax(testdata)-np.nanmin(testdata))
testlabel = np.concatenate([np.zeros((500,1)),np.ones((500,1))])
# print results
results=PreTrust.evaluate(tdata,testlabel)
print(results)
# results=PreTrust.predict(tdata,verbose=0)
# results=np.round(results,3)
# print(results)
