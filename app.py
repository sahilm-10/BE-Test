from flask import Flask,request,jsonify
import tensorflow as tf
import os
from os.path import isfile, join
import numpy as np
import shutil
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio
import subprocess
import librosa
import glob
from random import randint
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Dropout , Flatten
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Lambda
from keras import backend as K
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd
# import os
# import colab

app = Flask(__name__)

def audio2vector(file_path, max_pad_len=400):
    
    #read the audio file
    audio, sr = librosa.load(file_path, mono=True)
    #reduce the shape
    audio = audio[::3]
    
    #extract the audio embeddings using MFCC
    mfcc = librosa.feature.mfcc(audio, sr=sr) 
    
    #as the audio embeddings length varies for different audio, we keep the maximum length as 400
    #pad them with zeros
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

@app.route("/")
def predict():
    audiofile_1 = "C:/Users/acer/Desktop/BE/similar-sounds/1.wav"
    convertedAudio1 = audio2vector(audiofile_1)
    audiofile_2 = "C:/Users/acer/Desktop/BE/similar-sounds/2.wav"
    convertedAudio2 = audio2vector(audiofile_2)
    reshapeAudio = np.reshape(convertedAudio1,(1,20,400))
    reshapeAudio2 = np.reshape(convertedAudio2,(1,20,400))
    
    # Load the Siamese TFLite model
    model = tf.lite.Interpreter(model_path="model.tflite")
    model.allocate_tensors()

    # Use the audio vectors as inputs to the model
    input_details = model.get_input_details()
    model.set_tensor(input_details[0]['index'], reshapeAudio)
    model.set_tensor(input_details[1]['index'], reshapeAudio2)

    # Run the model and get the prediction
    model.invoke()
    output_details = model.get_output_details() 
    prediction = model.get_tensor(output_details[0]['index'])

    # Return the prediction as the response
    # return prediction.tolist()
    # Convert the ndarray to a list and return it as a JSON object
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run()


