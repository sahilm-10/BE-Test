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
# import scipy.io.wavefile
import wave
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
from pydub import AudioSegment
# import os
# import colab

app = Flask(__name__)

def audio2vector(file_path, max_pad_len=400):
    
    #read the audio file
    audio, sr = librosa.load(file_path, mono=True,sr=None)
    #reduce the shape
    audio = audio[::3]
    
    #extract the audio embeddings using MFCC
    mfcc = librosa.feature.mfcc(audio, sr=sr) 
    
    #as the audio embeddings length varies for different audio, we keep the maximum length as 400
    #pad them with zeros
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def resample_librosa(input_file, output_file, sample_rate):
    # Load the audio file using librosa
    audio, _ = librosa.load(input_file,sr=None)
    # Resample the audio data
    audio_resampled = librosa.resample(audio, 44100, sample_rate)
    # Save the resampled audio to a file
    # scipy.io.wavefile.write(output_file, audio_resampled, sample_rate)
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_resampled)


@app.route("/")
def predict():
    # wav_audio = AudioSegment.from_file('C:/Users/acer/Desktop/BE/audio/luffy/audio.wav', format="wav")
    # # save the MP3 file
    # wav_audio.export('C:/Users/acer/Desktop/BE/similar-sounds/convertluffy.mp3', format="mp3")

    # wav_audio2 = AudioSegment.from_file('C:/Users/acer/Desktop/BE/similar-sounds/2.wav', format="wav")
    # # save the MP3 file
    # wav_audio2.export('C:/Users/acer/Desktop/BE/similar-sounds/convert2.mp3', format="mp3")
    # outputfile_1 = "None.wav"
    # outputfile_2 = "None1.wav"
    audiofile_1 = "C:/Users/acer/Desktop/BE/test/Audio/file11.mp3"
    # Load the mp3 file
    # song = AudioSegment.from_mp3(audiofile_1)
    # Save the file as a wav file
    # song.export(outputfile_1, format="wav")
    # resample_librosa(outputfile_1, 'resampled_1.wav', 22050)
    convertedAudio1 = audio2vector(audiofile_1)
    
    # Resample the audio files
    audiofile_2 = "C:/Users/acer/Desktop/BE/test/Audio/ryan-firebase.mp3"
     # Load the mp3 file
    # song = AudioSegment.from_mp3(audiofile_2)
    # Save the file as a wav file
    # song.export(outputfile_2, format="wav")
    # resample_librosa(outputfile_2, 'resampled_2.wav', 22050)
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


