import speech_recognizer as sr
import os
import argparse
import warnings
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
import pickle

args = sr.build_arg_parser().parse_args()
input_folder = args.input_folder

hmm_models = []

# Parse the input directory
for dirname in os.listdir(input_folder):
    # Get the name of the subfolder 
    subfolder = os.path.join(input_folder, dirname)

    if not os.path.isdir(subfolder):
        continue

    # Extract the label 
    label = subfolder[subfolder.rfind('/') + 1:]

    # Initialize variables 
    X = np.array([])
    y_words = []
    warnings.filterwarnings('ignore')

    # Iterate through the audio files (leaving one file for testing in each class)
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
        # Read the input file 
        filepath = os.path.join(subfolder, filename)
        sampling_freq, audio = wavfile.read(filepath)

        # Extract the MFCC features
        mfcc_features = mfcc(audio, sampling_freq)

        # Append to the variable X
        if len(X) == 0:
            X = mfcc_features
        else:
            X = np.append(X, mfcc_features, axis=0)

        # Append the label
        y_words.append(label)

    # Train and save the HMM model
    hmm_trainer = sr.HMMTrainer()
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, label))
    hmm_trainer = None

print(len(hmm_models))
for model in hmm_models:
    folder = 'models/'
    model_name = model[1]
    with open(folder+model_name, 'wb') as file:
        pickle.dump(model, file)

