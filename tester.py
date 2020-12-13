from scipy.io import wavfile
from python_speech_features import mfcc
import os
import pickle
import warnings
import time

def predict(input_file):
    hmm_models = []
    # Read input file
    sampling_freq, audio = wavfile.read(input_file)

    # Extract MFCC features
    mfcc_features = mfcc(audio, sampling_freq)

    # Defining variables
    max_score = [float("-inf")]
    output_label = [float("-inf")]

    #Load the models
    for each_file in os.listdir('models/'):
        with open('models/'+each_file, 'rb') as file:
            hmm_models.append(pickle.load(file))
    
    #print(hmm_models)

    # Iterate through all the HMM models and pick the one with the highest score
    for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        if score > max_score:
            max_score = score
            output_label = label
    
    # Print the output
    print("Predicted:", output_label)
    warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Test the function and calculate time taken
    start_time = time.time()
    predict(input_file='test_files/Recording_13.wav')
    end_time = time.time()
    print("Time taken: ", end_time-start_time)