from scipy.io import wavfile
from python_speech_features import mfcc
import os
import pickle
import warnings
import time
import sounddevice as sd

def get_arr_from_file(input_file):
    # Read input file
    sampling_freq, audio = wavfile.read(input_file)
    return sampling_freq, audio

def get_arr_from_microphone():
    print("Speak Now... ")
    # Sampling frequency
    sampling_freq = 44100

    # Recording duration
    duration = 5

    # Start recorder with the given values of 
    # duration and sample frequency
    audio = sd.rec(int(duration * sampling_freq), 
                    samplerate=sampling_freq, channels=2)
    sd.wait()
    # This will convert the NumPy array to an audio 
    # file with the given sampling frequency 
    return sampling_freq, audio

def predict(sampling_freq, audio):

    hmm_models = []
    # Extract MFCC features
    mfcc_features = mfcc(audio, sampling_freq, nfft=1103)

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
    #sampling_freq, audio = get_arr_from_file(input_file='test_files/Recording_13.wav')
    sampling_freq, audio = get_arr_from_microphone()
    # Test the function and calculate time taken
    start_time = time.time()
    # Perform prediction
    predict(sampling_freq, audio)
    end_time = time.time()
    print("Time taken: ", end_time-start_time)