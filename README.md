# Voice-Prescription

The following code uses Hidden Markov Model for speech recognition.
## Dependencies

1. python=3.9
2. numpy==1.18.2
3. scikit-learn==0.22.2
4. hmmlearn==0.2.4
5. python-speech-features==0.6

## Description

Install all the dependencies and cd into the directory where this repo is and then run  

`python speech_recognizer.py --input-folder data/`

The data folder contains audio datasets that are to be trained where the last file of each folder in data is used to test the accuracy of the model.