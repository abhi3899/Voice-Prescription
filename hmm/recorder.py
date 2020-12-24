from scipy.io import wavfile
import sounddevice as sd

def record(location, i):
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
    file_path = location + 'recording' + str(i) + '.wav'
    wavfile.write(file_path, sampling_freq, audio) 

for i in range(10):
    print("Speek Now...  ", i)
    record(location='data/microorganism/', i=i)
