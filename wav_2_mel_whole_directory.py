import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the WAV file
wav_directory = './music_data_copy'
Mel_directory = './Mel_Spectrograms'
for file_name in os.listdir(wav_directory):
    file_name_tokens = file_name.split('.')
    if file_name_tokens[1] != "wav": # ensuring that file format is wav
        continue
    file_path = os.path.join(wav_directory,file_name)
    y, sr = librosa.load(file_path, sr=None)  # sr=None preserves the original sampling rate

    # Generate the Mel spectrogram
    n_fft = 2048  # Length of the FFT window
    hop_length = 512  # Number of samples between successive frames
    n_mels = 128  # Number of Mel bands
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to decibel (log scale) for better visualization
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # print(file_name_tokens)
    mel_path = os.path.join(Mel_directory,file_name_tokens[0])
    print(mel_path)
    
   

 
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    plt.savefig(mel_path, dpi = 300)
    plt.close()

    # exit()