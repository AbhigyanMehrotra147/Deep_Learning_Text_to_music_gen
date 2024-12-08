import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the WAV file
wav_directory = './music_data_copy'
magnitude_directory = './Magnitude_Spectrograms'

# Ensure output directory exists
os.makedirs(magnitude_directory, exist_ok=True)

for file_name in os.listdir(wav_directory):
    file_name_tokens = file_name.split('.')
    if file_name_tokens[1] != "wav":  # Ensure file format is WAV
        continue
    file_path = os.path.join(wav_directory, file_name)
    y, sr = librosa.load(file_path, sr=None)  # sr=None preserves the original sampling rate

    # Generate the magnitude spectrogram
    n_fft = 2048  # Length of the FFT window
    hop_length = 512  # Number of samples between successive frames
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))  # Magnitude of STFT

    # Convert to decibel (log scale) for better visualization
    magnitude_spec_db = librosa.amplitude_to_db(D, ref=np.max)

    # Save the spectrogram image
    magnitude_path = os.path.join(magnitude_directory, file_name_tokens[0] + ".png")
    print(f"Saving: {magnitude_path}")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(magnitude_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Magnitude Spectrogram')
    plt.tight_layout()

    plt.savefig(magnitude_path, dpi=300)
    plt.close()
