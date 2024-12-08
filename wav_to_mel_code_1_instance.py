import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the WAV file
file_path = './music_data_copy/_AjXY9cHCxA.wav'
y, sr = librosa.load(file_path, sr=None)  # sr=None preserves the original sampling rate

# Generate the Mel spectrogram
n_fft = 2048  # Length of the FFT window
hop_length = 512  # Number of samples between successive frames
n_mels = 128  # Number of Mel bands
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

# Convert to decibel (log scale) for better visualization
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Plot the Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
