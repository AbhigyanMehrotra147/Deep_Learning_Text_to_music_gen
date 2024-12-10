import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# Load WAV File Function
def load_wav(file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resample = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample(waveform)
    return waveform, target_sample_rate

# # Save Reconstructed Waveform to WAV File
# def save_wav(file_path, waveform, sample_rate):
#     torchaudio.save(file_path, waveform, sample_rate)

def save_wav(file_path, waveform, sample_rate):
    # Ensure waveform is detached, moved to CPU, and in the correct shape
    waveform = waveform.detach().cpu() if waveform.requires_grad else waveform
    if waveform.dim() == 1:  # If 1D, add channel dimension
        waveform = waveform.unsqueeze(0)
    torchaudio.save(file_path, waveform, sample_rate)


# STFT Module
class STFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft)

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        if waveform.dim() != 2:
            raise ValueError("Input waveform must be 1D or 2D (batch_size, signal_length)")

        complex_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            normalized=True,
            return_complex=True,
        )

        magnitude = complex_spec.abs()
        phase = torch.angle(complex_spec)
        return magnitude, phase


# ISTFT Module
class ISTFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft)

    def forward(self, magnitude, phase):
        complex_spec = torch.polar(magnitude, phase)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
        )


# 1D U-Net
class UNet1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UNet1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return latent, output


# Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DiffusionModel, self).__init__()
        self.unet = UNet1D(input_dim, latent_dim)

    def forward(self, x, noise_level):
        latent, denoised = self.unet(x)
        return latent, denoised


# Diffusion Magnitude Autoencoding (DMAE)
class DMAE(nn.Module):
    def __init__(self):
        super(DMAE, self).__init__()
        self.stft = STFT(n_fft=1024, hop_length=256)
        self.istft = ISTFT(n_fft=1024, hop_length=256)
        self.encoder = nn.Conv1d(513, 64, kernel_size=3, stride=2, padding=1)
        self.decoder = DiffusionModel(64, 513)

    def forward(self, waveform, noise):
        magnitude, phase = self.stft(waveform)  # [batch_size, freq_bins, time_frames]
        
        latent = self.encoder(magnitude)  # [batch_size, latent_channels, latent_frames]
        
        noise = noise.unsqueeze(1)  # Add channel dimension -> [batch_size, 1, signal_length]
        noise = F.interpolate(noise, size=latent.shape[-1], mode="linear", align_corners=False)  # Match time_frames
        noise = noise.expand_as(latent)  # Expand noise channels to match latent channels
        
        reconstructed_magnitude = self.decoder(latent + noise, noise_level=0.5)[1]
        reconstructed_magnitude = F.interpolate(
            reconstructed_magnitude, size=phase.shape[-1], mode="linear", align_corners=False
        )  # Resize to match phase
        
        reconstructed_waveform = self.istft(reconstructed_magnitude, phase)
        
        return latent, reconstructed_waveform


# Parameters
batch_size = 1  # Processing one audio file at a time
noise_level = 0.05

# Load WAV File
file_path = "./music_data/_AjXY9cHCxA.wav"
waveform, sample_rate = load_wav(file_path)

if waveform.dim() == 3:  # Handle shape [batch_size, 1, signal_length]
    waveform = waveform.squeeze(1)
print(waveform.shape)

# Add noise
noise = torch.randn_like(waveform.squeeze(1)) * noise_level

# Initialize the DMAE model
dmae = DMAE()

# Pass the waveform through the DMAE model
latent, reconstructed_waveform = dmae(waveform, noise)

# Truncate to the same length for MSE calculation
min_length = min(waveform.shape[-1], reconstructed_waveform.shape[-1])
waveform_truncated = waveform[..., :min_length]
reconstructed_truncated = reconstructed_waveform[..., :min_length]

print("DMAE operation completed successfully!")
print("Latent representation shape:", latent.shape)
print("Reconstructed waveform shape:", reconstructed_waveform.shape)

# Calculate MSE
mse = torch.mean((waveform_truncated.squeeze(1) - reconstructed_truncated) ** 2).item()
print(f"Reconstruction error (MSE): {mse:.6f}")

# Save the reconstructed waveform to a WAV file
save_wav("reconstructed.wav", reconstructed_truncated, sample_rate)

