import torch
import torch.nn.functional as F

import torchaudio

import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import window_sumsquare
from librosa.filters import mel as librosa_mel_fn

def get_mel_basis(sampling_rate=22050, filter_length=1024, n_mel_channels=80,  mel_fmin=0.0, mel_fmax=8000.0):
    mel_basis = librosa_mel_fn(
        sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)  # shape=(n_mels, 1 + n_fft/2)
    mel_basis = torch.from_numpy(mel_basis).float()
    return mel_basis

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

class Inverse(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(Inverse, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

        scale = filter_length / hop_length
        fourier_basis = np.fft.fft(np.eye(filter_length))
        cutoff = int((filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window != None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def forward(self, magnitude, phase):

        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window != None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :,
                              approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
            inverse_transform = inverse_transform[:, :, int(
                self.filter_length/2):]
            inverse_transform = inverse_transform[:,
                                                  :, :-int(self.filter_length/2):]

            return inverse_transform

def griffin_lim(magnitudes,  inverse, n_iters=30, filter_length=1024, hop_length=256, win_length=1024,):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        stft = torch.stft(signal, n_fft=filter_length, hop_length=hop_length,
                          win_length=win_length, window=torch.hann_window(win_length))
        real = stft[:, :, :, 0]
        imag = stft[:, :, :, 1]

        angles = torch.autograd.Variable(
            torch.atan2(imag.data, real.data))

        signal = inverse(magnitudes, angles).squeeze(1)
    return signal

def mel2wav(mel_outputs, n_iters=30, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0):
    mel_decompress = dynamic_range_decompression(mel_outputs)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

    mel_basis = librosa_mel_fn(
        sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)  # shape=(n_mels, 1 + n_fft/2)
    mel_basis = torch.from_numpy(mel_basis).float()

    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    inverse = Inverse(filter_length=filter_length,
                      hop_length=hop_length, win_length=win_length)
    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), inverse, n_iters, filter_length=filter_length, hop_length=hop_length, win_length=win_length)
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    return audio

class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(STFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        mel_basis = get_mel_basis(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax) #shape=(n_mels, 1 + n_fft/2)
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        stft = torch.stft(y,n_fft=self.filter_length, hop_length=self.hop_length,win_length=self.win_length,window=torch.hann_window(self.win_length))
        real = stft[:, :, :, 0]
        imag = stft[:, :, :, 1]

        magnitudes = torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))

        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

def load_wav(full_path, resample_rate=True, resample_rate_value=22500):
    data,sampling_rate  = torchaudio.load(full_path)
    if resample_rate and resample_rate_value != sampling_rate :
        resample = torchaudio.transforms.Resample(sampling_rate, resample_rate_value)
        data = resample(data)
        return data[0], resample_rate_value
    return data[0], resample_rate_value
