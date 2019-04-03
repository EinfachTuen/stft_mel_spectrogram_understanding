import run_demo
import audio_utilities
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display


def displaySTFT(STFTSpectrogram, string):
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(STFTSpectrogram), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(string)
    plt.show()


def displayExperimentSTFT(STFTSpectrogram, string):
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(STFTSpectrogram), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(string)
    plt.show()


def plotATestThingy():
    inputFileName = "bkvhi.wav"
    sample_rate_hz = 44100
    fft_size = 2048
    input_signal, sr = librosa.load("bkvhi.wav")
    #input_signal = audio_utilities.get_signal(inputFileName, expected_fs=sample_rate_hz) # this is different "normalized" ? than librosa
    print(input_signal)

    # Hopsamp is the number of samples that the analysis window is shifted after
    # computing the FFT. For example, if the sample rate is 44100 Hz and hopsamp is
    # 256, then there will be approximately 44100/256 = 172 FFTs computed per second
    # and thus 172 spectral slices (i.e., columns) per second in the spectrogram.
    hopsamp = fft_size // 8  # 2048 / 8 = 256

    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    stft = np.array([np.fft.rfft(window * input_signal[i:i + fft_size])
                     for i in range(0, len(input_signal) - fft_size, hopsamp)])  # geht immer hopsamp size weiter und nimmt die ganzen Werte
    stft = np.transpose(stft)
    print(stft)
    print("stft")
    displayExperimentSTFT(stft, "own calculated")


def librosaToCompare():
    y, sr = librosa.load("bkvhi.wav")
    stft_spectrogram = librosa.stft(y)
    print("stft_spectrogram")
    print(stft_spectrogram)
    displaySTFT(stft_spectrogram, "librosa calculated")


if __name__ == '__main__':
    plotATestThingy()
    librosaToCompare()
