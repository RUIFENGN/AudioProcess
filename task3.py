import numpy as np
import librosa
import soundfile

# 加载音频文件
audio_file = "beijing.wav"
y, sr = librosa.load(audio_file, sr=16000)

# 产生噪声
noise = 0.005 * np.random.normal(size=len(y))

# 添加噪声
noisy_signal = y + noise

# 将添加噪声后的信号保存到文件
soundfile.write("audio_noised.wav", noisy_signal, int(sr))

# 设置噪声功率谱门限下限
threshold1 = 5
threshold2 = 7
threshold3 = 20

# 计算带噪声的信号的功率谱
signal_power_spectrum = np.square(np.abs(np.fft.fft(y)))
noisy_power_spectrum = np.square(np.abs(np.fft.fft(noisy_signal)))
noise_spectrum = np.square(np.abs(np.fft.fft(noise)))

# 计算谱减噪声功率谱
subtracted_power_spectrum = np.maximum(noisy_power_spectrum - threshold1, 0)

# 计算加噪声信号的相位谱
phase_spectrum = np.angle(np.fft.fft(noisy_signal))

# 重构放音信号
reconstructed_spectrum = np.sqrt(subtracted_power_spectrum) * np.exp(1j * phase_spectrum)
reconstructed_signal = np.real(np.fft.ifft(reconstructed_spectrum))

# 将降噪后的信号保存到文件
soundfile.write("audio_denoised.wav", reconstructed_signal, int(sr))
