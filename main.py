import numpy as np
import librosa
import audioread as ar
import matplotlib.pyplot as plt

# 读入语音信号文件
audio_file ='beijing.wav'
signal, sample_rate = librosa.load(audio_file, sr=None)
n = len(signal)

# 定义窗长和帧移参数
win_size = 0.025  # 窗长为 25ms
hop_size = 0.010  # 帧移为 10ms

# 计算窗函数长度、帧移和帧数
window_length = int(win_size * sample_rate)
hop_length = int(hop_size * sample_rate)
num_frames = 1 + int(np.ceil((len(signal) - window_length) / hop_length))
print(window_length,hop_length,num_frames)

# 计算窗函数
window = np.hamming(window_length)
#window = np.hanning(window_length)
#window = np.blackman(window_length)
#window = np.ones(window_length)
# 分帧加窗
frames = librosa.util.frame(signal, frame_length=window_length, hop_length=hop_length, axis=0)
new_frames = frames.copy()
new_frames *= window
print(frames.shape)

# 打印分帧结果
print('分帧后的形状：', new_frames.shape)
print('前5帧的样本数：', new_frames[:5, :5])


# 计算短时能量、短时平均幅度、短时平均过零率等参数
energy = np.zeros(num_frames)
rms = np.zeros(num_frames)
zcr = np.zeros(num_frames)
'''
# 计算时域参数
energy = np.sum(new_frames**2, axis=0)
amplitude = np.mean(np.abs(new_frames), axis=0)
zcr = librosa.feature.zero_crossing_rate(signal, frame_length=window_length, hop_length=hop_length)
'''
for i in range(num_frames):
    # 计算当前帧的起始和结束位置
    start = i * hop_length
    end = start + window_length
    # 截取当前帧的信号
    frame = signal[start:end]
    # 计算短时能量
    energy[i] = np.sum(frame ** 2)
    # 计算短时平均幅度
    rms[i] = np.sqrt(np.mean(frame ** 2))
    # 计算短时平均过零率
    zcr[i] = np.mean(np.abs(np.diff(np.sign(frame))))

print(zcr)
# 绘制短时能量和短时过零率图像
time = np.arange(num_frames) * hop_size
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
ax[0].plot(time, energy, color='blue')
ax[0].set_ylabel('Energy')
ax[1].plot(time, zcr, color='red')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('ZCR')
ax[2].plot(time, rms, color='black')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('RMS')
plt.show()

# 绘制时域波形
time = np.arange(n) / sample_rate
plt.figure(figsize=(8, 6))
plt.plot(time, signal, color='black')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.show()

# 绘制分帧后的波形
plt.figure(figsize=(8, 6))
plt.plot(np.ravel(new_frames), color='black')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Framed Signal')
plt.show()

# 计算短时傅里叶变换
spec = np.abs(librosa.stft(signal, n_fft=window_length, hop_length=hop_length, win_length=window_length, window=window))

# 绘制短时谱图
spec_db = librosa.amplitude_to_db(spec, ref=np.max)
freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=window_length)
times = librosa.frames_to_time(np.arange(spec.shape[1]), sr=sample_rate, hop_length=hop_length)
plt.figure(figsize=(8, 6))
plt.pcolormesh(times, freqs, spec_db, cmap='viridis')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Short-time Fourier Transform')
plt.show()

