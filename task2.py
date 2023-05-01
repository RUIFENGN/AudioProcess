import numpy as np
import librosa
import audioread as ar
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav

# 读入语音信号文件
audio_file ='beijing.wav'
signal, sample_rate = librosa.load(audio_file, sr=None)
n = len(signal)

print(sample_rate)
# 定义窗长和帧移参数
win_size = 0.020  # 窗长为 25ms
hop_size = 0.010  # 帧移为 10ms

# 计算窗函数长度、帧移和帧数
window_length = int(win_size * sample_rate)
hop_length = int(hop_size * sample_rate)
num_frames = 1 + int(np.ceil((len(signal) - window_length) / hop_length))

# 计算窗函数
window = np.hamming(window_length)

# 分帧加窗
frames = librosa.util.frame(signal, frame_length=window_length, hop_length=hop_length, axis=0)
new_frames = frames.copy()
new_frames *= window

# 对每一帧计算短时自相关函数
stacfs = []
for frame in new_frames:
    stacf = librosa.core.autocorrelate(frame)
    stacfs.append(stacf[5:])

#print(stacfs)
# 对每一帧计算的短时自相关函数计算最大值
peaks = []
for stacf in stacfs:
    peak = np.argmax(stacf)
    peaks.append(peak+5)

print(peaks)
# 计算基音频率
fund_freqs = []
for peak in peaks:
    if peak > 5:
        fund_freq = sample_rate / peak
    else:
        fund_freq = 0
    fund_freqs.append(fund_freq)

# 打印预估的基音频率
print(fund_freqs)
print(np.average(fund_freqs))

# 设置判决参数
energy_thr = 1
zero_crossing_thr = 270
N = 32
M = 40

# 计算短时能量、短时平均幅度、短时平均过零率等参数
energy = np.zeros(num_frames)
rms = np.zeros(num_frames)
zcr = np.zeros(num_frames)

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

# 两级判决法
speech_frame_thresh = 10
energy_threshold = np.mean(energy) * 0.5 # 能量阈值
#print(energy_threshold)
speech_frames = []
#print(energy)
for i in range(len(energy)):
    if energy[i] > energy_threshold: # 第一级判决
        count = 0
        #print("str:"+ str(i*hop_length))
        for j in range(i, min(len(energy), i+int(sample_rate/5000))): # 第二级判决
            if energy[j] > energy_threshold:
                count += 1
            else:
                count = 0
            if count >= int(sample_rate / 10000):  # 达到一定连续帧数量，判决为语音起始点或终点
                if len(speech_frames) > 0 and i*int(sample_rate/100) - speech_frames[-1][1] < int(sample_rate / 33):  # 判断延续语音区间
                    speech_frames[-1] = (speech_frames[-1][0], j * hop_length)  # 更新上一帧的终止位置
                else:
                    speech_frames.append((i * hop_length, j * hop_length))  # 新增一组语音区间
                i = j + int(speech_frame_thresh)  # 跳过一个最小连续帧的长度，并更新起始点 i
                break

# 输出语音区域的开始和结束位置
for start, end in speech_frames:
    print('Speech region:', start/sample_rate, end/sample_rate)