import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav

# 读取语音文件
sample_rate, speech = wav.read('speech.wav')

# 设置判决参数
energy_thr = 3000000
zero_crossing_thr = 270
N = 32
M = 40

# 计算语音信号能量
speech_energy = np.sum(np.abs(speech)**2)

# 第一级判决：能量门限
speech_start = []
speech_end = []
N_count = 0
for i in range(len(speech)):
    energy = np.sum(np.abs(speech[i:i+N])**2)
    if energy > energy_thr:
        N_count += 1
        if N_count >= N:
            speech_start.append(i - N_count)
            N_count = 0
    else:
        N_count = 0
    if len(speech_start) > len(speech_end):
        if i > speech_start[-1] + M:
            speech_end.append(i - M)

# 第二级判决：过零率门限
speech_start_final = []
speech_end_final = []
for i in range(len(speech_start)):
    zero_crossing_count = np.sum(np.abs(np.diff(np.sign(speech[speech_start[i]:speech_end[i]])))) / 2
    if zero_crossing_count > zero_crossing_thr:
        speech_start_final.append(speech_start[i])
        speech_end_final.append(speech_end[i])

# 输出检测结果
print("语音段数:", len(speech_start_final))
for i in range(len(speech_start_final)):
    print("第", i+1, "个语音段起始点:", speech_start_final[i]/sample_rate, "s")
    print("第", i+1, "个语音段结束点:", speech_end_final[i]/sample_rate, "s")