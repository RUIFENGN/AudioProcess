import numpy as np
import librosa
from dtw import dtw
from librosa import feature

# 计算MFCC特征
def calculate_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T

# 计算两段音频之间的DTW距离
def calculate_dtw(audio_file_1, audio_file_2):
    mfccs_1 = calculate_mfcc(audio_file_1)
    mfccs_2 = calculate_mfcc(audio_file_2)
    dist, _, _, _ = dtw(mfccs_1, mfccs_2, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    return dist

# 定义HMM模型
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)

# 训练HMM模型
def train_hmm(audio_files):
    mfccs = np.vstack((calculate_mfcc(audio_file) for audio_file in audio_files))
    model.fit(mfccs)
    return model

# 预测给定音频的孤立词
def predict_word(audio_file, trained_hmm):
    mfccs = calculate_mfcc(audio_file)
    score = trained_hmm.score(mfccs)
    return score


import os
import random

# 获取训练集文件夹中的文件列表
def get_audio_files(folder_path):
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

# 读取训练集文件列表
train_folder_path = "D:/python_code/audio_Test1/data/train2/"
train_audio_files = get_audio_files(train_folder_path)

# 训练HMM模型
trained_hmm = train_hmm(train_audio_files)

# 读取训练集文件列表
test_folder_path = "D:/python_code/audio_Test1/data/test2/"
test_audio_files = get_audio_files(test_folder_path)

# 存储每个训练集音频文件对应的单词和得分
train_scores = {}
for train_file in train_audio_files:
    word = os.path.basename(train_file).split("_")[0]
    score = predict_word(train_file, trained_hmm)
    train_scores[word] = score

# 评估测试集文件
for test_file in test_audio_files:
    predicted_score = predict_word(test_file, trained_hmm)
    min_distance = float('inf')
    predicted_word = ""
    for word, score in train_scores.items():
        distance = calculate_dtw(test_file, os.path.join(train_folder_path, word + "_1.wav"))
        if distance < min_distance:
            min_distance = distance
            predicted_word = word
    print("Test file: {}, True word: {}, Predicted word: {}, Predicted score: {}".format(test_file, os.path.basename(test_file).split("_")[0], predicted_word, predicted_score))