import os
import numpy as np
import librosa
import hmmlearn.hmm as hmm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from python_speech_features import mfcc
from dtw import dtw

# 数据路径和标签
data_path = "data/isolated_words"
labels = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero"]

# MFCC参数
numcep = 13
winlen = 0.025
winstep = 0.01
nfft = 512

# 训练和测试数据划分比例
test_size = 0.2

# 加载音频文件并提取MFCC特征
def load_data(data_path, labels):
    X = []
    y = []
    for label in labels:
        label_path = os.path.join(data_path, label)
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            signal, sr = librosa.load(file_path)
            mfcc_feat = mfcc(signal, sr, numcep=numcep, winlen=winlen, winstep=winstep, nfft=nfft)
            X.append(mfcc_feat)
            y.append(label)
    return X, y

# 训练和测试数据划分
def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# 标签编码
def encode_labels(labels):
    le = LabelEncoder()
    le.fit(labels)
    return le

# 对齐MFCC特征
def align_mfcc(X_train, X_test):
    aligned_X_train = []
    aligned_X_test = []
    for mfcc_train in X_train:
        min_dist = np.inf
        for mfcc_test in X_test:
            dist, _, _, _ = dtw(mfcc_train.T, mfcc_test.T)
            if dist < min_dist:
                min_dist = dist
                best_mfcc_test = mfcc_test
        aligned_X_train.append(mfcc_train.T)
        aligned_X_test.append(best_mfcc_test.T)
    return aligned_X_train, aligned_X_test

# 训练HMM模型
def train_hmm(X_train, y_train, n_components=4, n_iter=1000):
    models = {}
    for label in set(y_train):
        X = [x for x, y in zip(X_train, y_train) if y == label]
        lengths = [len(x) for x in X]
        model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)
        model.fit(X, lengths)
        models[label] = model
    return models

# 定义预测标签函数
def predict_label(test_mfcc, models):
    min_dist = np.inf
    pred_label = None
    for label, model in models.items():
        dist, _ = fastdtw(model, test_mfcc, dist=euclidean)
        if dist < min_dist:
            min_dist = dist
            pred_label = label
    return pred_label


# 对齐训练和测试数据的MFCC特征
aligned_X_train, aligned_X_test = align_mfcc(X_train, X_test)

# 将MFCC特征变形成3D数组，用于HMM模型训练
X_train = [np.transpose(mfcc) for mfcc in aligned_X_train]
X_test = [np.transpose(mfcc) for mfcc in aligned_X_test]

X_train_lengths = [len(mfcc) for mfcc in aligned_X_train]
X_test_lengths = [len(mfcc) for mfcc in aligned_X_test]

# 训练HMM模型
models = train_hmm(X_train, y_train)

# 测试HMM模型
correct = 0
for i, mfcc in enumerate(X_test):
    pred_label = predict_label(mfcc, models)
    true_label = y_test[i]
    if pred_label == true_label:
        correct += 1
    print(f"Test sample {i+1}, true label: {true_label}, predicted label: {pred_label}")
accuracy = correct / len(X_test)
print("Accuracy:", accuracy)
