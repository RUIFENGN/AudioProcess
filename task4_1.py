import os
import numpy as np
import librosa
import hmmlearn.hmm as hmm
from sklearn.model_selection import train_test_split
from librosa import feature

# 加载声音文件，进行预处理
def load_data(filename, n_mfcc=13):
    y, sr = librosa.load(filename, sr=None)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    return mfcc

# 读取数据集，将不同类别的样本分别存储到列表中
def load_dataset(data_path):
    labels = []
    dataset = []
    for label in data_path:
        labels.append(label)
    for root, dirs, files in os.walk(data_path):
        for name in files:
            data_file = os.path.join(root, name)
            dataset.append(load_data(data_file))
    return dataset, labels

import os

def load_dataset(data_path):
    labels = []
    dataset = []
    for label in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, label)): # 只处理子目录
            labels.append(label)
            for file in os.listdir(os.path.join(data_path, label)):
                if file.endswith(".txt"): # 只处理txt文件
                    data_file = os.path.join(data_path, label, file)
                    dataset.append(load_data(data_file))
    return dataset, labels

# 训练HMM模型
def train_hmm_model(X, y, n_components=4, cov_type="diag", n_iter=100):
    models = {}
    labels = set(y)
    for label in labels:
        indices = [i for i, x in enumerate(y) if x == label]
        X_train = [X[i] for i in indices]
        model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)
        model.fit(X_train)
        models[label] = model
    return models

# 预测声音样本所属类别
def predict(model, X):
    scores = []
    for label, hmm_model in model.items():
        score = hmm_model.score(X)
        scores.append(score)
    best_label = max(model, key=model.get)
    return best_label

# 主函数
def main():
    data_path = "./data"
    n_mfcc = 13
    n_components = 4
    cov_type = "diag"
    n_iter = 100

    # 加载数据集
    X, y = load_dataset(data_path)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 训练模型
    models = train_hmm_model(X_train, y_train, n_components=n_components, cov_type=cov_type, n_iter=n_iter)

    # 测试模型
    n_test = len(X_test)
    correct = 0
    for i in range(n_test):
        X_sample = X_test[i]
        y_pred = predict(models, X_sample)
        if y_pred == y_test[i]:
            correct += 1
    accuracy = correct / n_test
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()