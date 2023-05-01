import os
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
from sklearn.cluster import KMeans
from hmmlearn import hmm

# 定义参数
DATA_PATH = "./data/"
SOURCE_WORD = "bed"
SAMPLE_RATE = 16000
NUM_CEPT = 13
NUM_MIXTURES = 3
NUM_TRAIN_ITERS = 100
NUM_TEST_FILES = 10

# 加载数据集
def load_data():
    all_files = os.listdir(DATA_PATH)
    source_files = [f for f in all_files if SOURCE_WORD in f]
    target_files = [f for f in all_files if SOURCE_WORD not in f][:NUM_TEST_FILES]
    source_vectors = []
    for f in source_files:
        _, signal = wavfile.read(DATA_PATH + f)
        vector = mfcc(signal, SAMPLE_RATE, numcep=NUM_CEPT)
        source_vectors.append(vector)
    target_vectors = []
    for f in target_files:
        _, signal = wavfile.read(DATA_PATH + f)
        vector = mfcc(signal, SAMPLE_RATE, numcep=NUM_CEPT)
        target_vectors.append(vector)
    return source_vectors, target_vectors

# 训练 GMM-HMM 模型
def train_model(X):
    # 聚类
    kmeans = KMeans(n_clusters=NUM_MIXTURES)
    kmeans.fit(np.vstack(X))
    means = kmeans.cluster_centers_

    # 训练 HMM 模型
    model = hmm.GMMHMM(n_components=NUM_MIXTURES, n_mix=NUM_MIXTURES)
    model.means_ = means
    model.covars_ = [np.cov(np.transpose(X[kmeans.labels_ == i])) for i in range(NUM_MIXTURES)]
    model_transmat = np.ones((NUM_MIXTURES, NUM_MIXTURES)) / NUM_MIXTURES
    model.startprob_ = model_transmat[0]
    model.transmat_ = model_transmat
    model.n_iter = NUM_TRAIN_ITERS
    model.fit([np.vstack(X)])
    return model

# 测试模型
def test_model(model, X):
    scores = [model.score(vector) for vector in X]
    return np.average(scores)

# 加载数据集
source_vectors, target_vectors = load_data()
print("Loaded data.")

# 训练模型
model = train_model(source_vectors)
print("Trained model.")

# 测试模型
accuracy = test_model(model, target_vectors)
print("Tested model. Accuracy:", accuracy)