import os

# 输入文件目录路径和要替换的字符串和新字符串
directory = "./data/happy"
new_str = "happy_"

count = 1
# 遍历目录下所有文件名
for filename in os.listdir(directory):
    if filename.endswith(".wav"): # 只处理txt文件
        # 构建旧文件路径和新文件路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_str + str(count)+ ".wav")
        # 重命名文件
        os.rename(old_file, new_file)
        # 更新下标
        count += 1