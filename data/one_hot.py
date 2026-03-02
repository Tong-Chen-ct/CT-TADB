import pandas as pd
import numpy as np

# 定义独热编码函数，不调整序列长度
def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping.get(i, 0) for i in seq]  # 默认值为0，处理非ACGT字符
    encoded = np.eye(4)[seq2]
    return encoded

# 处理训练数据集
def process_train_dataset(csv_file_path):
    df = pd.read_csv(csv_file_path)
    sequences = list(df['Sequence'])
    labels = list(df['Label'])

    # 对训练数据进行独热编码
    data = []
    for seq in sequences:
        encoded_seq = one_hot_encode(seq)
        data.append(encoded_seq)

    labels = np.array(labels)

    return data, labels

# 主函数，只处理训练数据
def main():
    csv_file_path = 'GM12878_TAD_boundaries_combined_labeled.csv'  # 请替换为你的训练集CSV文件路径
    train_data, train_labels = process_train_dataset(csv_file_path)

    # 保存编码后的训练数据和标签
    np.save('GM12878_Independent_encoded_data.npy', train_data, allow_pickle=True)
    np.save('GM12878_Independent_encoded_labels.npy', train_labels)

    print("训练数据编码和保存成功")
    print(f"编码数据中的序列数量: {len(train_data)}")
    print(f"标签形状: {train_labels.shape}")
    print(f"第一个序列的形状: {train_data[0].shape}")

if __name__ == "__main__":
    main()
