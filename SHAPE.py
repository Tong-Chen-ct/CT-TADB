import os
import numpy as np
import pandas as pd
import pyBigWig
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import font_manager
from tensorflow.keras.models import load_model
from matplotlib import font_manager

font_path = os.path.join(os.getcwd(), '../matplotlib/ARIAL.TTF')
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'


# CUDA配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 加载标签
y_train = np.load('GM12878_train_encoded_labels.npy')
y_test = np.load('GM12878_test_encoded_labels.npy')
y_val = np.load('GM12878_val_encoded_labels.npy')

# bigWig文件路径
bigwig_files = [
    'wgEncodeBroadHistoneGm12878H4k20me1StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k9me3StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k9acStdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k79me2StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k4me3StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k4me2StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k4me1StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k36me3StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k27me3StdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H3k27acStdSig.bigWig',
    'wgEncodeBroadHistoneGm12878H2azStdSig.bigWig',
    'wgEncodeBroadHistoneGm12878CtcfStdSig.bigWig'
]

bed_files_train = ['GM12878_trian_positive_sequences.bed', 'GM12878_train_negative_sequences.bed']
bed_files_val = ['GM12878_val_positive_sequences.bed', 'GM12878_val_negative_sequences.bed']
bed_files_test = ['GM12878_test_positive_sequences.bed', 'GM12878_test_negative_sequences.bed']

# 读取bigwig信号函数
def load_histone_modification_data(bed_files, bigwig_files):
    all_features = []
    for bed_file in bed_files:
        bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=["chrom", "start", "end"])
        bed_df = bed_df[(bed_df["start"] >= 0) & (bed_df["end"] > bed_df["start"])]
        for _, row in bed_df.iterrows():
            chrom, start, end = row["chrom"], row["start"], row["end"]
            sample_features = []
            for bigwig_file in bigwig_files:
                try:
                    bw = pyBigWig.open(bigwig_file)
                    if chrom not in bw.chroms() or end > bw.chroms()[chrom]:
                        sample_features.append(0)
                        bw.close()
                        continue
                    value = bw.stats(chrom, start, end, exact=True)[0]
                    sample_features.append(value if value is not None else 0)
                    bw.close()
                except:
                    sample_features.append(0)
            all_features.append(sample_features)
    return np.array(all_features)

# 加载组蛋白信号
histone_train = load_histone_modification_data(bed_files_train, bigwig_files)
histone_val = load_histone_modification_data(bed_files_val, bigwig_files)
histone_test = load_histone_modification_data(bed_files_test, bigwig_files)

# 归一化
h_min = np.min(histone_train)
h_max = np.max(histone_train)
histone_train = (histone_train - h_min) / (h_max - h_min)
histone_val = (histone_val - h_min) / (h_max - h_min)
histone_test = (histone_test - h_min) / (h_max - h_min)

# 加载模型
model_path = "GM12878_0.0001_32_4_64_Transformer_12histone.hdf5"
model = load_model(model_path)

# 特征名提取
feature_names = [os.path.basename(f).split('Gm12878')[1].split('StdSig')[0] for f in bigwig_files]

# SHAP分析
background = histone_train[:100]
small_test = histone_test[:50]

def model_predict(x):
    return model.predict(x)

explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(small_test)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# 平均绝对值
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = dict(zip(feature_names, mean_abs_shap))
sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
sorted_features, sorted_importance = zip(*sorted_feature_importance)

# 保存平均条形图
plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(sorted_features)), sorted_importance, align='center')
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('Mean |SHAP value|')
plt.title('Average Feature Importance for GM12878 Histone Modifications')
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, value in zip(bars, sorted_importance):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{value:.5f}', va='center', ha='left')
plt.savefig('SHAP_GM12878_histone_importance_kernel.png')
plt.close()

import matplotlib.colors as mcolors

# 自定义的颜色映射：从深蓝色到紫色，再到红色
cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue_purple_red", ["#2E54A1", "#9B4D96", "#D84E41"])

# SHAP summary plot - dot with custom color map (blue -> purple -> red)
shap.summary_plot(shap_values, small_test, feature_names=feature_names, plot_type='dot', cmap=cmap, show=False)
plt.title("SHAP Summary Plot: GM12878 Histone Modifications", fontsize=14)
plt.tight_layout()
plt.savefig('SHAP_summary_dotplot_GM12878_custom_color-1200.png', dpi=1200)
plt.close()

# SHAP summary plot - bar with custom color for bar
shap.summary_plot(shap_values, small_test, feature_names=feature_names, plot_type='bar', color="#2E54A1", show=False)
plt.title("SHAP Bar Summary: GM12878 Histone Modifications", fontsize=14)
plt.tight_layout()
plt.savefig('SHAP_summary_barplot_GM12878_custom_bar_color-1200.png', dpi=1200)
plt.close()


# 输出排名
print("特征重要性排名 (KernelExplainer):")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance:.5f}")
