# -*- coding: utf-8 -*-
#采用CNN和Transformer结合的模型，输入DNA序列加组蛋白修饰（不加距离特征）
import os
import sys
import numpy as np
from numpy import array, argmax
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pyBigWig
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Activation, Flatten, Dropout, LayerNormalization,
    MultiHeadAttention, Layer, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Embedding, Add
from tensorflow.keras.regularizers import l2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


x_train = np.load('GM12878_train_encoded_data.npy')
x_test = np.load('GM12878_test_encoded_data.npy')
x_val = np.load('GM12878_val_encoded_data.npy')
y_train = np.load('GM12878_train_encoded_labels.npy')
y_test = np.load('GM12878_test_encoded_labels.npy')
y_val = np.load('GM12878_val_encoded_labels.npy')

bigwig_files = ['wgEncodeBroadHistoneGm12878H4k20me1StdSig.bigWig',
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
                'wgEncodeBroadHistoneGm12878CtcfStdSig.bigWig']

bed_files_train_anchors1 = ['GM12878_trian_positive_sequences.bed', 'GM12878_train_negative_sequences.bed']
bed_files_val_anchors1 = ['GM12878_val_positive_sequences.bed', 'GM12878_val_negative_sequences.bed']
bed_files_test_anchors1 = ['GM12878_test_positive_sequences.bed', 'GM12878_test_negative_sequences.bed']



def load_histone_modification_data(bed_files, bigwig_files):
    all_features = []
    for bed_file in bed_files:
        print(f"Processing BED file: {bed_file}")
        bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=["chrom", "start", "end"])
        
        bed_df = bed_df[(bed_df["start"] >= 0) & (bed_df["end"] > bed_df["start"])]
        
        for index, row in bed_df.iterrows():
            chrom, start, end = row["chrom"], row["start"], row["end"]
            sample_features = []
            for bigwig_file in bigwig_files:
                try:
                    bw = pyBigWig.open(bigwig_file)
                    
                    if chrom not in bw.chroms():
                        sample_features.append(0)
                        bw.close()
                        continue
                    
                    if end > bw.chroms()[chrom]:
                        sample_features.append(0)
                        bw.close()
                        continue
                    
                    value = bw.stats(chrom, start, end, exact=True)[0]
                    sample_features.append(value if value is not None else 0)
                    bw.close()
                except Exception as e:
                    sample_features.append(0)
                    
            all_features.append(sample_features)
    
    return np.array(all_features)


histone_train = load_histone_modification_data(bed_files_train_anchors1, bigwig_files)
histone_val = load_histone_modification_data(bed_files_val_anchors1, bigwig_files)
histone_test = load_histone_modification_data(bed_files_test_anchors1, bigwig_files)

# 归一化 Histone 数据
h_min = np.min(histone_train)  # 使用训练集计算最小值
h_max = np.max(histone_train)  # 使用训练集计算最大值

histone_train = (histone_train - h_min) / (h_max - h_min)
histone_val = (histone_val - h_min) / (h_max - h_min)  # 使用训练集的 min/max 归一化验证集
histone_test = (histone_test - h_min) / (h_max - h_min)  # 使用训练集的 min/max 归一化测试集

print(histone_train.shape) 
print(histone_test.shape) 
print(histone_val.shape) 


# Set some parameters
input_shape = x_train.shape[1:3]
kernel_size = 9
learning_rate = 0.001
num_kernels = 64
output_file = 'dm3.kc167'
max_position_embeddings =  max(x_train.shape[1], 512)  
transformer_dim = 32
embedding_dim = transformer_dim
num_heads = 4
ff_dim = 32
name = "GM12878"

np.random.seed(1671)
seed = 1671
np.random.seed(seed)

# network and training
NB_EPOCH = 150
BATCH_SIZE = 16
VERBOSE = 1
NB_CLASSES = 2  # number of classes
METRICS = ['accuracy']
LOSS = 'binary_crossentropy'
KERNEL_INITIAL = 'glorot_uniform'


# SaveHistory function for saving training history
def SaveHistory(Tuning, outfile):
    Hist = np.empty(shape=(len(Tuning.history['loss']), 4))
    Hist[:, 0] = Tuning.history['val_loss']
    Hist[:, 1] = Tuning.history['val_accuracy']
    Hist[:, 0] = Tuning.history['loss']
    Hist[:, 1] = Tuning.history['accuracy']
    np.savetxt(outfile, Hist, fmt='%.8f', delimiter=",", header="val_loss,val_acc,train_loss,train_acc", comments="")
    return Hist

# GetMetrics function for model evaluation
def GetMetrics(model, x, y):
    # 确保输入是一个包含两个张量的列表
    assert isinstance(x, list) and len(x) == 2, "Input x must be a list with two elements [dna_input, histone_input]"
    pred_p = model.predict(x)
    pred = (pred_p > 0.5).astype("int32")
    fpr, tpr, thresholdTest = roc_curve(y, pred_p)
    aucv = auc(fpr, tpr)
    precision, recall, fscore, support = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    print('auc,accuracy,mcc,precision,recall,fscore,support:', aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred), precision, recall, fscore, support)
    return [aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred), precision, recall, fscore, support]

# Transformer block definition
def transformer_block(inputs, num_heads, ff_dim, dropout=0.1):
    # Adding positional embeddings
    input_shape = inputs.shape  # Assumes shape is (batch_size, seq_len, embedding_dim)
    seq_len = input_shape[1]  # Sequence length
    embedding_dim = input_shape[-1]  # Embedding dimension
    
    # Positional embedding layer
    position_indices = tf.range(start=0, limit=seq_len, delta=1)  # Generate position indices
    position_embeddings = Embedding(input_dim=max_position_embeddings, output_dim=embedding_dim)(position_indices)
    
    # Add position embeddings to input
    inputs_with_position = Add()([inputs, position_embeddings])
    
    #Multi-head attention
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=dropout)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    #Feed-forward network
    ffn = Dense(ff_dim, activation="relu")(attention)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)
    ffn = LayerNormalization(epsilon=1e-6)(ffn)
    
    return ffn


# X CNN layers followed by a Transformer
def two_CNN_Transformer(x_train, y_train, x_test, y_test,x_val,y_val,histone_train,histone_val,histone_test,learning_rate, INPUT_SHAPE, KERNEL_SIZE, NUM_KERNEL, transformer_dim, num_heads, name):

    dna_input = Input(shape=INPUT_SHAPE, name="DNA_Input")

    x = Conv1D(NUM_KERNEL, kernel_size=KERNEL_SIZE, kernel_initializer='glorot_uniform')(dna_input)
    #x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    #x = MaxPooling1D()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    x = Conv1D(NUM_KERNEL, kernel_size=KERNEL_SIZE, kernel_initializer='glorot_uniform')(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    #x = MaxPooling1D()(x) 
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
#    x = Conv1D(NUM_KERNEL, kernel_size=KERNEL_SIZE, kernel_initializer='glorot_uniform')(x)
#    x = Activation("relu")(x)
#    x = Dropout(0.3)(x)
#    #x = MaxPooling1D()(x)   
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    
    histone_input = Input(shape=(histone_train.shape[1],), name="Histone_Input")
    print(histone_input.shape)
    histone_features = tf.expand_dims(histone_input, axis=2) 
    histone_features = tf.tile(histone_features, [1, 1, num_kernels])

    combined_features = Concatenate(axis=1)([x, histone_features])
    
    transformer_output = transformer_block(combined_features, num_heads=num_heads, ff_dim=transformer_dim)
    flattened_transformer_output = Flatten()(transformer_output)
    output = Dense(1, activation="sigmoid", name="Output")(flattened_transformer_output)


    model = Model(inputs=[dna_input, histone_input], outputs=output)
    model.compile(loss=LOSS, optimizer=Adam(learning_rate=learning_rate), metrics=METRICS)

    filepath = "_".join([name, str(learning_rate), str(kernel_size), str(num_kernels), str(transformer_dim),str(num_heads),str(ff_dim)]) + "_CNN_Transformer_DNA_12histone.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=8)

    print(model.summary())
    Tuning = model.fit([x_train,histone_train], y_train,validation_data=([x_val,histone_val], y_val), batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,callbacks=[checkpoint, early_stopping_monitor])

    print("train," + filepath)
    saved_model = load_model(filepath)
    GetMetrics(saved_model, [x_train, histone_train], y_train)
    print("test," + filepath)
    GetMetrics(saved_model, [x_test, histone_test], y_test)
    SaveHistory(Tuning, "_".join([name, str(learning_rate), str(kernel_size), str(num_kernels), str(transformer_dim),str(num_heads),str(ff_dim)]) + "_CNN_Transformer_DNA_12histone.txt")
    
    return Tuning, model

# Call the function
Tuning, model = two_CNN_Transformer(x_train, y_train, x_test, y_test,x_val,y_val,histone_train,histone_val,histone_test,learning_rate,input_shape, kernel_size, num_kernels, transformer_dim, num_heads, name)


