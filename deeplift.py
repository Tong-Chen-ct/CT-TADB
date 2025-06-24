from __future__ import print_function
import tensorflow as tf
import keras
import numpy as np
import h5py
from keras.models import model_from_json
from deeplift.layers import NonlinearMxtsMode
import deeplift.conversion.kerasapi_conversion as kc
import deeplift.util
from deeplift.util import compile_func
from deeplift.visualization import viz_sequence
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib.backends.backend_pdf import PdfPages  # 添加PdfPages支持 

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from keras.initializers import glorot_uniform
print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Numpy version:", np.__version__)

def read_fasta(fasta_file):
    """读取FASTA文件并返回序列列表及其实际长度"""
    sequences = []
    sequence_lengths = []
    with open(fasta_file, 'r') as f:
        current_seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:  # 保存之前的序列
                    sequences.append(current_seq)
                    sequence_lengths.append(len(current_seq))
                current_seq = ''
            else:
                current_seq += line
        if current_seq:  # 保存最后一个序列
            sequences.append(current_seq)
            sequence_lengths.append(len(current_seq))
    return sequences, sequence_lengths

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return, sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if (one_hot_axis == 0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis == 1): 
        assert zeros_array.shape[0] == len(sequence)
    for (i, char) in enumerate(sequence):
        if (char == "A" or char == "a"):
            char_idx = 0
        elif (char == "C" or char == "c"):
            char_idx = 1
        elif (char == "G" or char == "g"):
            char_idx = 2
        elif (char == "T" or char == "t"):
            char_idx = 3
        elif (char == "N" or char == "n"):
            continue
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        if (one_hot_axis == 0):
            zeros_array[char_idx, i] = 1
        elif (one_hot_axis == 1):
            zeros_array[i, char_idx] = 1

# 读取FASTA文件
fasta_file = "logo.fa"  # 替换为您的FASTA文件路径
sequences, sequence_lengths = read_fasta(fasta_file)
onehot_data = np.array([one_hot_encode_along_channel_axis(seq) for seq in sequences])

# 调整序列长度为5000
onehot_data = pad_sequences(onehot_data,
                          maxlen=10000,          # 目标长度
                          truncating='post',    # 从后面截断
                          padding='post')       # 如果序列不足5000，在后面填充0

# 模型文件路径
keras_model_weights = "GM12878_0.001_9_64_final_weights.h5"
keras_model_json = "GM12878_0.001_9_64_architecture.json"


def load_model_workaround(json_path, weights_path):
    with open(json_path, 'r') as f:
        # 在加载模型时，显式指定 GlorotUniform 初始化器
        model = model_from_json(f.read(), custom_objects={'GlorotUniform': glorot_uniform()})
    weights_file = h5py.File(weights_path, 'r')
    for layer in model.layers:
        if layer.name in weights_file:
            layer_weights = []
            weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                          for n in weights_file[layer.name].attrs['weight_names']]
            for weight_name in weight_names:
                weight_value = weights_file[layer.name][weight_name][...]
                layer_weights.append(weight_value)
            layer.set_weights(layer_weights)
    weights_file.close()
    return model

# 加载模型
try:
    print("Attempting to load model...")
    keras_model = load_model_workaround(keras_model_json, keras_model_weights)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# 转换模型
method_to_model = OrderedDict()
for method_name, nonlinear_mxts_mode in [
    ('rescale_conv_revealcancel_fc', NonlinearMxtsMode.DeepLIFT_GenomicsDefault),
    ('rescale_all_layers', NonlinearMxtsMode.Rescale),
    ('revealcancel_all_layers', NonlinearMxtsMode.RevealCancel),
    ('grad_times_inp', NonlinearMxtsMode.Gradient),
    ('guided_backprop', NonlinearMxtsMode.GuidedBackprop)]:
    method_to_model[method_name] = kc.convert_model_from_saved_files(
        h5_file=keras_model_weights,
        json_file=keras_model_json,
        nonlinear_mxts_mode=nonlinear_mxts_mode)

# DeepLIFT分析
model_to_test = method_to_model['rescale_conv_revealcancel_fc']
deeplift_prediction_func = compile_func([model_to_test.get_layers()[0].get_activation_vars()],
                                     model_to_test.get_layers()[-1].get_activation_vars())

# 进行预测
original_model_predictions = keras_model.predict(onehot_data, batch_size=200)
converted_model_predictions = deeplift.util.run_function_in_batches(
                            input_data_list=[onehot_data],
                            func=deeplift_prediction_func,
                            batch_size=200,
                            progress_update=None)

print("maximum difference in predictions:", np.max(np.abs(converted_model_predictions - original_model_predictions)))
predictions = converted_model_predictions

# 计算重要性得分
method_to_scoring_func = OrderedDict()
for method, dl_model in method_to_model.items():
    print("Compiling scoring function for: " + method)
    # 对于单任务二分类，task_idx 应该是 0
    method_to_scoring_func[method] = dl_model.get_target_contribs_func(find_scores_layer_idx=0,
                                                                       target_layer_idx=-2)

background = OrderedDict([('A', 0.25), ('C', 0.25), ('G', 0.25), ('T', 0.25)])

# 计算每个方法的得分
method_to_task_to_scores = OrderedDict()
for method_name, score_func in method_to_scoring_func.items():
    print("on method", method_name)
    method_to_task_to_scores[method_name] = OrderedDict()
    # 只有一个任务
    task_idx = 0
    scores = np.array(score_func(
        task_idx=task_idx,
        input_data_list=[onehot_data],
        input_references_list=[
            np.array([background['A'],
                      background['C'],
                      background['G'],
                      background['T']])[None, None, :]
        ],
        batch_size=200,
        progress_update=None
    ))
    scores = np.sum(scores, axis=2)
    method_to_task_to_scores[method_name][task_idx] = scores





# 保存结果
output_dir = "logo"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 检查DeepLIFT版本
import deeplift
print(f"DeepLIFT version: {deeplift.__version__}")

# 查看plot_weights函数的参数
import inspect
print(inspect.signature(viz_sequence.plot_weights))



# 在可视化部分之前添加数据检查
print("\n检查得分数据:")
for method_name in method_to_task_to_scores:
    scores = method_to_task_to_scores[method_name][0]  # 只检查第一个任务
    print(f"Method: {method_name}")
    print(f"  Score range: {np.min(scores)} to {np.max(scores)}")
    print(f"  Score mean: {np.mean(scores)}")
    print(f"  Score std: {np.std(scores)}")

# 在代码开头添加这些导入和设置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免TkAgg相关错误
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")  # 忽略agg后端警告

# 在代码开头添加这些导入和设置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免TkAgg相关错误
import matplotlib.pyplot as plt

for seq_idx in range(len(sequences)):
    actual_length = sequence_lengths[seq_idx]
    for task in [0]:
        for method_name in method_to_task_to_scores:
            # 关闭之前的所有图形
            plt.close('all')
            
            # 获取数据
            scores = method_to_task_to_scores[method_name][task]
            scores_for_seq = scores[seq_idx][:actual_length]
            original_onehot = onehot_data[seq_idx][:actual_length]
            
            # 计算贡献值
            contributions = original_onehot * scores_for_seq[:, None]
            
            # 图1: 热图形式的碱基贡献可视化
            try:
                fig1, ax1 = plt.subplots(figsize=(20, 4))
                
                # 使用热图显示贡献值
                max_abs_val = np.max(np.abs(contributions))
                if max_abs_val > 0:
                    # 使用热图可视化
                    im = ax1.imshow(contributions.T, aspect='auto', cmap='RdBu_r', 
                               vmin=-max_abs_val, vmax=max_abs_val,
                               interpolation='nearest')
                    
                    # 添加颜色条
                    cbar = fig1.colorbar(im, ax=ax1, label='Contribution Score')
                    
                    # 设置y轴标签为碱基
                    ax1.set_yticks([0, 1, 2, 3])
                    ax1.set_yticklabels(['A', 'C', 'G', 'T'])
                    
                    # 设置x轴范围和标签
                    ax1.set_xlim(-0.5, actual_length-0.5)
                    ax1.set_xlabel('Position')
                    
                    # 每10个位置显示一个刻度
                    ax1.set_xticks(np.arange(0, actual_length, 10))
                    
                    # 添加标题
                    ax1.set_title(f'Sequence {seq_idx}, Method: {method_name}')
                    
                    # 确保图形被完全渲染
                    fig1.tight_layout()
                    
                    # 保存为PNG和PDF格式
                    #save_path_png1 = os.path.join(output_dir, f'task_{task}_seq_{seq_idx}_method_{method_name}_heatmap.png')
                    save_path_pdf1 = os.path.join(output_dir, f'task_{task}_seq_{seq_idx}_method_{method_name}_heatmap.pdf')
                    
                    # 保存PNG
                    #fig1.savefig(save_path_png1, format='png', dpi=300, bbox_inches='tight')
                    
                    # 保存PDF
                    fig1.savefig(save_path_pdf1, format='pdf', bbox_inches='tight')
                    
                    #print(f"热图已保存: {save_path_png1}")
                else:
                    print(f"警告: 序列 {seq_idx}, 方法 {method_name} 的热图贡献值全为零")
            except Exception as e:
                print(f"热图生成或保存时出错: {e}")
            finally:
                plt.close(fig1)
            
            # 关闭所有图形，确保图2是独立的
            plt.close('all')
           
            
            # 图2: 使用viz_sequence.plot_weights绘制
            try:
                # 创建一个新的图形对象
                plt.figure(figsize=(15, 4))
                
                # 直接调用plot_weights函数
                viz_sequence.plot_weights(contributions, subticks_frequency=10,height_padding_factor=1.5)
                
                # 获取当前坐标轴并调整其位置和大小
                ax = plt.gca()
                pos = ax.get_position()
                # 增加高度但保持其他参数不变
                ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 1.5])


                # 调整Y轴范围
                #plt.ylim(-0.1, 0.1)  # 根据实际权重范围调整
                # 方法1: 基于数据的最大最小值自动计算合适的范围
                data_min = np.min(contributions)
                data_max = np.max(contributions)
                y_margin = (data_max - data_min) * 0.1  # 增加10%的边距
                plt.ylim(data_min - y_margin, data_max + y_margin)
                # 设置y轴范围，最小值为0，移除负值部分
                #plt.ylim(0, data_max + y_margin)
                
                # 限制x轴以匹配实际序列长度
                plt.xlim(0, actual_length-1)
                
                # 将标题放置在y轴左侧，并竖直显示
                title_text = f'Sequence {seq_idx}\nMethod: {method_name}'
                plt.text(-0.04, 0.04, title_text, fontsize=12, rotation=90, 
                         verticalalignment='center', horizontalalignment='right', 
                         transform=plt.gca().transAxes)
                
                # 移除默认的标题
                plt.gca().set_title('')
                
                # 确保图形被完全渲染
                #plt.tight_layout()
                
                # 保存为PNG和PDF格式
                #save_path_png2 = os.path.join(output_dir, f'task_{task}_seq_{seq_idx}_method_{method_name}_weights.png')
                save_path_pdf2 = os.path.join(output_dir, f'task_{task}_seq_{seq_idx}_method_{method_name}_weights.pdf')
                
                # 保存图形 - 不调用canvas.draw()
                #plt.savefig(save_path_png2, format='png', dpi=300, bbox_inches='tight')
                plt.savefig(save_path_pdf2, format='pdf', bbox_inches='tight',dpi=300)
                
                #print(f"权重图已保存: {save_path_png2}")
                
            except Exception as e:
                print(f"权重图生成或保存时出错: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 确保关闭图形窗口
                plt.close('all')
               

# 保存得分和预测结果
for task in [0]:
    task_results = {}
    for method_name in method_to_task_to_scores:
        scores = method_to_task_to_scores[method_name][task]
        task_results[method_name] = scores
    
    for method_name, scores in task_results.items():
        # 只取每个序列指定长度的得分
        scores_truncated = [score[:156] for score in scores]
        df = pd.DataFrame(scores_truncated)
        
        # 确保列名从0开始到segment_length-1
        df.columns = range(156)
        
        save_path = os.path.join(output_dir, f'task_{task}_scores_{method_name}.csv')
        df.to_csv(save_path, index=True)

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(os.path.join(output_dir, 'model_predictions.csv'))

print(f"\nAll results have been saved to the '{output_dir}' directory")


def save_meme_format(sequences, sequence_lengths, method_to_task_to_scores, output_dir):
    for method_name in method_to_task_to_scores:
        file_path = os.path.join(output_dir, f'motif_scores_{method_name}.meme')
        
        with open(file_path, 'w') as f:
            # 对每个序列进行处理
            for seq_idx in range(len(sequences)):
                actual_length = sequence_lengths[seq_idx]
                scores = method_to_task_to_scores[method_name][0][seq_idx]  # 获取第一个任务的得分
                
                # 获取该序列实际长度的原始one-hot编码和得分
                seq_onehot = onehot_data[seq_idx][:actual_length]
                seq_scores = scores[:actual_length]
                
                # 计算每个碱基的贡献值
                base_contributions = seq_onehot * seq_scores[:, None]
                
                # 确保所有值为正数
                min_value = np.min(base_contributions)
                if min_value < 0:
                    base_contributions = base_contributions - min_value
                
                # 归一化，使每行和为1
                row_sums = np.sum(base_contributions, axis=1)
                row_sums[row_sums == 0] = 1  # 避免除以0
                normalized_scores = base_contributions / row_sums[:, np.newaxis]
                
                # 写入数据，保留6位小数
                for row in normalized_scores:
                    f.write(' '.join(f'{x:9.6f}' for x in row) + '\n')
                
                f.write('\n')  # 在序列之间添加空行

# 在代码末尾添加这行来生成MEME格式文件
save_meme_format(sequences, sequence_lengths, method_to_task_to_scores, output_dir)
print("MEME format scores have been saved in the output directory")