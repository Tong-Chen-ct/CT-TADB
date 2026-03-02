import pandas as pd
import numpy as np

# 可配置的边界扩展大小
EXPAND_SIZE = 5000  # Xkb，可根据需求修改

# 读取TAD数据
bed_data = pd.read_csv('GM12878_manually_annotated_TADs.bed', sep='\t', header=None, 
                       usecols=[0, 1, 2], names=['chr', 'start', 'end'])

# 按染色体和起始位置排序
bed_data = bed_data.sort_values(by=['chr', 'start']).reset_index(drop=True)

# 创建边界列表
boundaries = []
overlapping_tads = []

# 处理每个染色体
for chrom in bed_data['chr'].unique():
    chrom_data = bed_data[bed_data['chr'] == chrom].reset_index(drop=True)
    
    for i in range(len(chrom_data)):
        # 当前TAD
        current_tad = chrom_data.iloc[i]
        
        # 左边界：只向左扩展
        left_start = max(0, current_tad['start'] - EXPAND_SIZE)
        left_end = current_tad['start']
        
        # 右边界：只向右扩展
        right_start = current_tad['end']
        right_end = current_tad['end'] + EXPAND_SIZE
        
        # 检查左边界是否与前一个TAD的右边界重叠
        left_boundary_overlaps = False
        if i > 0:
            prev_tad = chrom_data.iloc[i-1]
            prev_right_start = prev_tad['end']
            prev_right_end = prev_tad['end'] + EXPAND_SIZE
            
            if left_start < prev_right_end and prev_right_start < left_end:
                left_boundary_overlaps = True
                overlapping_tads.append({
                    'type': 'left_with_prev_right',
                    'current_tad': current_tad.to_dict(),
                    'prev_tad': prev_tad.to_dict(),
                    'overlap_start': max(left_start, prev_right_start),
                    'overlap_end': min(left_end, prev_right_end)
                })
        
        # 检查右边界是否与下一个TAD的左边界重叠
        right_boundary_overlaps = False
        if i < len(chrom_data) - 1:
            next_tad = chrom_data.iloc[i+1]
            next_left_start = max(0, next_tad['start'] - EXPAND_SIZE)
            next_left_end = next_tad['start']
            
            if right_start < next_left_end and next_left_start < right_end:
                right_boundary_overlaps = True
                overlapping_tads.append({
                    'type': 'right_with_next_left',
                    'current_tad': current_tad.to_dict(),
                    'next_tad': next_tad.to_dict(),
                    'overlap_start': max(right_start, next_left_start),
                    'overlap_end': min(right_end, next_left_end)
                })
        
        # 只添加没有重叠的边界
        if not left_boundary_overlaps:
            boundaries.append({
                'chr': chrom,
                'start': left_start,
                'end': left_end,
                'type': 'left_boundary',
                'original_tad_start': current_tad['start'],
                'original_tad_end': current_tad['end']
            })
        
        if not right_boundary_overlaps:
            boundaries.append({
                'chr': chrom,
                'start': right_start,
                'end': right_end,
                'type': 'right_boundary',
                'original_tad_start': current_tad['start'],
                'original_tad_end': current_tad['end']
            })

# 创建边界DataFrame
boundaries_df = pd.DataFrame(boundaries)

# 检查并移除重复的边界
boundaries_df = boundaries_df.drop_duplicates(subset=['chr', 'start', 'end']).reset_index(drop=True)

# 保存边界信息到单个文件
boundaries_df[['chr', 'start', 'end']].to_csv('GM12878_TAD_boundaries_positive.bed', sep='\t', header=False, index=False)

# 输出重叠情况报告
if overlapping_tads:
    print(f"发现 {len(overlapping_tads)} 个边界重叠情况，这些重叠边界已被排除:")
    for i, overlap in enumerate(overlapping_tads):
        print(f"\n重叠 #{i+1}:")
        if overlap['type'] == 'left_with_prev_right':
            print(f"当前TAD ({overlap['current_tad']['chr']}:{overlap['current_tad']['start']}-{overlap['current_tad']['end']}) 的左边界")
            print(f"与前一个TAD ({overlap['prev_tad']['chr']}:{overlap['prev_tad']['start']}-{overlap['prev_tad']['end']}) 的右边界重叠")
        else:
            print(f"当前TAD ({overlap['current_tad']['chr']}:{overlap['current_tad']['start']}-{overlap['current_tad']['end']}) 的右边界")
            print(f"与下一个TAD ({overlap['next_tad']['chr']}:{overlap['next_tad']['start']}-{overlap['next_tad']['end']}) 的左边界重叠")
        print(f"重叠区域: {overlap['overlap_start']}-{overlap['overlap_end']}")
        print(f"重叠长度: {overlap['overlap_end'] - overlap['overlap_start']} bp")

# 计算左右边界的数量
left_count = sum(1 for b in boundaries if b['type'] == 'left_boundary')
right_count = sum(1 for b in boundaries if b['type'] == 'right_boundary')

print(f"已成功生成TAD边界文件，共包含 {len(boundaries_df)} 个非重叠边界")
print(f"其中左边界 {left_count} 个，右边界 {right_count} 个")
