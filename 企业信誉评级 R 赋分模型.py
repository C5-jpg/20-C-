import pandas as pd
import os

# --- 0. 文件路径配置 ---
# 详细注释：请在此处配置您的文件路径。
file_path_1 = r'D:\google-downloads\附件1：123家有信贷记录企业的相关数据.xlsx'
file_path_2 = r'D:\google-downloads\附件2：302家无信贷记录企业的相关数据.xlsx'

# --- 1. 数据加载 (Data Loading) ---
# 详细注释：我们只需要加载两个文件的'企业信息'表。
# '附件1'包含我们需要的信誉评级，'附件2'用于获取完整的企业名单。
try:
    info1 = pd.read_excel(file_path_1, sheet_name='企业信息')
    info2 = pd.read_excel(file_path_2, sheet_name='企业信息')
    print("企业信息文件加载成功。")
except FileNotFoundError as e:
    print(f"文件加载错误: {e}。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"读取Excel文件时发生错误: {e}。")
    exit()

# --- 2. 数据整合 (Data Consolidation) ---
# 详细注释：创建一个包含所有企业代号的完整列表，以确保最终结果覆盖所有公司。
all_companies = pd.concat(
    [info1[['企业代号']], info2[['企业代号']]],
    ignore_index=True
)
print(f"数据整合完成，总共 {len(all_companies)} 家企业。")

# --- 3. 信誉评级赋分 (Credit Rating Scoring) ---
# 详细注释：这是本脚本的核心。我们将分类评级转换为数值分数。

# a. 定义信誉评级到分数的映射字典
#    这是根据您提供的规则建立的。
rating_map = {
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 0
}

# b. 提取有信誉评级的企业信息
#    我们只关心'企业代号'和'信誉评级'这两列。
rated_companies = info1[['企业代号', '信誉评级']].copy()

# c. 应用映射关系，创建新分数列
#    .map() 函数会根据字典将'信誉评级'列中的每个值（A, B, C, D）替换为对应的分数（10, 8, 5, 0）。
rated_companies['信誉评级R_Score'] = rated_companies['信誉评级'].map(rating_map)

print("信誉评级到分数的映射转换完成。")

# --- 4. 合并结果 (Merge Results) ---
# 详细注释：将赋分后的结果与完整的企业列表进行合并。

# a. 使用左合并（left merge），以 all_companies 为基准
#    这样可以确保所有企业都在最终的列表中。
results_df = pd.merge(all_companies, rated_companies, on='企业代号', how='left')

# b. 处理没有信誉评级的企业
#    来自'附件2'的企业在合并后，其'信誉评级'和'信誉评级R_Score'列会是 NaN (Not a Number)。
#    我们将其分数填充为0，因为无信贷记录本身就是一种高风险信号，与D级类似。
results_df['信誉评级R_Score'].fillna(0, inplace=True)
results_df['信誉评级'].fillna('无评级', inplace=True) # 给一个描述性文本

# c. 转换分数列为整数类型，使其更美观
results_df['信誉评级R_Score'] = results_df['信誉评级R_Score'].astype(int)

# --- 5. 结果展示 (Result Presentation) ---
print("\n--- 企业信誉评级 R 赋分结果 (部分展示) ---")
# 为了更好地展示，我们显示一些有评级和无评级的混合样本
# 这里我们展示E1-E5（有评级）和E124-E128（无评级）的样本
sample_e1_e5 = results_df[results_df['企业代号'].isin([f'E{i}' for i in range(1, 6)])]
sample_e124_e128 = results_df[results_df['企业代号'].isin([f'E{i}' for i in range(124, 129)])]
display_sample = pd.concat([sample_e1_e5, sample_e124_e128])

print(display_sample)

# 您可以取消下面的注释，查看所有结果或将结果保存到文件中
print("\n--- 全部结果 ---")
print(results_df)
output_filename = '企业信誉评级R分数.xlsx'
results_df.to_excel(output_filename, index=False)
print(f"\n结果已保存至 '{output_filename}'")
