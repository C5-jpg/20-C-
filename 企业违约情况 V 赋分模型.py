import pandas as pd
import os

# --- 0. File Path Configuration ---
# 详细注释：请在此处配置您的文件路径。
file_path_1 = r'D:\google-downloads\附件1：123家有信贷记录企业的相关数据.xlsx'
file_path_2 = r'D:\google-downloads\附件2：302家无信贷记录企业的相关数据.xlsx'

# --- 1. Data Loading ---
# 详细注释：我们只需要加载两个文件的'企业信息'表。
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

# --- 2. Data Consolidation ---
# 详细注释：创建一个包含所有企业代号的完整列表。
all_companies = pd.concat(
    [info1[['企业代号']], info2[['企业代号']]],
    ignore_index=True
)
print(f"数据整合完成，总共 {len(all_companies)} 家企业。")

# --- 3. Default Status Scoring ---
# 详细注释：这是本脚本的核心，将'是'/'否'的违约情况转换为数值分数。

# a. 定义违约情况到分数的映射字典
#    根据您提供的规则：有违约(是)=3分，无违约(否)=9分
status_map = {
    '是': 3,
    '否': 9
}

# b. 提取有信贷记录的企业的违约信息
#    我们只关心'企业代号'和'是否违约'这两列。
default_status_df = info1[['企业代号', '是否违约']].copy()

# c. 应用映射关系，创建新分数列
#    .map() 函数会根据字典将'是否违约'列中的每个值替换为对应的分数。
default_status_df['违约情况V_Score'] = default_status_df['是否违约'].map(status_map)

print("违约情况到分数的映射转换完成。")

# --- 4. Merge Results ---
# 详细注释：将赋分后的结果与完整的企业列表进行合并。

# a. 使用左合并（left merge），以 all_companies 为基准
results_df = pd.merge(all_companies, default_status_df, on='企业代号', how='left')

# b. 处理没有信贷记录的企业
#    来自'附件2'的企业在合并后，其'是否违约'和'违约情况V_Score'列会是 NaN。
#    我们将这些企业的分数填充为3分。业务逻辑是：没有信贷历史意味着风险未知，
#    从审慎角度出发，不能给予其最高的“无违约”分，因此赋予其与“有违约”相同的分数。
results_df['违约情况V_Score'] = results_df['违约情况V_Score'].fillna(3)
results_df['是否违约'] = results_df['是否违约'].fillna('无信贷记录')

# c. 转换分数列为整数类型
results_df['违约情况V_Score'] = results_df['违约情况V_Score'].astype(int)

# --- 5. Result Presentation ---
print("\n--- 企业违约情况 V 赋分结果 (部分展示) ---")
# 为了更好地展示，我们显示一些有违约、无违约和无信贷记录的混合样本
# E29 和 E36 是违约企业，E1 和 E2 是无违约企业
sample_defaults = results_df[results_df['企业代号'].isin(['E1', 'E2', 'E29', 'E36'])]
sample_no_history = results_df[results_df['企业代号'].isin([f'E{i}' for i in range(124, 127)])]
display_sample = pd.concat([sample_defaults, sample_no_history])

print(display_sample)

# 您可以取消下面的注释，查看所有结果或将结果保存到文件中
print("\n--- 全部结果 ---")
print(results_df)
output_filename = '企业违约情况V分数.xlsx'
results_df.to_excel(output_filename, index=False)
print(f"\n结果已保存至 '{output_filename}'")
