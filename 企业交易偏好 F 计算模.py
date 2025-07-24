import pandas as pd
import numpy as np
import os

# --- 0. File Path Configuration ---
# 详细注释：请在此处配置您的文件路径。
file_path_1 = r'D:\google-downloads\附件1：123家有信贷记录企业的相关数据.xlsx'
file_path_2 = r'D:\google-downloads\附件2：302家无信贷记录企业的相关数据.xlsx'

# --- 1. Data Loading ---
# 详细注释：加载所有需要的发票数据和企业名单。
try:
    info1 = pd.read_excel(file_path_1, sheet_name='企业信息')
    sales1 = pd.read_excel(file_path_1, sheet_name='销项发票信息')
    purchases1 = pd.read_excel(file_path_1, sheet_name='进项发票信息')

    info2 = pd.read_excel(file_path_2, sheet_name='企业信息')
    sales2 = pd.read_excel(file_path_2, sheet_name='销项发票信息')
    purchases2 = pd.read_excel(file_path_2, sheet_name='进项发票信息')
    print("所有Excel数据文件加载成功。")
except FileNotFoundError as e:
    print(f"文件加载错误: {e}。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"读取Excel文件时发生错误: {e}。")
    exit()

# --- 2. Data Consolidation and Preprocessing ---
# 详细注释：整合所有发票数据，并进行关键的预处理。
all_companies = pd.concat([info1[['企业代号']], info2[['企业代号']]], ignore_index=True)
all_invoices = pd.concat([
    sales1[['企业代号', '金额', '税额', '发票状态']],
    purchases1[['企业代号', '金额', '税额', '发票状态']],
    sales2[['企业代号', '金额', '税额', '发票状态']],
    purchases2[['企业代号', '金额', '税额', '发票状态']]
], ignore_index=True)

# a. 只保留有效发票，因为它们代表了真实的交易
valid_invoices = all_invoices[all_invoices['发票状态'] == '有效发票'].copy()

# b. 转换金额和税额为数值类型
valid_invoices['金额'] = pd.to_numeric(valid_invoices['金额'], errors='coerce')
valid_invoices['税额'] = pd.to_numeric(valid_invoices['税额'], errors='coerce')

# c. 移除计算税率所需数据不完整的行
valid_invoices.dropna(subset=['金额', '税额'], inplace=True)

# d. 过滤掉金额为0或负数的发票，因为税率计算无意义或会产生干扰
#   (负数发票是退货，不反映常规交易偏好)
valid_invoices = valid_invoices[valid_invoices['金额'] > 0]

print(f"数据预处理完成，总共有 {len(valid_invoices)} 条有效交易发票用于计算。")

# --- 3. Calculate Tax Rate for Each Invoice ---
# 详细注释：计算每张有效发票的实际税率。
valid_invoices['税率'] = valid_invoices['税额'] / valid_invoices['金额']

# **关键步骤**：处理税率的微小浮动偏差
# 由于浮点数精度问题，计算出的税率可能是 0.169999 或 0.030001。
# 我们需要将它们归一化到最接近的标准税率上。
def round_to_standard_rate(rate):
    standard_rates = [0.17, 0.16, 0.13, 0.11, 0.10, 0.09, 0.06, 0.05, 0.03, 0.01]
    # 找到与该税率最接近的标准税率
    closest_rate = min(standard_rates, key=lambda x: abs(x - rate))
    # 如果误差在一定范围内（例如0.5%），则认为是该标准税率
    if abs(closest_rate - rate) < 0.005:
        return closest_rate
    return rate # 如果误差过大，保留原始计算值

valid_invoices['税率_标准化'] = valid_invoices['税率'].apply(round_to_standard_rate)

print("每张发票的税率计算和标准化完成。")

# --- 4. Calculate Transaction Preference F ---
# 详细注释：这是本脚本的核心。我们使用高效的 groupby 操作来计算 F。
# F = Σ(t * pt) 的数学本质就是税率的均值。
# 因此，直接计算每个企业所有有效发票的'税率_标准化'的平均值，即可得到 F。
# 这是最高效的计算方法，避免了复杂的循环和比例计算。

f_series = valid_invoices.groupby('企业代号')['税率_标准化'].mean()

# --- 5. Assemble Final DataFrame and Save ---
# a. 将计算结果转换为 DataFrame
results_df = pd.DataFrame(f_series).reset_index()
results_df.rename(columns={'税率_标准化': '交易偏好_F'}, inplace=True)

# b. 将结果与完整的企业列表合并，确保所有企业都被包含
final_results = pd.merge(all_companies, results_df, on='企业代号', how='left')

# c. 对于没有任何有效交易发票的企业，其 F 值为空，我们填充为0
final_results['交易偏好_F'].fillna(0, inplace=True)

# --- 6. Result Presentation and Saving ---
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print("\n--- 企业交易偏好 F 计算结果 (部分展示) ---")
print(final_results.head(10))

# Save the final results to an Excel file
output_filename = '企业交易偏好F分析.xlsx'
try:
    final_results.to_excel(output_filename, index=False)
    print(f"\n计算结果已成功保存至: '{output_filename}'")
except Exception as e:
    print(f"\n保存文件时发生错误: {e}")

