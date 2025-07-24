import pandas as pd
import os

# --- 0. File Path Configuration ---
# 详细注释：请在此处配置您的文件路径。
file_path_1 = r'D:\google-downloads\附件1：123家有信贷记录企业的相关数据.xlsx'
file_path_2 = r'D:\google-downloads\附件2：302家无信贷记录企业的相关数据.xlsx'

# --- 1. Data Loading ---
# 详细注释：加载所有需要的发票数据和企业名单。
try:
    # Load data from Attachment 1
    info1 = pd.read_excel(file_path_1, sheet_name='企业信息')
    sales1 = pd.read_excel(file_path_1, sheet_name='销项发票信息')
    purchases1 = pd.read_excel(file_path_1, sheet_name='进项发票信息')

    # Load data from Attachment 2
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

# --- 2. Data Consolidation ---
# 详细注释：将所有发票数据合并到一个DataFrame中，以便进行统一计算。
all_companies = pd.concat([info1[['企业代号']], info2[['企业代号']]], ignore_index=True)
all_sales = pd.concat([sales1, sales2], ignore_index=True)
all_purchases = pd.concat([purchases1, purchases2], ignore_index=True)

# Combine all invoices to get the total count (N)
all_invoices = pd.concat([all_sales, all_purchases], ignore_index=True)
print(f"数据整合完成，总共有 {len(all_invoices)} 条发票记录。")

# --- 3. Calculate Component Counts ---
# 详细注释：使用高效的 groupby 操作分别计算 N, n1, 和 n2。

# a. 计算总发票数 (N)
total_invoices_N = all_invoices.groupby('企业代号').size().reset_index(name='总发票数_N')

# b. 计算作废发票数 (n1)
#    筛选出所有'作废发票'，然后按企业分组计数。
voided_invoices_n1 = all_invoices[all_invoices['发票状态'] == '作废发票'].groupby('企业代号').size().reset_index(name='作废发票数_n1')

# c. 计算销项负数发票数 (n2)
#    首先确保'金额'列是数值类型，然后筛选出销项发票中金额为负的记录。
all_sales['金额'] = pd.to_numeric(all_sales['金额'], errors='coerce')
negative_sales_n2 = all_sales[all_sales['金额'] < 0].groupby('企业代号').size().reset_index(name='销项负数发票数_n2')

print("各组成部分的计数完成。")

# --- 4. Merge and Calculate Ratios ---
# 详细注释：将计算出的各项计数合并，并计算相应的比例。

# a. 从完整的企业列表开始，逐一合并计数结果
results_df = pd.merge(all_companies, total_invoices_N, on='企业代号', how='left')
results_df = pd.merge(results_df, voided_invoices_n1, on='企业代号', how='left')
results_df = pd.merge(results_df, negative_sales_n2, on='企业代号', how='left')

# b. 填充NaN为0。如果一家企业没有作废或负数发票，其计数应为0。
results_df.fillna(0, inplace=True)

# c. 将计数值转换为整数类型
for col in ['总发票数_N', '作废发票数_n1', '销项负数发票数_n2']:
    results_df[col] = results_df[col].astype(int)

# d. 计算比例 p1 和 p2，并处理总发票数为0的情况以避免除零错误。
results_df['作废发票占比_p1'] = results_df.apply(
    lambda row: row['作废发票数_n1'] / row['总发票数_N'] if row['总发票数_N'] > 0 else 0,
    axis=1
)
results_df['销项负数发票占比_p2'] = results_df.apply(
    lambda row: row['销项负数发票数_n2'] / row['总发票数_N'] if row['总发票数_N'] > 0 else 0,
    axis=1
)

# --- 5. Calculate Final Metrics and Save ---
# 详细注释：根据公式计算最终的 Bp 指标和校正值。

# a. 计算无效发票比例 Bp = 0.3*p1 + 0.7*p2
results_df['无效发票比例_Bp'] = 0.3 * results_df['作废发票占比_p1'] + 0.7 * results_df['销项负数发票占比_p2']

# b. 计算校正值 (1 - Bp)
results_df['校正值_1_minus_Bp'] = 1 - results_df['无效发票比例_Bp']

# --- 6. Result Presentation and Saving ---
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print("\n--- 企业无效发票比例 Bp 计算结果 (部分展示) ---")
print(results_df.head(10))

# Save the final results to an Excel file
output_filename = '企业无效发票比例分析.xlsx'
try:
    results_df.to_excel(output_filename, index=False)
    print(f"\n计算结果已成功保存至: '{output_filename}'")
except Exception as e:
    print(f"\n保存文件时发生错误: {e}")

