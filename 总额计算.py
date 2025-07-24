import pandas as pd
import numpy as np
import os

# --- 0. 文件路径配置 (File Path Configuration) ---
# 详细注释：请在此处配置您的文件路径。使用'r'前缀可以防止路径中的反斜杠被错误地解析。

# 输入文件路径
file_path_1 = r'D:\WeChat\xwechat_files\wxid_5ex8kjms8fa022_408c\msg\file\2025-07\附件1：123家有信贷记录企业的相关数据.xlsx'
file_path_2 = r'D:\WeChat\xwechat_files\wxid_5ex8kjms8fa022_408c\msg\file\2025-07\附件2：302家无信贷记录企业的相关数据.xlsx'

# 输出文件夹路径
output_dir = r'D:\python-rep\dateclean-competiton'
output_filename = '企业总收益分析结果.xlsx'
output_filepath = os.path.join(output_dir, output_filename)

# 确保输出目录存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"输出目录 '{output_dir}' 不存在，已自动创建。")


# --- 1. 数据加载 (Data Loading) ---
# 详细注释：我们现在使用 pd.read_excel 直接从Excel文件中读取指定的工作表。

try:
    # 加载有信贷记录的企业数据
    info1 = pd.read_excel(file_path_1, sheet_name='企业信息')
    sales1 = pd.read_excel(file_path_1, sheet_name='销项发票信息')
    purchases1 = pd.read_excel(file_path_1, sheet_name='进项发票信息')

    # 加载无信贷记录的企业数据
    info2 = pd.read_excel(file_path_2, sheet_name='企业信息')
    sales2 = pd.read_excel(file_path_2, sheet_name='销项发票信息')
    purchases2 = pd.read_excel(file_path_2, sheet_name='进项发票信息')

    print("所有Excel数据文件加载成功。")

except FileNotFoundError as e:
    print(f"文件加载错误: {e}。请检查文件路径是否正确。")
    exit()
except Exception as e:
    print(f"读取Excel文件时发生错误: {e}。请确保文件未损坏且'openpyxl'库已安装。")
    exit()


# --- 2. 数据整合与预处理 (Data Consolidation & Preprocessing) ---

# 详细注释：为了对所有企业进行统一分析，我们将附件1和附件2中的同类数据进行合并。

# 合并企业信息
all_companies = pd.concat([info1[['企业代号']], info2[['企业代号']]], ignore_index=True)
# 合并销项发票信息
all_sales = pd.concat([sales1, sales2], ignore_index=True)
# 合并进项发票信息
all_purchases = pd.concat([purchases1, purchases2], ignore_index=True)

print(f"数据整合完成。总共有 {len(all_companies)} 家企业，{len(all_sales)} 条销项记录，{len(all_purchases)} 条进项记录。")

# **关键清洗步骤：筛选有效发票**
# 详细注释：这是整个分析中最关键的预处理步骤。我们只保留'发票状态'为'有效发票'的记录。
valid_sales = all_sales[all_sales['发票状态'] == '有效发票'].copy()
valid_purchases = all_purchases[all_purchases['发票状态'] == '有效发票'].copy()

print(f"筛选有效发票后，剩余 {len(valid_sales)} 条有效销项记录，{len(valid_purchases)} 条有效进项记录。")

# 数据类型转换，确保'金额'和'税额'为数值类型，便于计算
for df in [valid_sales, valid_purchases]:
    df['金额'] = pd.to_numeric(df['金额'], errors='coerce')
    df['税额'] = pd.to_numeric(df['税额'], errors='coerce')
    df.fillna({'金额': 0, '税额': 0}, inplace=True)


# --- 3. 按企业分组聚合计算 (Groupby & Aggregation) ---

# 详细注释：使用`groupby()`方法，高效计算每家企业的各项总和。

sales_agg = valid_sales.groupby('企业代号').agg(
    total_sales_amount=('金额', 'sum'),
    total_sales_tax=('税额', 'sum')
).reset_index()

purchases_agg = valid_purchases.groupby('企业代号').agg(
    total_purchase_amount=('金额', 'sum'),
    total_purchase_tax=('税额', 'sum')
).reset_index()

print("\n已按企业代号完成进项和销项数据的聚合计算。")


# --- 4. 合并数据并计算最终指标 (Merge & Final Calculation) ---

# 详细注释：将聚合后的数据合并到完整的企业列表中，并计算最终指标。
results_df = pd.merge(all_companies, sales_agg, on='企业代号', how='left')
results_df = pd.merge(results_df, purchases_agg, on='企业代号', how='left')
results_df.fillna(0, inplace=True)

# **核心公式计算**
results_df['vat_payable_T'] = results_df['total_sales_tax'] - results_df['total_purchase_tax']
results_df['vat_payable_T'] = results_df['vat_payable_T'].clip(lower=0)
results_df['total_profit_P'] = results_df['total_sales_amount'] - results_df['total_purchase_amount'] - results_df['vat_payable_T']


# --- 5. 结果展示与保存 (Result Presentation & Saving) ---

# 详细注释：为了清晰地展示和保存结果，我们重命名列名。
final_columns = {
    '企业代号': '企业代号',
    'total_sales_amount': '总销项金额 (ΣXa)',
    'total_purchase_amount': '总进项金额 (ΣJa)',
    'total_sales_tax': '总销项税额 (ΣXt)',
    'total_purchase_tax': '总进项税额 (ΣJt)',
    'vat_payable_T': '应缴增值税 (T)',
    'total_profit_P': '总收益 (P)'
}
results_df = results_df.rename(columns=final_columns)

# 设置pandas显示选项，以便更好地查看数字
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("\n--- 计算结果预览 ---")
print("已为所有企业计算出总收益 P。")
print(results_df[list(final_columns.values())].head(10))

# 将结果保存到您指定的Excel文件中
try:
    results_df.to_excel(output_filepath, index=False, sheet_name='企业总收益分析')
    print(f"\n计算结果已成功保存至: '{output_filepath}'")
except Exception as e:
    print(f"\n保存文件时发生错误: {e}")

