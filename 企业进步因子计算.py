import pandas as pd
import numpy as np
import time
import os

# --- 0. 计时器与文件路径配置 ---
start_time = time.time()

# 详细注释：请在此处配置您的文件路径。
# 使用 'r' 前缀可以防止路径中的反斜杠被错误地解析。
file_path_1 = r'D:\google-downloads\附件1：123家有信贷记录企业的相关数据.xlsx'
file_path_2 = r'D:\google-downloads\附件2：302家无信贷记录企业的相关数据.xlsx'

# --- 1. 数据加载 (Data Loading) ---
# 详细注释：我们现在使用 pd.read_excel 直接从Excel文件中读取指定的工作表。
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
    print(f"读取Excel文件时发生错误: {e}。请确保文件未损坏且'openpyxl'库已安装。")
    exit()

# --- 2. 数据整合与预处理 (Data Consolidation & Preprocessing) ---
# 详细注释：合并数据，并进行关键的预处理，为时间序列分析做准备。

all_companies = pd.concat([info1[['企业代号']], info2[['企业代号']]], ignore_index=True)
all_sales = pd.concat([sales1, sales2], ignore_index=True)
all_purchases = pd.concat([purchases1, purchases2], ignore_index=True)

# **关键预处理步骤**
invoices_dfs = {'sales': all_sales, 'purchases': all_purchases}
monthly_agg = {}

for name, df in invoices_dfs.items():
    # a. 筛选有效发票
    valid_df = df[df['发票状态'] == '有效发票'].copy()

    # b. 转换日期类型
    valid_df['开票日期'] = pd.to_datetime(valid_df['开票日期'], errors='coerce')
    valid_df.dropna(subset=['开票日期'], inplace=True)

    # c. 创建 '年月' 列，用于按月分组
    valid_df['年月'] = valid_df['开票日期'].dt.to_period('M')

    # d. 转换金额和税额为数值类型
    valid_df['金额'] = pd.to_numeric(valid_df['金额'], errors='coerce').fillna(0)
    valid_df['税额'] = pd.to_numeric(valid_df['税额'], errors='coerce').fillna(0)

    # e. 按企业和年月进行分组聚合，计算月度总额
    agg_df = valid_df.groupby(['企业代号', '年月']).agg(
        amount=('金额', 'sum'),
        tax=('税额', 'sum')
    ).reset_index()
    monthly_agg[name] = agg_df

print("数据预处理和月度聚合完成。")

# --- 3. 构建完整的月度收益时间序列 (Build Monthly Profit Timeseries) ---

# a. 合并聚合后的月度销项和进项数据
merged_df = pd.merge(
    monthly_agg['sales'],
    monthly_agg['purchases'],
    on=['企业代号', '年月'],
    how='outer',
    suffixes=('_sales', '_purchase')
)

# b. 填充缺失值
merged_df.fillna(0, inplace=True)

# c. 计算每个月的月度收益 P_month
vat_payable_monthly = merged_df['tax_sales'] - merged_df['tax_purchase']
merged_df['vat_payable_monthly'] = vat_payable_monthly.clip(lower=0)
merged_df['profit_monthly'] = (merged_df['amount_sales'] - merged_df['amount_purchase']) - merged_df[
    'vat_payable_monthly']

print("月度收益 P_month 计算完成。")

# --- 4. 计算进步因子 α (Calculate Progress Factor Alpha) ---
# 详细注释：这是本脚本的核心计算部分。我们将使用高效的 pandas 操作来计算增长率。

# a. 排序并设置索引
data = merged_df.sort_values(['企业代号', '年月']).set_index(['企业代号', '年月'])

# b. 计算月度收益增长率 I_r
p0 = data.groupby('企业代号')['profit_monthly'].shift(1)
p1 = data['profit_monthly']

denominator = p0.abs()
growth_rate = np.divide(p1 - p0, denominator, out=np.full_like(p1, np.nan), where=denominator != 0)

data['growth_rate'] = growth_rate

# c. 计算每个企业的进步因子 α
alpha_series = data.groupby('企业代号')['growth_rate'].mean()

# d. 将 α 合并到最终结果中
results_df = pd.DataFrame(alpha_series).reset_index()
results_df.rename(columns={'growth_rate': '进步因子_alpha'}, inplace=True)
results_df.fillna({'进步因子_alpha': 0}, inplace=True)

print("进步因子 α 计算完成。")

# --- 5. 结果展示 (Result Presentation) ---
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("\n--- 企业进步因子 α 计算结果 (前10家) ---")
print(results_df.head(10))

# 您可以取消下面的注释，将结果保存到文件中
output_filename = '企业进步因子分析结果.xlsx'
results_df.to_excel(output_filename, index=False)
print(f"\n结果已保存至 '{output_filename}'")

end_time = time.time()
print(f"\n脚本总运行时间: {end_time - start_time:.2f} 秒")
