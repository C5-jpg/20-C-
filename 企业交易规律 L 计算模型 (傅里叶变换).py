import pandas as pd
import numpy as np
import os
from scipy.fft import fft  # Using scipy's FFT is a modern and robust choice

# --- 0. File Path and Dependency Check ---
# 详细注释：请在此处配置您的文件路径。
file_path_1 = r'D:\google-downloads\附件1：123家有信贷记录企业的相关数据.xlsx'
file_path_2 = r'D:\google-downloads\附件2：302家无信贷记录企业的相关数据.xlsx'

# --- 1. Data Loading ---
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
all_companies = pd.concat([info1[['企业代号']], info2[['企业代号']]], ignore_index=True)

# a. Unify sales and purchase invoices into a single transaction log
sales1['交易伙伴代号'] = sales1['购方单位代号']
sales2['交易伙伴代号'] = sales2['购方单位代号']
purchases1['交易伙伴代号'] = purchases1['销方单位代号']
purchases2['交易伙伴代号'] = purchases2['销方单位代号']

all_invoices = pd.concat([sales1, sales2, purchases1, purchases2], ignore_index=True)

# b. Filter for valid transactions only
valid_invoices = all_invoices[all_invoices['发票状态'] == '有效发票'].copy()

# c. Calculate the specific 'transaction_value' as defined: 金额 - 税额
valid_invoices['金额'] = pd.to_numeric(valid_invoices['金额'], errors='coerce')
valid_invoices['税额'] = pd.to_numeric(valid_invoices['税额'], errors='coerce')
valid_invoices.dropna(subset=['金额', '税额'], inplace=True)
valid_invoices['交易价值'] = valid_invoices['金额'] - valid_invoices['税额']

# d. Sort by date to create a time series for each partner
valid_invoices['开票日期'] = pd.to_datetime(valid_invoices['开票日期'], errors='coerce')
valid_invoices.dropna(subset=['开票日期', '交易伙伴代号'], inplace=True)
valid_invoices = valid_invoices.sort_values('开票日期')

print(f"数据预处理完成，总共有 {len(valid_invoices)} 条有效交易发票用于计算。")


# --- 3. Define the Core Calculation Function ---
# 详细注释：这是本脚本的核心。此函数对一组交易数据执行傅里叶变换并计算S^2。
def calculate_s_squared(transactions):
    """
    Calculates the variance of the FFT amplitude spectrum for a series of transactions.
    :param transactions: A pandas Series of transaction values.
    :return: The variance S^2.
    """
    # B is the number of transactions
    B = len(transactions)

    # If there's only one transaction, its regularity cannot be measured, variance is 0.
    if B <= 1:
        return 0.0

    # Perform Fast Fourier Transform (FFT)
    fft_values = fft(transactions.values)

    # Calculate the amplitude spectrum (abs in the formula)
    amplitude_spectrum = np.abs(fft_values)

    # Calculate the variance of the amplitude spectrum (S^2)
    # np.var calculates the population variance by default (ddof=0), which matches the formula.
    s_squared = np.var(amplitude_spectrum)

    return s_squared


# --- 4. Apply the Function and Calculate L ---
# 详细注释：使用 groupby().apply() 是在 pandas 中执行此类复杂分组计算的最高效方法。

print("开始计算每个企业与其交易伙伴的 S^2...")

# a. Group by target company and their trading partner
#    Then apply our function to the '交易价值' series of each group.
s_squared_series = valid_invoices.groupby(['企业代号', '交易伙伴代号'])['交易价值'].apply(calculate_s_squared)

print("S^2 计算完成。开始计算最终的交易规律 L...")

# b. Calculate L by taking the mean of S^2 for each target company
#    The result of the apply is a Series with a MultiIndex (企业代号, 交易伙伴代号).
#    We can group it by the first level of the index ('企业代号') and calculate the mean.
l_series = s_squared_series.groupby('企业代号').mean()

# --- 5. Assemble Final DataFrame and Save ---
# a. Convert the final L series to a DataFrame
results_df = pd.DataFrame(l_series).reset_index()
results_df.rename(columns={'交易价值': '交易规律_L'}, inplace=True)

# b. Merge with the full list of companies
final_results = pd.merge(all_companies, results_df, on='企业代号', how='left')

# c. For companies with no valid transactions, L is undefined. Fill with 0.
#    A higher L means less regularity (higher risk), so 0 is a safe baseline.
final_results['交易规律_L'].fillna(0, inplace=True)

# --- 6. Result Presentation and Saving ---
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print("\n--- 企业交易规律 L 计算结果 (部分展示) ---")
print(final_results.head(10))

# Save the final results to an Excel file
output_filename = '企业交易规律L分析.xlsx'
try:
    final_results.to_excel(output_filename, index=False)
    print(f"\n计算结果已成功保存至: '{output_filename}'")
except Exception as e:
    print(f"\n保存文件时发生错误: {e}")

