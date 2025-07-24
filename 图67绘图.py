import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. 环境与绘图风格设置 ---
# 设置一个美观的绘图风格
sns.set_theme(style="whitegrid")
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 1. 数据加载与准备 ---
print("--- 步骤 1: 加载客户流失率数据 ---")
try:
    # 修正后的文件路径，直接指向您的 .xlsx 文件
    file_path_3 = r'D:\google-downloads\附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx'

    # 使用 pd.read_excel 读取，并指定 header=1 来将第二行作为表头
    df_churn = pd.read_excel(file_path_3, header=1)

    # 重命名列以方便使用
    df_churn.rename(columns={
        'Unnamed: 0': '贷款年利率',
        '信誉评级A': '流失率_A',
        '信誉评级B': '流失率_B',
        '信誉评级C': '流失率_C'
    }, inplace=True)

    print("Excel数据加载成功！数据预览：")
    print(df_churn.head())

except FileNotFoundError:
    print(f"文件加载错误: '{file_path_3}' 未找到。请确保文件路径和名称完全正确。")
    exit()
except Exception as e:
    print(f"读取Excel文件时发生错误: {e}")
    exit()

# --- 2. 绘制图6：贷款年利率与客户流失率关系图 ---
print("\n--- 步骤 2: 正在生成图6 ---")
plt.figure(figsize=(12, 7))

plt.plot(df_churn['贷款年利率'], df_churn['流失率_A'], marker='o', linestyle='-', label='客户流失率 (信誉评级A)',
         markersize=4)
plt.plot(df_churn['贷款年利率'], df_churn['流失率_B'], marker='s', linestyle='-', label='客户流失率 (信誉评级B)',
         markersize=4)
plt.plot(df_churn['贷款年利率'], df_churn['流失率_C'], marker='^', linestyle='-', label='客户流失率 (信誉评级C)',
         markersize=4)

plt.title('图6：贷款年利率与客户流失率关系图', fontsize=16)
plt.xlabel('贷款年利率', fontsize=12)
plt.ylabel('客户流失率', fontsize=12)
plt.xlim(0, 0.16)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

output_fig6_path = '图6_贷款年利率与客户流失率关系图.png'
plt.savefig(output_fig6_path, dpi=300)
print(f"图6 已保存至 '{output_fig6_path}'")

# --- 3. 定义回归模型函数 ---
print("\n--- 步骤 3: 定义用于图7的回归模型 ---")


def model_cubic(r):
    return 640.944 * r ** 3 - 258.570 * r ** 2 + 37.970 * r - 1.121


def model_quadratic(r):
    return -76.410 * r ** 2 + 21.984 * r - 0.697


def model_linear(r):
    return 7.524 * r - 0.098


def model_logarithmic(r):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 0.669 * np.log(r) + 2.239
    return result


# --- 4. 绘制图7：客户流失率（信誉评级A）与贷款年利率关系拟合曲线 ---
print("\n--- 步骤 4: 正在生成图7 ---")
plt.figure(figsize=(12, 7))

plt.scatter(df_churn['贷款年利率'], df_churn['流失率_A'], label='实测数据点', color='black', s=20, zorder=5)

r_smooth = np.linspace(df_churn['贷款年利率'].min(), df_churn['贷款年利率'].max(), 400)

plt.plot(r_smooth, model_linear(r_smooth), ':', label='线性拟合', linewidth=2)
plt.plot(r_smooth, model_logarithmic(r_smooth), '-.', label='对数拟合', linewidth=2)
plt.plot(r_smooth, model_quadratic(r_smooth), '--', label='二次拟合', linewidth=2)
plt.plot(r_smooth, model_cubic(r_smooth), '-', label='三次拟合 (最优)', linewidth=2.5, color='red')

plt.title('图7：客户流失率(信誉评级A)与贷款年利率关系拟合曲线', fontsize=16)
plt.xlabel('贷款年利率', fontsize=12)
plt.ylabel('客户流失率', fontsize=12)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)

output_fig7_path = '图7_客户流失率A的拟合曲线图.png'
plt.savefig(output_fig7_path, dpi=300)
print(f"图7 已保存至 '{output_fig7_path}'")
plt.show()

print("\n所有任务完成！")
