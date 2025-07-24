import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. 环境与绘图风格设置 ---
# 设置一个美观的绘图风格
sns.set_theme(style="whitegrid")
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 1. 数据加载与整合 ---
# 详细注释：这是模型的第一步。我们假设之前计算出的6个指标文件都已生成。
# 我们将把这些文件的数据加载进来，并基于'企业代号'将它们合并成一个宽表。

print("--- 步骤 1: 加载并整合所有指标数据 ---")
try:
    # 定义所有指标文件的路径
    # 注意：请确保这些文件名与您之前脚本生成的文件名一致
    file_paths = {
        'P': '企业总收益分析结果.xlsx',
        'alpha': '企业进步因子分析结果.xlsx',
        'R': '企业信誉评级R分数.xlsx',
        'V': '企业违约情况V分数.xlsx',
        'Bp': '企业无效发票比例分析.xlsx',
        'F': '企业交易偏好F分析.xlsx',
        'L': '企业交易规律L分析.xlsx'
    }

    # 读取所有数据文件
    data_frames = {name: pd.read_excel(path) for name, path in file_paths.items()}

    # 从一个基础DataFrame开始（例如总收益P），逐一合并其他指标
    # 使用 'outer' 合并可以确保即使某个文件缺少某个企业，该企业也不会被丢弃
    final_df = data_frames['P'][['企业代号', '总收益 (P)']]

    # 整理要合并的列
    merge_map = {
        'alpha': ['企业代号', '进步因子_alpha'],
        'R': ['企业代号', '信誉评级R_Score'],
        'V': ['企业代号', '违约情况V_Score'],
        'Bp': ['企业代号', '校正值_1_minus_Bp'],
        'F': ['企业代号', '交易偏好_F'],
        'L': ['企业代号', '交易规律_L']
    }

    for name, cols in merge_map.items():
        final_df = pd.merge(final_df, data_frames[name][cols], on='企业代号', how='outer')

    # 填充可能因合并产生的缺失值，这里用0填充是合理的
    final_df.fillna(0, inplace=True)

    print("所有指标数据成功加载并合并！")
    print("合并后的数据预览：")
    print(final_df.head())

except FileNotFoundError as e:
    print(f"文件加载错误: {e}。请确保所有指标的Excel文件都存在于项目目录中。")
    exit()

# --- 2. 主成分分析 (PCA) ---
print("\n--- 步骤 2: 执行主成分分析 (PCA) ---")

# a. 准备用于PCA的7个指标数据
#    注意：我们使用的是 Bp 的校正值 (1 - Bp)
indicator_columns = [
    '总收益 (P)', '进步因子_alpha', '交易偏好_F',
    '交易规律_L', '信誉评级R_Score', '违约情况V_Score', '校正值_1_minus_Bp'
]
X = final_df[indicator_columns].values

# b. (1) 对原始数据进行标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("数据标准化完成。")

# c. (2, 3) 计算相关系数矩阵、特征值和特征向量
#    我们选择主成分的数量，使其累计贡献率超过一个阈值，例如90%
pca = PCA(n_components=0.90)
Y = pca.fit_transform(X_scaled)  # Y 就是降维后的主成分 y1, y2, ...

# d. (4) 计算信息贡献率和累积贡献率
eigenvalues = pca.explained_variance_
contribution_rates = pca.explained_variance_ratio_
cumulative_contribution = np.cumsum(contribution_rates)

print(f"PCA分析完成。选择了 {pca.n_components_} 个主成分。")
print(f"各主成分的贡献率 (b_j): {contribution_rates}")
print(f"累计贡献率: {cumulative_contribution}")

# e. (5) 计算综合得分 I_tmp
#    I_tmp = Σ(b_j * y_j)
#    这里的 b_j 是贡献率，y_j 是对应的主成分得分
I_tmp = np.dot(Y, contribution_rates)
final_df['综合得分_Itmp'] = I_tmp
print("综合得分 I_tmp 计算完成。")

# f. (6) 对综合得分进行归一化处理，得到信贷风险安全指数 I
min_I_tmp = final_df['综合得分_Itmp'].min()
max_I_tmp = final_df['综合得分_Itmp'].max()
final_df['信贷风险安全指数_I'] = (final_df['综合得分_Itmp'] - min_I_tmp) / (max_I_tmp - min_I_tmp)
print("信贷风险安全指数 I 计算完成 (已归一化到 [0, 1] 区间)。")

# --- 3. 最优信贷策略决策模型 ---
print("\n--- 步骤 3: 计算企业违约概率 ---")


# a. (1) 定义并应用违约概率函数 d(I)
def calculate_default_probability(I):
    """根据公式 (11) 计算违约概率 d"""
    return np.sqrt(5 / np.pi) * np.exp(-10 * I ** 2)


final_df['违约概率_d'] = final_df['信贷风险安全指数_I'].apply(calculate_default_probability)
print("企业违约概率 d 计算完成。")

print("\n最终结果预览：")
print(final_df[['企业代号', '信贷风险安全指数_I', '违约概率_d']].head())

# --- 4. 可视化：复现函数关系图 ---
print("\n--- 步骤 4: 生成函数关系图 ---")

plt.figure(figsize=(10, 6))

# 为了绘制平滑的曲线，我们创建一个从0到1的密集点集
I_smooth = np.linspace(0, 1, 400)
d_smooth = calculate_default_probability(I_smooth)

# 绘制拟合的函数曲线
plt.plot(I_smooth, d_smooth, label='拟合函数 $d(I) = \\sqrt{5/\\pi} \\exp(-10I^2)$', color='tab:blue', linewidth=2)

# 设置图表标题和坐标轴标签
plt.title('信贷风险安全指数与违约概率的函数关系图', fontsize=16)
plt.xlabel('信贷风险安全指数 (I)', fontsize=12)
plt.ylabel('违约概率 (d)', fontsize=12)

# 设置坐标轴范围，与您的示例图保持一致
plt.xlim(0, 1)
plt.ylim(0, 1)

# 添加网格线并显示图例
plt.grid(True)
plt.legend()

# 保存并显示图像
output_image_path = '信贷风险安全指数与违约概率关系图.png'
plt.savefig(output_image_path, dpi=300)
print(f"图像已保存至 '{output_image_path}'")
plt.show()

