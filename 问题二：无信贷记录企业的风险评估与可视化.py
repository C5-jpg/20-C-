import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- 0. 全局设置 ---
# 忽略一些来自sklearn的警告
warnings.filterwarnings('ignore', category=UserWarning)
# 设置绘图风格和中文显示
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义违约概率 d(I) 函数
def calculate_default_probability(I):
    return np.sqrt(5 / np.pi) * np.exp(-10 * I ** 2)


# 定义信誉评级和违约情况的映射
rating_to_score_map = {'A': 10, 'B': 8, 'C': 5, 'D': 0}
score_to_rating_map = {v: k for k, v in rating_to_score_map.items()}
status_to_score_map = {'否': 9, '是': 3}

# --- 1. 加载所有已计算的指标数据 ---
print("--- 步骤 1: 加载所有预处理好的指标数据 ---")
try:
    # 加载附件1的完整指标数据 (用于训练)
    df_attach1_info = pd.read_excel('附件1：123家有信贷记录企业的相关数据.xlsx', sheet_name='企业信息')

    # 加载附件2的企业名单
    df_attach2_info = pd.read_excel('附件2：302家无信贷记录企业的相关数据.xlsx', sheet_name='企业信息')

    # 加载所有指标文件
    all_indicators = {}
    indicator_files = ['P', 'alpha', 'R_score', 'V_score', 'Bp', 'F', 'L']
    file_name_map = {
        'P': '企业总收益分析结果.xlsx', 'alpha': '企业进步因子分析结果.xlsx',
        'R_score': '企业信誉评级R分数.xlsx', 'V_score': '企业违约情况V分数.xlsx',
        'Bp': '企业无效发票比例分析.xlsx', 'F': '企业交易偏好F分析.xlsx',
        'L': '企业交易规律L分析.xlsx'
    }
    for indicator in indicator_files:
        all_indicators[indicator] = pd.read_excel(file_name_map[indicator])

    # 合并成一个大的DataFrame
    df_full = pd.concat([df_attach1_info[['企业代号']], df_attach2_info[['企业代号']]], ignore_index=True)
    for indicator, df in all_indicators.items():
        # 选择正确的列进行合并
        col_map = {
            'P': '总收益 (P)', 'alpha': '进步因子_alpha', 'R_score': '信誉评级R_Score',
            'V_score': '违约情况V_Score', 'Bp': '校正值_1_minus_Bp', 'F': '交易偏好_F', 'L': '交易规律_L'
        }
        df_to_merge = df[['企业代号', col_map[indicator]]]
        df_full = pd.merge(df_full, df_to_merge, on='企业代号', how='left')

    df_full.fillna(0, inplace=True)

    # 分离出附件1和附件2的数据
    df_train = pd.merge(df_attach1_info, df_full, on='企业代号', how='inner')
    df_predict = pd.merge(df_attach2_info, df_full, on='企业代号', how='inner')

    print("数据加载和准备完成。")

except FileNotFoundError as e:
    print(f"文件加载错误: {e}。请确保所有指标的Excel文件都存在。")
    exit()

# --- 2. 模拟BP神经网络：训练预测模型 ---
print("\n--- 步骤 2: 训练信誉和违约预测模型 (模拟BP神经网络) ---")
indicator_cols = ['总收益 (P)', '进步因子_alpha', '交易偏好_F', '交易规律_L', '校正值_1_minus_Bp']

X_train = df_train[indicator_cols].values

# a. 训练信誉评级预测模型
y_train_rating = df_train['信誉评级'].values
rating_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
rating_model.fit(X_train, y_train_rating)
print("信誉评级预测模型训练完成。")

# b. 训练是否违约预测模型
y_train_status = df_train['是否违约'].values
status_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
status_model.fit(X_train, y_train_status)
print("是否违约预测模型训练完成。")

# --- 3. 应用模型：预测附件2企业的数据 ---
print("\n--- 步骤 3: 应用模型，预测附件2企业的数据 ---")
X_predict = df_predict[indicator_cols].values

# a. 进行预测
predicted_ratings = rating_model.predict(X_predict)
predicted_statuses = status_model.predict(X_predict)

# b. 将预测结果添加到df_predict中
df_predict['预测信誉评级'] = predicted_ratings
df_predict['预测是否违约'] = predicted_statuses

# c. 根据预测结果计算R分数和V分数
df_predict['信誉评级R_Score'] = df_predict['预测信誉评级'].map(rating_to_score_map)
df_predict['违约情况V_Score'] = df_predict['预测是否违约'].map(status_to_score_map)
print("附件2企业的信誉评级和违约情况预测完成。")

# --- 4. 应用PCA模型，计算信贷风险安全指数 I ---
print("\n--- 步骤 4: 应用PCA模型，计算附件2企业的安全指数 ---")
# a. 准备完整的7个指标数据
full_indicator_cols = ['总收益 (P)', '进步因子_alpha', '交易偏好_F', '交易规律_L',
                       '信誉评级R_Score', '违约情况V_Score', '校正值_1_minus_Bp']
X_attach2 = df_predict[full_indicator_cols].values

# b. 使用在附件1数据上训练好的标准化和PCA模型
#    这里我们重新在完整数据上训练以保持一致性
X_full_for_pca = df_full[full_indicator_cols].values
scaler = StandardScaler().fit(X_full_for_pca)
X_attach2_scaled = scaler.transform(X_attach2)

# c. 使用您论文中提供的特征向量和贡献率
eigenvectors = np.array([
    [0.4525, 0.6055, 0.1088, 0.5910, 0.1450, 0.1072],
    [0.2262, 0.0943, 0.0709, 0.0789, 0.6698, 0.6754],
    [0.3712, 0.0040, 0.5846, 0.0361, 0.1402, 0.1982],
    [0.1594, 0.0562, 0.7918, 0.0506, 0.1020, 0.0131],
    [0.7514, 0.2599, 0.0747, 0.4599, 0.0490, 0.0172],
    [0.0317, 0.0176, 0.0929, 0.0238, 0.7052, 0.7016],
    [0.1237, 0.7439, 0.0116, 0.6547, 0.0256, 0.0189]
])
contribution_rates = np.array([0.3726, 0.2681, 0.1570, 0.1323, 0.0493, 0.0205])

# d. 计算综合得分并归一化
Y_attach2 = np.dot(X_attach2_scaled, eigenvectors)
I_tmp_attach2 = np.dot(Y_attach2, contribution_rates)
min_I, max_I = I_tmp_attach2.min(), I_tmp_attach2.max()
df_predict['信贷风险安全指数_I'] = (I_tmp_attach2 - min_I) / (max_I - min_I)
df_predict['违约概率_d'] = df_predict['信贷风险安全指数_I'].apply(calculate_default_probability)
print("附件2企业的安全指数 I 和违约概率 d 计算完成。")

# --- 5. 图表复现 ---
print("\n--- 步骤 5: 正在生成图10和图11 ---")

# a. 绘制图10：信贷风险安全指数分布
df_plot10 = df_predict.sort_values('信贷风险安全指数_I', ascending=False).reset_index(drop=True)
plt.figure(figsize=(12, 7))
scatter10 = plt.scatter(
    df_plot10.index,
    df_plot10['信贷风险安全指数_I'],
    c=df_plot10['信贷风险安全指数_I'],
    cmap='viridis_r',  # 使用反转的viridis色谱，使高分偏黄
    s=15
)
plt.colorbar(scatter10, label='信贷风险安全指数')
plt.title('图10：附件2中302家企业信贷风险安全指数分布概况', fontsize=16)
plt.xlabel('企业排序 (按安全指数从高到低)', fontsize=12)
plt.ylabel('信贷风险安全指数', fontsize=12)
plt.ylim(0, 1)
plt.xlim(0, 302)
plt.grid(True)
plt.savefig('图10_附件2安全指数分布.png', dpi=300)
print("图10 已保存。")

# b. 绘制图11：违约概率评估
df_plot11 = df_predict.sort_values('违约概率_d', ascending=True).reset_index(drop=True)
plt.figure(figsize=(12, 7))
scatter11 = plt.scatter(
    df_plot11.index,
    df_plot11['违约概率_d'],
    c=df_plot11['违约概率_d'],
    cmap='viridis',  # 使用viridis色谱，使低概率偏蓝
    s=15
)
plt.colorbar(scatter11, label='违约概率')
plt.title('图11：附件2中302家企业违约概率评估', fontsize=16)
plt.xlabel('企业排序 (按违约概率从低到高)', fontsize=12)
plt.ylabel('违约概率', fontsize=12)
plt.ylim(0, 1)
plt.xlim(0, 302)
plt.grid(True)
plt.savefig('图11_附件2违约概率评估.png', dpi=300)
print("图11 已保存。")

plt.show()
print("\n所有任务完成！")
