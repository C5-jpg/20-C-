import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. 全局设置与函数定义 ---
# 设置绘图风格和中文显示
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义客户流失率 l_r(R, r) 函数
def calculate_churn_rate(R_score, r):
    # 将信誉评级分数映射回 A, B, C, D
    if R_score == 10:  # A
        lr = 640.944 * r ** 3 - 258.570 * r ** 2 + 37.970 * r - 1.121
    elif R_score == 8:  # B
        lr = 552.829 * r ** 3 - 225.051 * r ** 2 + 33.995 * r - 1.017
    elif R_score == 5:  # C
        lr = 504.717 * r ** 3 - 207.386 * r ** 2 + 32.157 * r - 0.973
    else:  # D 或无评级
        lr = 1.0
    # 确保流失率在 [0, 1] 区间
    return np.clip(lr, 0, 1)


# 定义违约概率 d(I) 函数
def calculate_default_probability(I):
    return np.sqrt(5 / np.pi) * np.exp(-10 * I ** 2)


# --- 1. 数据加载与整合 ---
print("--- 步骤 1: 加载并整合所有指标数据 ---")
try:
    # 定义所有指标文件的路径
    file_paths = {
        'info': '附件1：123家有信贷记录企业的相关数据.xlsx',
        'P': '企业总收益分析结果.xlsx',
        'alpha': '企业进步因子分析结果.xlsx',
        'R_score': '企业信誉评级R分数.xlsx',
        'V_score': '企业违约情况V分数.xlsx',
        'Bp': '企业无效发票比例分析.xlsx',
        'F': '企业交易偏好F分析.xlsx',
        'L': '企业交易规律L分析.xlsx'
    }

    # 读取数据
    df_info = pd.read_excel(file_paths['info'], sheet_name='企业信息')
    df_p = pd.read_excel(file_paths['P'])
    df_alpha = pd.read_excel(file_paths['alpha'])
    df_r = pd.read_excel(file_paths['R_score'])
    df_v = pd.read_excel(file_paths['V_score'])
    df_bp = pd.read_excel(file_paths['Bp'])
    df_f = pd.read_excel(file_paths['F'])
    df_l = pd.read_excel(file_paths['L'])

    # **修正后的合并逻辑**
    # 从包含'信誉评级'的df_info开始
    final_df = df_info[['企业代号', '信誉评级']]

    # 定义要合并的数据框和它们的关键列，避免引入重复的'信誉评级'列
    dfs_to_merge_info = [
        (df_p, ['企业代号', '总收益 (P)']),
        (df_alpha, ['企业代号', '进步因子_alpha']),
        (df_r, ['企业代号', '信誉评级R_Score']),  # **关键修改：只取分数，不取重复的评级列**
        (df_v, ['企业代号', '违约情况V_Score']),
        (df_bp, ['企业代号', '校正值_1_minus_Bp']),
        (df_f, ['企业代号', '交易偏好_F']),
        (df_l, ['企业代号', '交易规律_L'])
    ]

    # 循环进行精确合并
    for df, cols in dfs_to_merge_info:
        final_df = pd.merge(final_df, df[cols], on='企业代号', how='left')

    final_df.fillna(0, inplace=True)
    print("所有指标数据成功加载并合并！")

except FileNotFoundError as e:
    print(f"文件加载错误: {e}。请确保所有指标的Excel文件都存在。")
    exit()

# --- 2. 主成分分析 (PCA) 与信贷风险安全指数 I 计算 ---
print("\n--- 步骤 2: 执行主成分分析 (PCA) ---")
indicator_cols = ['总收益 (P)', '进步因子_alpha', '交易偏好_F', '交易规律_L',
                  '信誉评级R_Score', '违约情况V_Score', '校正值_1_minus_Bp']
X = final_df[indicator_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用您论文中提供的特征向量和贡献率
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

# 计算6个主成分得分 y1, ..., y6
Y = np.dot(X_scaled, eigenvectors)

# 计算综合得分 I_tmp
I_tmp = np.dot(Y, contribution_rates)
final_df['综合得分_Itmp'] = I_tmp

# 归一化得到信贷风险安全指数 I
min_I, max_I = I_tmp.min(), I_tmp.max()
final_df['信贷风险安全指数_I'] = (I_tmp - min_I) / (max_I - min_I)
final_df['违约概率_d'] = final_df['信贷风险安全指数_I'].apply(calculate_default_probability)
print("信贷风险安全指数 I 和违约概率 d 计算完成。")

# --- 3. 图表复现 (图8) ---
print("\n--- 步骤 3: 正在生成图8 ---")
plt.figure(figsize=(12, 7))
ratings = ['A', 'B', 'C', 'D']
markers = ['o', '*', '+', '.']
linestyles = ['-', '-', '-', '--']

for rating, marker, ls in zip(ratings, markers, linestyles):
    subset = final_df[final_df['信誉评级'] == rating]
    # 使用 reset_index() 来获得一个简单的整数索引用于绘图
    plt.plot(subset.index, subset['信贷风险安全指数_I'], marker=marker, linestyle=ls, label=f'{rating}级', alpha=0.7)

plt.title('图8：不同信誉评级企业的信贷风险安全指数统计图', fontsize=16)
plt.xlabel('企业排序（按原始顺序）', fontsize=12)
plt.ylabel('信贷风险安全指数', fontsize=12)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.savefig('图8_信贷风险安全指数统计图.png', dpi=300)
print("图8 已保存。")
# plt.show()

# --- 4. 遗传算法 (GA) 求解最优信贷策略 ---
print("\n--- 步骤 4: 正在使用遗传算法求解最优信贷策略 ---")

# a. 准备GA所需的数据 (只对附件1的123家企业进行决策)
ga_data = final_df.iloc[:123].copy().reset_index(drop=True)  # 使用 reset_index 确保 loc[i] 能正常工作
num_companies = len(ga_data)

# b. GA 参数定义
TOTAL_LOAN_AMOUNT = 1e8  # 1亿元
POPULATION_SIZE = 100
GENERATIONS = 300
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# 离散决策空间
LOAN_AMOUNTS = np.arange(0, 101, 1) * 1e4  # 0到100万，步长1万
INTEREST_RATES = np.arange(0.04, 0.1505, 0.0001)  # 4%到15%，步长0.01%


# c. 适应度函数
def calculate_fitness(chromosome):
    total_profit = 0
    total_loan = 0
    for i in range(num_companies):
        a = chromosome[i][0]
        r = chromosome[i][1]

        # 如果信誉评级为D，不予贷款
        if ga_data.loc[i, '信誉评级'] == 'D':
            a = 0

        d = ga_data.loc[i, '违约概率_d']
        lr = calculate_churn_rate(ga_data.loc[i, '信誉评级R_Score'], r)

        profit_i = (a * r * (1 - d) - a * d) * (1 - lr)
        total_profit += profit_i
        total_loan += a

    # 惩罚项：如果总贷款额超出，则适应度为0
    if total_loan > TOTAL_LOAN_AMOUNT:
        return 0

    return total_profit


# d. GA 主流程
# 初始化种群
population = []
for _ in range(POPULATION_SIZE):
    chromosome = []
    current_total_loan = 0
    for i in range(num_companies):
        # 确保初始种群满足约束
        remaining_budget = TOTAL_LOAN_AMOUNT - current_total_loan
        max_possible_amount = min(remaining_budget, LOAN_AMOUNTS.max())

        possible_amounts = LOAN_AMOUNTS[LOAN_AMOUNTS <= max_possible_amount]
        a = random.choice(possible_amounts) if len(possible_amounts) > 0 else 0

        r = random.choice(INTEREST_RATES)
        chromosome.append([a, r])
        current_total_loan += a
    population.append(chromosome)

# 迭代进化
best_fitness_history = []
print("开始进化...")
for gen in range(GENERATIONS):
    fitness_scores = [calculate_fitness(c) for c in population]

    if sum(fitness_scores) == 0: continue

    selection_probs = [score / sum(fitness_scores) for score in fitness_scores]
    new_population = []

    # 保留最优个体
    best_idx = np.argmax(fitness_scores)
    new_population.append(population[best_idx])

    # 生成新个体
    while len(new_population) < POPULATION_SIZE:
        parents_indices = np.random.choice(len(population), size=2, p=selection_probs)
        parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]

        if random.random() < CROSSOVER_RATE:
            crossover_point = random.randint(1, num_companies - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
        else:
            child1 = parent1

        if random.random() < MUTATION_RATE:
            gene_to_mutate = random.randint(0, num_companies - 1)
            child1[gene_to_mutate][0] = random.choice(LOAN_AMOUNTS)
            child1[gene_to_mutate][1] = random.choice(INTEREST_RATES)

        new_population.append(child1)

    population = new_population

    best_fitness = max(fitness_scores)
    best_fitness_history.append(best_fitness)
    if (gen + 1) % 50 == 0:
        print(f"第 {gen + 1}/{GENERATIONS} 代, 当前最优利润: {best_fitness / 1e4:.2f} 万元")

# e. 获取并展示最终结果
final_fitness_scores = [calculate_fitness(c) for c in population]
best_chromosome_index = np.argmax(final_fitness_scores)
best_strategy = population[best_chromosome_index]
best_profit = final_fitness_scores[best_chromosome_index]

ga_data['最优贷款额度/万元'] = [s[0] / 1e4 for s in best_strategy]
ga_data['最优贷款利率/%'] = [s[1] * 100 for s in best_strategy]

# 根据信誉评级为D的不贷款原则进行修正
ga_data.loc[ga_data['信誉评级'] == 'D', ['最优贷款额度/万元', '最优贷款利率/%']] = 0

print(f"\n遗传算法求解完成！银行年度最大利润为: {best_profit / 1e4:.2f} 万元")
print("部分企业的最优信贷策略展示：")
print(ga_data[['企业代号', '信誉评级', '信贷风险安全指数_I', '最优贷款额度/万元', '最优贷款利率/%']].head(8))
