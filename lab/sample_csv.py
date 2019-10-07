import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_normal = pd.read_csv('normal_loglikelihoods.csv')
df_svi = pd.read_csv('svi_loglikelihoods.csv')
df_stochastic = pd.read_csv('stochastic_loglikelihoods.csv')
# 列ごとの総和
normal_sum_columns = df_normal.sum(axis=0)
svi_sum_columns = df_svi.sum(axis=0)
stochastic_sum_columns = df_stochastic.sum(axis=0)
# 行数
rows_num = len(df_normal) + 1
print("行数: ", rows_num)

# 平均計算
mean_normal_log_likelihoods = normal_sum_columns / rows_num
mean_svi_log_likelihoods = svi_sum_columns / rows_num
mean_stochastic_log_likelihoods = stochastic_sum_columns / rows_num

# numpy配列へ変換
mean_normal_log_likelihoods = np.array(mean_normal_log_likelihoods)
mean_svi_log_likelihoods = np.array(mean_svi_log_likelihoods)
mean_stochastic_log_likelihoods = np.array(mean_stochastic_log_likelihoods)
plt.figure()
plt.plot(mean_normal_log_likelihoods, color='#2971e5', linestyle='solid')
plt.plot(mean_svi_log_likelihoods, color='#ffff00', linestyle='solid')
plt.plot(mean_stochastic_log_likelihoods, color='#ed3b3b', linestyle='solid')
plt.legend(['VI', 'SGD+VI', 'SEM+VI'])
plt.show()