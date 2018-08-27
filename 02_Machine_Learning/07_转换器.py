# fit获取【训练数据】的均值和方差, 为后面的transform做准备，
# transform测试和训练数据都能用, 使用均值和方差进行无量化
# fit_transform 仅限训练数据使用
# 测试数据的均值和方差用fit获取，然后在用transform进行无量化
# 先划分，再特征工程

from sklearn.preprocessing import StandardScaler
import numpy as np

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,30]])

std1 = StandardScaler()
data1 = std1.fit_transform(a)
print(data1)

std2 = StandardScaler()
data2 = std1.fit(a)
print(data2)
data3 = std1.transform(b)
print(data3)