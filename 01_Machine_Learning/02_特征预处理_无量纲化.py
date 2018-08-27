# 特征与处理：通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程
# 无量纲化：使不同规格的数据转换到同一规格

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 归一化处理：对于异常数据不好处理，弃用
def minmaxscaler():
    """对约会对象数据进行归一化处理，因为要保持数据规格一致"""
    # 使用pandas读取数据，选择要处理的特征
    dating = pd.read_csv("./dating.txt")
    data = dating[["milage", "Liters", "Consumtime"]]

    # 实例化minmaxscaler进行fit_transform，默认的feature_range是[0,1]
    mm = MinMaxScaler(feature_range=(2,3))
    data = mm.fit_transform(data)

    print(data)
    print(data.shape)

    return 0


# 标准化处理，常用
def standardscaler():
    """对约会对象数据进行标准化处理，每列的特征数据的均值为0，标准差为1"""
    # 使用pandas读取数据，选择要处理的特征
    dating = pd.read_csv("./dating.txt")
    data = dating[["milage", "Liters", "Consumtime"]]

    # 实例化StandardScaler进行fit_transform，默认的feature_range是[0,1]
    std = StandardScaler()
    data = std.fit_transform(data)

    print(data)
    print(data.shape)

    return 0

if __name__ == '__main__':
    standardscaler()
