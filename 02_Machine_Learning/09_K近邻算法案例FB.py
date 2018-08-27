"""
数据介绍：将根据用户的位置，准确性和时间戳预测用户正在查看的业务。
说明：
    train.csv，test.csv
    row_id：登记事件的ID （相当于索引，与目标值无关，删掉）
    xy：坐标
    准确性：定位准确性
    时间：时间戳
    place_id：业务的ID，这是您预测的目标
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knncls():
    """K近邻算法预测用户查询的业绩"""
    data = pd.read_csv("./K近邻算法案例数据/train.csv")

    # 0. 缩小数据范围（为了方便快速查询）
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 1. 标准化处理，一般只能拿到训练集数据，所以在x_train之后做标准化处理

    # 2. 把签到位置小于N个人的位置删除掉（可不做）
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]


    # 3. 分割数据集到训练集和测试集
    # 取出特征值和目标值
    y = data[["place_id"]]
    x = data[["x","y","accuracy","time"]]
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    # 进行数据的标准化处理
    std = StandardScaler()
    # 对训练集的特征值做标准化处理
    x_train = std.fit_transform(x_train)
    # 对测试集的特征值做标准化处理
    x_test = std.fit_transform(x_test)

    # 4. 利用K近邻算法去训练和预测
    knn = KNeighborsClassifier(n_neighbors=5)
    # 调用fit，predict，score方法
    knn.fit(x_train,y_train)
    # 预测 测试集办理的业务类型
    y_predict = knn.predict(x_test)
    print("K近邻算法预测的这些事件的业务类型: ", y_predict)
    print("K近邻算法预测的准确率为：", knn.score(x_test,y_test))

    return None


if __name__ == '__main__':
    knncls()
