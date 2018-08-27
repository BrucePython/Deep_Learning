
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def knncls():
    lr = load_iris()
    print(lr.data)
    print(lr.target)
    x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.3)

    # 对数据进行标准化处理：
    std = StandardScaler()
    # 对训练集的特征值做标准化处理
    x_train = std.fit_transform(x_train)
    # 对测试集的特征值做标准化处理
    x_test = std.fit_transform(x_test)

    # 使用k近邻算法去进行训练预测
    knn = KNeighborsClassifier(n_neighbors=5)

    # 调用fit和predict或者score（估计器的流程）
    knn.fit(x_train,y_train)

    # 预测测试集
    y_predict = knn.predict(x_test)
    print("y_test:", y_test)
    print("K近邻算法：", y_predict)
    print("K近邻算法的准确率", knn.score(x_test, y_test))
    return None
if __name__ == '__main__':
    knncls()