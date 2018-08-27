# 应用网格搜索+交叉验证对k近邻算法进行优化

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def knncls():
    lr = load_iris()
    # print(lr.data)
    # print(lr.target)
    x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.3)


    # 对数据进行标准化处理：
    std = StandardScaler()
    # 对训练集的特征值做标准化处理
    x_train = std.fit_transform(x_train)
    # 对测试集的特征值做标准化处理
    x_test = std.fit_transform(x_test)

    # 使用k近邻算法去进行训练预测
    knn = KNeighborsClassifier()

    # 应用网格搜索+交叉验证对k近邻算法进行优化
    # 构造一个超参数的字典,传入param_grid；对于数据量较大时，k=sqrt(样本数)
    param = {"n_neighbors":[1,3,5,7,10]}
    # cv: 交叉验证的次数
    gc = GridSearchCV(knn,param_grid=param,cv=2)   # gc相当于estimator

    # fit 输入数据
    gc.fit(x_train,y_train)
    print(y_train.shape)

    # 查看模型超参数调优的过程，交叉验证的结果
    print("在2折交叉验证中的最好结果", gc.best_score_)
    print("选择的最好的模型参数是", gc.best_estimator_)
    print("每次交叉验证的验证集的预测结果", gc.cv_results_)

    # 预测测试集的准确率
    print("在测试集中的最终测试结果为：", gc.score(x_test,y_test))
    return None


if __name__ == '__main__':
    knncls()