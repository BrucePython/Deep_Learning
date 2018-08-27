from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd



def knncls():
    data = pd.read_csv("")
    # 0. 缩小数据范围（为了方便快速查询）
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 2. 把签到位置小于N个人的位置删除掉（可不做）
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    # 3. 分割数据集到训练集和测试集
    # 取出特征值和目标值
    # y = data[["place_id"]]  # y_train的索引太多了，y还是一个二维数据，需变成一维数据
    y = data["place_id"]
    x = data[["x", "y", "accuracy", "time"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 进行数据的标准化处理
    std = StandardScaler()
    # 对训练集的特征值做标准化处理
    x_train = std.fit_transform(x_train)
    # 对测试集的特征值做标准化处理
    x_test = std.fit_transform(x_test)

    # 应用网格搜索 + 交叉验证 对 K近邻算法进行调优
    knn = KNeighborsClassifier()

    # 构造超参数字典：
    # 对于knn，数据量较大时，k=根号（样本数）
    param = {"n_neighbors":[1,3,5,7,10]}

    gc = GridSearchCV(knn, param_grid=param, cv=2)   # 为了快速看到效果，只做2折分析

    # 训练数据
    gc.fit(x_train,y_train)

    # 查看模型超参数调优的过程，交叉验证的结果
    print("在2折交叉验证当中的最好结果", gc.best_score_)
    print("选择的最好的模型参数是：", gc.best_estimator_)
    print("每次交叉验证的验证集的预测结果：", gc.cv_results_)

    # 预测测试集的准确率
    print("在测试集当中的最终预测结果为：", gc.score(x_test,y_test))


if __name__ == '__main__':
    knncls()