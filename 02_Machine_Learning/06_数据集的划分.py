from sklearn.datasets import load_iris, load_boston, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 【分类】数据集，【目标值】是【离散】型数据
lr = load_iris()
# print("特征值：", lr.data)
# print("目标值：", lr.target)
# print("数据介绍：", lr.DESCR)
# print("每个特征的特征名是", lr.feature_names)
# print("每个目标的目标名是",lr.target_names)  # 【目标】就是【标签】

# 【回归】数据集，【目标值】是【连续】型数据
# lb = load_boston()
# print("特征值：", lb.data)
# print("目标值：", lb.target)
# print("数据介绍：", lb.DESCR)      # 回归数据集没有特征名和目标名

# 数据量比较大的数据集, 新闻型数据集，目标值为离散型
news = fetch_20newsgroups(subset="all")   # 可选trian, test, all

# print("特征值：", news.data)
# print("目标值：", news.target)
# print("数据介绍：", news.DESCR)
# print(news.feature_names)
# print(news.target_names)


# 将【数据集】划分为【训练集】和【测试集】
# 返回值由4个部分接收: x_train, x_test, y_train, y_test
# x,y：特征值和目标值； train,test：训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.3)     # test_size指定测试集的比例大小
print("训练集的特征值", x_train)
print("测试集的特征值", x_test)
print("训练集的目标值", y_train)
print("测试集的目标值", y_test)