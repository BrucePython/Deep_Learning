import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def logistic():
    """使用逻辑回归进行肿瘤数据预测"""
    # 由于数据没有列标签，则会默认把第一列作为列标签。为了避免，需要定义列标签
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    # 读取数据
    # data = pd.read_csv("./breast-cancer-wisconsin.data", names=column_name)    # 指定列标签的名字
    data = pd.read_csv("./breast-cancer-wisconsin.txt", names=column_name)    # 指定列标签的名字
    print(data.shape)
    # 处理缺失值: 将问号替换成np.nan
    data = data.replace(to_replace="?", value=np.nan)
    # 删掉np.nan数据
    data = data.dropna()
    print(data.shape)

    # 取出特征值，共11列数据，第一列用语检索的id，后9列分别是与肿瘤相关的医学特征，最后一列表示肿瘤类型的数值。
    x = data.iloc[:,1:10]   # : 取出所有的样本；不包含第11列和第0列（id），左闭右开。包含第2列到第10列
    # 取出目标值
    y = data.iloc[:,10]     # 第11列为特征值，2或4

    # 将数据集分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

    # 进行标准化(因为前面的输入也是一个线性回归)
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归进行训练和预测（也是不断更新权重和偏置）
    lr = LogisticRegression()
    # 默认会把4（恶性）当做正例（1类别）；把2（良性）当做反例（0类别）
    lr.fit(x_train,y_train)
    print("逻辑回归计算出的权重：", lr.coef_)
    print("逻辑回归方程计算出的偏置：", lr.intercept_)

    print("逻辑回归在测试集当中的预测类别", lr.predict(x_test))

    # print(np.array(y_test)) # 将pandas转换成numpy数组
    print("逻辑回归预测的准确度：", lr.score(x_test, y_test))

    # 召回率
    print("预测的召回率为：", classification_report(y_test,y_pred=lr.predict(x_test),labels=[2,4],target_names=["良性", "恶性"]))

    # 衡量样本不均衡下的评估，ROC和AUC
    # 查看分类中验证数据的AUC指标值，一定要结合场景，因为有样本均衡的情况
    # 先把2和4转换成0，1：如果y_test大于2.5则是y_test=4，输出1；小于2.5，则y_test=2,输出0
    y_test = np.where(y_test > 2.5, 1, 0)
    print("此场景的分类器的AUC指标为：", roc_auc_score(y_test, lr.predict(x_test)))     # 越高，说明分类越好

    return None

if __name__ == '__main__':
    logistic()