from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def sgdregressor():
    """线性回归两种求解方法进行房价预测之梯度下降"""
    # 获取数据进行分割
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.3)

    # 对数据进行标准化处理，避免权重与特征值相乘时造成影响
    std = StandardScaler()

    # x_train = StandardScaler().fit_transform(x_train)
    x_train = std.fit_transform(x_train)

    # 处理测试集特征值。测试集与训练集无关。如果在这里使用fit_transform, 得到的是测试集的方差和均值，训练集的均值方差将被覆盖
    x_test = std.transform(x_test)

    # 使用线性回归的模型进行训练和预测
    # 2. 梯度下降求解方式：LinearRegression
    # sgd = SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate="invscaling")
    # 自定义学习率: 0~1, 0.01, 0.1etc.
    sgd = SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate="constant",eta0=0.1)
    sgd.fit(x_train,y_train)

    # 进行矩阵运行，得出参数w的结果
    sgd.fit(x_train, y_train)

    print("梯度下降方程计算出的权重：", sgd.coef_)
    print("梯度下降方程计算出的偏置：", sgd.intercept_)

    # 如果要对新数据进行预测，那么只需要将该数据的特征与相对于的权重一一相乘，相加之后再加上偏置，就是预测的结果

    # 调用predict取预测目标值
    y_sgd_predict = sgd.predict(x_test)
    # 预测前100个样本的结果
    print("测试集预测的价格为：", y_sgd_predict[:100])

    # 调用均方误差评估SGDRegressor的结果误差: ; 平均每个样本的误差为根号（23）
    sgd_error = mean_squared_error(y_test, y_sgd_predict)

    print("SGDRegressor的结果误差为：", sgd_error)
    return None


if __name__ == '__main__':
    sgdregressor()
