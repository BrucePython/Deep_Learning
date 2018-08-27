from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def ridge():
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

    # 使用带有L2正则化的线性回归去预测
    # 3. 岭回归求解方式：
    rd = Ridge(alpha=1.0)
    rd.fit(x_train,y_train)

    # 进行矩阵运行，得出参数w的结果
    rd.fit(x_train, y_train)

    print("岭回归计算出的权重：", rd.coef_)
    print("岭回归方程计算出的偏置：", rd.intercept_)

    # 调用predict取预测目标值
    y_rd_predict = rd.predict(x_test)
    # 预测前100个样本的结果
    print("测试集预测的价格为：", y_rd_predict[:100])

    # 调用均方误差评估SGDRegressor的结果误差: ; 平均每个样本的误差为根号（23）
    rd_error = mean_squared_error(y_test, y_rd_predict)

    print("岭回归的结果误差为：", rd_error)
    return None


if __name__ == '__main__':
    ridge()
