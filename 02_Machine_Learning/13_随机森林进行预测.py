import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

def decision():
    """
    决策树预测乘客生存分类
    :return:
    """
    # 1、获取乘客数据
    titan = pd.read_csv("./决策树算法案例数据/data.txt")

    # 2、确定特征值和目标值，缺失值处理，特征类别数据 -->one-hot编码
    x = titan[['pclass', 'age', 'sex']]

    y = titan['survived']

    # 填充缺失值
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 特征类别数据 -->one-hot编码(sparse=False)
    dic = DictVectorizer(sparse=False)

    # [["1st","2","female"],[]]--->[{"pclass":, "age":2, "sex: female"}, ]
    x = dic.fit_transform(x.to_dict(orient="records"))

    print(dic.get_feature_names())
    print(x)

    # 3、分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    """
    # 4、决策树进行预测
    dec = DecisionTreeClassifier()
    # dec = DecisionTreeClassifier(max_depth=5)

    dec.fit(x_train, y_train)

    print("预测的准确率为：", dec.score(x_test, y_test))

    # 导出到dot文件（estimator）, 只能导出一棵树
    export_graphviz(dec, out_file="./tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])
    """

    # 5. 随机森林进行预测, 使用网格搜索填补超参数
    rf = RandomForestClassifier()

    # 构造超参数字典
    param = {"n_estimators":[120,200,300,500,800,1200],
             "max_depth":[5,8,15,25,30],
             "min_samples_split":[2,3,5]}


    # 所有超参数的选择可能
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train,y_train)
    print("随机森林的准确性率：", gc.score(x_test,y_test))
    print("交叉验证选择的参数是： ", gc.best_estimator_)

    return None


if __name__ == '__main__':
    decision()