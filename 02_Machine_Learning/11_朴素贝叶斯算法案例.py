"""
文档分类步骤：
    1.分割数据集
    2.tfidf进行的特征抽取
    3.朴素贝叶斯预测
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def multinormalcls():
    """
    朴素贝叶斯对20类新闻进行分类
    :return:
    """
    news = fetch_20newsgroups(subset='all')

    # 1、分割数据集，此时分割出来的训练测试集的特征值是一篇文章，故不能作为特征值
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3)

    # 2、进行文本特征抽取（算法能够知道特征值 - 特征值数字化）
    tfidf = TfidfVectorizer()

    # 对训练集特征值进行特征抽取
    x_train = tfidf.fit_transform(x_train)

    # print(x_train) 返回值为sparse矩阵
    print(x_train.toarray())
    # 之所以有很多0，是因为有些词，在求tiidf时为0，即重要性为0

    print(tfidf.get_feature_names())

    """
    # 由于是训练【历史数据】，即训练集里面的特征值，所以【不能】重新再训练【测试集的特征值】。因为【x_test】值有可能没有出现在【x_train】里
    # a = tfidf.get_feature_names()
    # print(a)
    # print(x_train.toarray())

    # 对测试集的特征值进行抽取
    # x_test = tfidf.fit_transform(x_test)
    # b = tfidf.get_feature_names()
    # print(b)
    #
    # print(a==b)
    """

    x_test = tfidf.transform(x_test)

    # 进行朴素贝叶斯算法预测
    mlb = MultinomialNB(alpha=1.0)

    mlb.fit(x_train, y_train)

    # 预测，准确率
    print("预测【测试集】当中的文档类别是：", mlb.predict(x_test)[:50])
    print("真实【测试集】当中的文档类别是：", y_test[:50])

    # 得出准确率
    print("文档分类的准确率为：", mlb.score(x_test, y_test))

    return None

if __name__ == '__main__':
    multinormalcls()

