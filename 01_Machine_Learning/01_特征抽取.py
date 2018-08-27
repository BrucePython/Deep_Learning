import jieba

# 特征工程里面都是fit_transform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


# ictvectorizer:默认返回sparse（稀疏）矩阵,sparse=False

def dictvec():
    """对字典数据进行特征抽取"""
    # 实例化dictvec
    # 类别特征
    # dic = DictVectorizer()  # 默认返回稀疏矩阵（节省内存），
    dic = DictVectorizer(sparse=False)  # 返回one-hot编码
    # dictvec调用fit_transform
    # 三个样本的特征数据（字典形式）
    data = dic.fit_transform(
        [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}])
    print(dic.get_feature_names())
    print(data)
    return None


def countvec():
    # 实例化count
    count = CountVectorizer()  # 文本特征

    # 对两篇文章进行特征抽取：将两个文本转化成2维的array数组
    data = count.fit_transform(["Life is is short, I like python", "Life is too long, I dislike python"])
    # 内容
    print(count.get_feature_names())        # 返回值:单词列表；将所有文章的单词统计到一个列表当中（重复的词只当做一次），默认会过滤掉单个字母
    # print(data)          # 默认是sparse矩阵，但是CountVectorizer()函数里面没有sparse参数，
    print(data.toarray())  # 只能通过利用toarray()进行sparse矩阵转换array数组。
    # 对每篇文章在词的列表：['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']中， 统计出现额次数

    return None

def cutword():
    """进行分词处理"""
    c1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    c2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    c3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    content1 = ' '.join(list(c1))
    content2 = ' '.join(list(c2))
    content3 = ' '.join(list(c3))
    return content1, content2, content3


def chinesevec():
    # 实例化count
    count = CountVectorizer(stop_words=["不会","如果"])     # 指定停止词（过滤这两个词）

    # 进行三句话的分词
    content1, content2, content3 = cutword()

    # 对两篇文章进行特征处理
    data = count.fit_transform([content1, content2, content3])

    # 内容
    print(count.get_feature_names())
    print(data.toarray())
    # print(data)

    return None


def tfidfvec():
    # 实例化tfidf
    tfidf = TfidfVectorizer()


    # 进行三句话的分词
    content1, content2, content3 = cutword()

    # 对两篇文章进行特征处理
    data = tfidf.fit_transform([content1, content2, content3])

    # 内容
    print(tfidf.get_feature_names())
    print(data.toarray())
    # print(data)

    return None

if __name__ == '__main__':
    # countvec()
    # dictvec()
    tfidfvec()
