# 主成分分析：压缩维度

from sklearn.decomposition import PCA
import pandas as pd


def pca_test():
    pca = PCA(n_components=0.95)
    # n_components=int, 减少到int个特征
    # = 小数，减少到的百分比

    data = pca.fit_transform([[2, 8, 4, 5],
                              [6, 3, 0, 8],
                              [5, 4, 9, 1]])
    print(data)


def pca_mergedata():
    """将股票的'revenue'和'total_expense'合成一个新的特征（高度相关的数据合并）"""
    data = pd.read_csv("./data_returns.csv")
    pca = PCA(n_components=1)
    data = pca.fit_transform(data[['revenue', 'total_expense']])


if __name__ == '__main__':
    pca_test()
