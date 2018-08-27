# 特征降维：减少相关特征的数据（包括特征选择和主成分分析）

# 特征选择：
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

# 过滤式：1.方差选择法；2.相关系数

def varthreshold():
    """使用【方差法】进行【过滤】"""
    factor = np.array([[1, 2, 5],
                       [1, 3, 1],
                       [1, 4, 1]], dtype=int)

    # 使用VarianceThreshold，不知道每个特征的方差，无法设定阈值，故少用
    var = VarianceThreshold(threshold=0.0)      # <=阈值的列（特征）会被删除
    data = var.fit_transform(factor)
    print(data)
    return None

# 广告费投入与月均销售额

def correlated_factor():
    """对股票的一些常见财务指标进行相关性计算"""
    factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev',
              'earnings_per_share', 'revenue', 'total_expense']
    data = pd.read_csv("./data_returns.csv")

    for i in range(len(factor)):
        for j in range(i,len(factor)-1):
            print("指标1：%s和指标2：%s 线性相关系数为：%f" % (
                factor[i],
                factor[j+1],
                pearsonr(data[factor[i]],data[factor[j+1]])[0]
            ))
    return None


if __name__ == '__main__':
    correlated_factor()
