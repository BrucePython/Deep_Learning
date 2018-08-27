# 探究用户对物品类别的喜好细分降维

import pandas as pd
from sklearn.decomposition import PCA

# 1. 导入四张表
# /Users/zhengtaizhong/Documents/Python_code/MachineLearning/day01/instacart/aisles.csv
aisles = pd.read_csv("./instacart/aisles.csv")
priors = pd.read_csv("./instacart/order_products_prior.csv")
orders = pd.read_csv("./instacart/orders.csv")
products = pd.read_csv("./instacart/products.csv")

# 2. 合并四张表到一张表当中，按照两张表相同的键
# on指定两张表共同拥有的键, 传两遍
mt = pd.merge(priors, products, on=["product_id", "product_id"])  # 按照内链接
mt1 = pd.merge(mt, orders, on=["order_id", "order_id"])
mt2 = pd.merge(mt1, aisles, on=["aisle_id", "aisle_id"])
# print(mt2.shape)

# 3. 进行交叉表(用于计算分组个数，寻找两个列之间的关系)变换，用户跟物品类别的分组次数统计
# 用户买了哪些物品
user_aisle = pd.crosstab(mt2["user_id"],mt2["aisle"])
# print(user_aisle)

# 4. 进行PCA主成分分析
pca = PCA(n_components=0.95)
data = pca.fit_transform(user_aisle )
print(pd.DataFrame(data))