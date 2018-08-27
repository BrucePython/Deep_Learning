# 将数据拆分成 x_train,x_test,y_train,y_test后
# 先训练：fit(x_train,y_train)
# 根据训练后的模型，使用predict(x_test) 预测结果
# 或使用score(x_test,y_test) 衡量预测结果的准确率