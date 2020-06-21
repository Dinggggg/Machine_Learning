# -*- coding=utf-8 -*-

# @Author: Ding Junwei
# @Student ID: 320180939671
# @Email: dingjw18@lzu.edu.cn
# @WeChat: D521101815
# @File: machine_learning.py
# @Time: 2020/6/20 23:23
# @Source: https://blog.csdn.net/qq_15537309/article/details/89048496?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase

from sklearn.datasets import load_iris
# load_iris()返回的是一个类似字典的对象通过关键字则可以获取对应的数据
import pandas as pd ,numpy as np , matplotlib as plt , csv
from sklearn.model_selection import train_test_split # 导入训练测试数据划分包
from sklearn.neighbors import KNeighborsClassifier # 导入KNN分类包

dataSet = load_iris()
data = dataSet['data'] # 数据(花对应的四个参数数据)
target = dataSet['target'] # 数据对应的标签(花的品种代号012)
feature_names = dataSet['feature_names'] # 数据特征的名称(四个参数名称)
target_names = dataSet['target_names'] # 标签的名称(花的品种名称，对应012)
# print(data[:10])
'''[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 [4.6 3.4 1.4 0.3]
 [5.  3.4 1.5 0.2]
 [4.4 2.9 1.4 0.2]
 [4.9 3.1 1.5 0.1]]'''
# print(target[:10])
'''[0 0 0 0 0 0 0 0 0 0]'''
# print(feature_names)
'''['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']'''
# print(target_names)
'''['setosa' 'versicolor' 'virginica']'''

# 鸢尾花数据存入csv
# with open('iris_data.csv','w',newline='',encoding= 'utf-8') as f:
#     file = csv.writer(f)
#     file.writerow(feature)
#     file.writerows(data)

# 训练数据&测试数据
X_train, X_test,y_train,y_test = train_test_split(data, target, random_state=0)

# 构建模型 k邻近分类器
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

# 做出预测
X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(target_names[prediction]))

# 计算模型精度
print("Model score:{}".format(knn.score(X_test,y_test)))


'''
Prediction:[0]
Predicted target name:['setosa']
Model score:0.9736842105263158
'''