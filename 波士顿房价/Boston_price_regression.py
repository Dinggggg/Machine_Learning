# -*- coding=utf-8 -*-

# @Author: Ding Junwei
# @Student ID: 320180939671
# @Email: dingjw18@lzu.edu.cn
# @WeChat: D521101815
# @File: Boston_price_regression.py
# @Time: 2020/6/21 17:33
# @Source: https://my.oschina.net/u/2245781/blog/1855834

# 一元回归预测波士顿房价
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_predict
from numpy import shape
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import csv
import matplotlib.pyplot as plt

dataSet = load_boston()
data_X = dataSet['data']

data_y = dataSet['target']
feature_names= dataSet['feature_names']
# print(shape(data_X))
'''(506, 13)'''
# print(shape(data_y))
'''(506,)'''
# print(data_X[:2])
'''[[6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 6.5750e+00
  6.5200e+01 4.0900e+00 1.0000e+00 2.9600e+02 1.5300e+01 3.9690e+02
  4.9800e+00]
 [2.7310e-02 0.0000e+00 7.0700e+00 0.0000e+00 4.6900e-01 6.4210e+00
  7.8900e+01 4.9671e+00 2.0000e+00 2.4200e+02 1.7800e+01 3.9690e+02
  9.1400e+00]]'''
# print(data_y[:2])
'''[24.  21.6]'''
# print(feature_names)
'''['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']'''

# 数据存入csv文件
# with open('Boston.csv','w',newline='',encoding='utf-8') as f:
#     file = csv.writer(f)
#     file.writerow(feature_names)
#     file.writerows(data_X)
    # file.writerow(["MEDV"])
    # file.writerow(data_y)

# 划分训练、测试数据
X_train,X_test,y_train,y_test = train_test_split(data_X,data_y,test_size=0.2) # 将20%的样本划分为测试集，80%为训练集，即test_size=0.2
# print(shape(X_train))
'''(404, 13)'''
# print(shape(X_test))
'''(102, 13)'''

# 运行线性模型。选用sklearn中基于最小二乘的线性回归模型，并用训练集进行拟合，得到拟合直线y=wTx+b中的权重参数w和b
model = LinearRegression()
model.fit(X_train,y_train)
# print(model.coef_)
'''[-1.19241009e-01  5.79545236e-02  3.56219853e-03  1.89616206e+00
 -1.27218069e+01  3.67960121e+00 -7.53661572e-03 -1.53775349e+00
  2.92338076e-01 -1.37730131e-02 -9.17261411e-01  9.99698385e-03
 -5.35117019e-01]'''
# print(model.intercept_)
'''35.20513940837345'''

# 模型测试。利用测试集得到对应的结果，并利用均方根误差（MSE）对测试结果进行评价
'''
MSE: Mean Squared Error 
均方误差是指参数估计值与参数真值之差平方的期望值; 
MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。
'''
y_pred = model.predict(X_test)
# print("MSE:{}".format(metrics.mean_squared_error(y_test,y_pred)))
'''MSE:29.411284648524305'''
'''MSE:20.913114518321517'''
'''MSE:28.217779699975278'''

# 交叉验证。使用10折交叉验证，即cv=10，并求出交叉验证得到的MSE值
predicted = cross_val_predict(model,data_X,data_y,cv = 10)
# print("MSE:{}".format(metrics.mean_squared_error(data_y,predicted)))
'''MSE:34.53965953999329'''

# 画图。将实际房价数据与预测数据作出对比，接近中间天蓝色直线的数据表示预测准确
plt.scatter(data_y,predicted,c = '#7A475A',marker= '.')
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.title('Boston_Price_Predicted(thousand dollars)')
plt.scatter(data_y,data_y,c = '#1CDFB3',marker='+')
# plt.show()
# # plt.savefig('Boston_Price_Regression.png')