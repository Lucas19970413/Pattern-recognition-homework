# Pattern-recognition-homework
# Bike sharing demand

# coding:utf-8
import numpy as np
import calendar
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 第一步：数据导入
DataTrain = pd.read_csv("../MS/train.csv")
DataTest = pd.read_csv("../MS/test.csv")
DataTrain.head()
DataTest.head()
DataTrain.info()
DataTest.info()
DataTrain.isnull().sum()
DataTest.isnull().sum()

# print(DataTrain.head())
# print(DataTest.head())
# print(DataTrain.info())
# print(DataTest.info())
# print(DataTrain.isnull().sum())
# print(DataTest.isnull().sum())
# 由结果可知提供的数据集没有数据丢失现象

# 第二步：数据可视化分析
# 查找异常值
DataTrain.reset_index(inplace=True)
fig1,(ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig1.set_size_inches(10, 5)
sns.regplot(x="index", y="temp", data=DataTrain, ax=ax1)
sns.regplot(x="index", y="atemp", data=DataTrain, ax=ax2)
fig2 = plt.figure()
fig2.set_size_inches(10, 5)
ax3 = fig2.add_subplot(1, 1, 1)
sns.regplot(x="index", y="humidity", data=DataTrain, ax=ax3)
fig3 = plt.figure()
fig3.set_size_inches(10, 5)
ax4 = fig3.add_subplot(1, 1, 1)
sns.regplot(x="index", y="windspeed", data=DataTrain, ax=ax4)
# 存在异常值体感温度，风速

# 数据预处理
# DataTrain.drop('index', inplace=True, axis=1)
DataTrain['datetime'] = pd.to_datetime(DataTrain['datetime'])
DataTrain['year'] = DataTrain['datetime'].dt.year
DataTrain['month'] = DataTrain['datetime'].dt.month
DataTrain['day'] = DataTrain['datetime'].dt.day
DataTrain['hour'] = DataTrain['datetime'].dt.hour
DataTrain['minute'] = DataTrain['datetime'].dt.minute
DataTrain.drop(labels="datetime", axis=1)
DataTrain.drop('index', inplace=True, axis=1)

# 采用随机森林为异常风速wind speed赋值,atep后文已舍掉，不做异常值处理
dataWind0 = DataTrain[DataTrain["windspeed"] == 0]
dataWindNot0 = DataTrain[DataTrain["windspeed"] != 0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
DataTrain = dataWindNot0.append(dataWind0)
DataTrain.sort_values("datetime", inplace=True)
# DataTrain.reset_index(inplace=True)
# DataTrain.drop('index', inplace=True, axis=1)
# DataTrain.to_csv("../MS/train2.csv", index=False)

# 特征分析
fig4, (ax6,ax7,ax8,ax9) = plt.subplots(ncols=4)
fig4.set_size_inches(16, 8)
fig5, (ax10,ax11,ax12,ax13) = plt.subplots(ncols=4)
fig5.set_size_inches(16, 8)
# season
sns.factorplot(x="season", y="count", data=DataTrain, kind="bar", ax=ax6)
# holiday
sns.factorplot(x="holiday", y="count", data=DataTrain, kind="bar", ax=ax7)
# working day
sns.factorplot(x="workingday", y="count", data=DataTrain, kind="bar", ax=ax8)
# weather
sns.factorplot(x="weather", y="count", data=DataTrain, kind="bar", ax=ax9)
# year
sns.factorplot(x="year", y="count", data=DataTrain, kind="bar", ax=ax10)
# month
sns.factorplot(x="month", y="count", data=DataTrain, kind="bar", ax=ax11)
# day
sns.factorplot(x="day", y="count", data=DataTrain, kind="bar", ax=ax12)
# hour
sns.factorplot(x="hour", y="count", data=DataTrain, kind="bar", ax=ax13)

fig6, (ax14,ax15,ax16,ax17) = plt.subplots(ncols=4)
fig6.set_size_inches(16, 8)
# temp
sns.factorplot(x="temp",  y="count", data=DataTrain, kind="bar", ax=ax14)
# atemp
sns.factorplot(x="atemp",  y="count", data=DataTrain, kind="bar", ax=ax15)
# humidity
sns.factorplot(x="humidity",  y="count", data=DataTrain, kind="bar", ax=ax16)
# windspeed
sns.factorplot(x="windspeed",  y="count", data=DataTrain, kind="bar", ax=ax17)

# 相关性分析
Correlation = DataTrain[:].corr()
# Correlation = DataTrain[["season", "holiday", "workingday", "weather", "temp",
#                         "atemp", "humidity", "windspeed", "year", "month", "day", "hour", "count"]].corr()
mask = np.array(Correlation)
mask[np.tril_indices_from(mask)] = False
fig7 = plt.figure()
fig7.set_size_inches(12, 15)
ax5 = fig4.add_subplot(1, 1, 1)
sns.heatmap(Correlation, mask=mask, square=True, annot=True, cbar=True, ax=ax5)

# 补充分析
fig8,(ax18,ax19) = plt.subplots(nrows=2)
holiday_hour = pd.DataFrame(DataTrain.groupby(["hour","holiday"],sort=True)["count"].mean()).reset_index()
sns.pointplot(x=holiday_hour["hour"], y=holiday_hour["count"],hue=holiday_hour["holiday"], data=holiday_hour, join=True,ax=ax18)
ax18.set(xlabel='Hour', ylabel='count',title="Count group by Hour Of holiday", label='big')

workingday_hour = pd.DataFrame(DataTrain.groupby(["hour","workingday"],sort=True)["count"].mean()).reset_index()
sns.pointplot(x=workingday_hour["hour"], y=workingday_hour["count"],hue=workingday_hour["workingday"], data=workingday_hour, join=True,ax=ax19)
ax19.set(xlabel='Hour', ylabel='count',title="Count group by Hour Of workingday",label='big')


plt.show()
# 确定特征变量
FeatureNames = ["season","holiday","workingday","weather","temp","humidity","windspeed","month","year","hour"]

# 建立模型


# 模型评估
