# import manipulation lybrary
import pandas as pd
import numpy as np

# import visualization lybrary
import matplotlib.pyplot as plt
import seaborn as sns

# import machine learning lybraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder,LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


from collections import OrderedDict

from data_ingestion import data_ingestion
# data exploration 
def data_exploration(df):
    # segregate numerical and categorical columns 
    numerical_col = df.select_dtypes(exclude='object').columns
    categorical_col = df.select_dtypes(include = 'object').columns

    # numerical descriptive stats
    numerical_stats = []

    Q1 = df[numerical_col].quantile(0.25)
    Q3 = df[numerical_col].quantile(0.75)
    IQR = Q3 - Q1
    LW = Q1 - 1.5*IQR
    UW = Q3 + 1.5*IQR
    Outlier_Count = (df[numerical_col] < LW) | (df[numerical_col] > UW)
    Outlier_Percentage = (Outlier_Count.sum()/len(df))*100

    for i in numerical_col:
        num_stats = OrderedDict({
        "Feature":i,
        "Count":df[i].count(),
        "Maximum":df[i].max(),
        "Minimum":df[i].min(),
        "Mean":df[i].mean(),
        "Median":df[i].median(),
        "Q1":Q1,
        "Q3":Q3,
        "IQR":IQR,
        "Lower_Whisker":LW,
        "Upper_Whisker":UW,
        "Outlier_Count": Outlier_Count.sum(),
        "Outlier_Percentage":Outlier_Percentage,
        "Skewness":df[i].skew(),
        "Kurtosis":df[i].kurtosis(),
        "Standard Deviation":df[i].std()

    })
    
        numerical_stats.append(num_stats)
        numerical_stats_report = pd.DataFrame(numerical_stats)
    return numerical_stats_report

numerical_stats_report = data_exploration(df)
numerical_stats_report


      


