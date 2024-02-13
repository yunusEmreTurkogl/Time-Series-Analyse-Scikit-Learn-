import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, cross_validate
from sklearn import model_selection
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from warnings import filterwarnings
filterwarnings('ignore')
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import matplotlib.pylab as plt
import matplotlib.dates as mdates

df = pd.read_csv("/Users/yunusemreturkoglu/Desktop/izyco/train.csv")
df_sub = pd.read_csv("/Users/yunusemreturkoglu/Desktop/izyco/sample_submission.csv")

df.head(10)
df_sub.head(10)

df_sub["month_id"] = df_sub["id"].apply(lambda x: x[:6])
df_sub["merchant_id"] = df_sub["id"].apply(lambda x: x[6:])
df_sub["net_payment_count"] = np.nan
df_sub = df_sub[["merchant_id", "month_id", "net_payment_count"]]

df_sub = df_sub.merge(
    df[
        [
            "merchant_id",
            "merchant_source_name",
            "settlement_period",
            "working_type",
            "mcc_id",
            "merchant_segment",
        ]
    ].drop_duplicates(),
    on=["merchant_id"],
    how="left",
)

df_sub.head()
df_sub.info

df_asil = pd.concat([df, df_sub], axis=0).reset_index(drop=True)
df_asil.head()
df_asil.isna().sum()

df_asil.to_csv('train_sample.csv', index=False)
