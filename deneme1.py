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

df = pd.read_csv("/Users/yunusemreturkoglu/Desktop/BootCamp/izyco/train_sample.csv")
df.head()
df.info()

df["date"] = df['month_id'].apply(lambda x: str(x)[:4] + "-" + str(x)[4:])
df["date"] = pd.to_datetime(df["date"])
df["mcc_id"] = df["mcc_id"].apply(lambda x: str(x)[4:])
# df["mcc_id"] = df['mcc_id'].apply(lambda x: str(x)[4:])
# df["mcc_id"] = df["mcc_id"].astype(str).astype(int)

color_pal = sns.color_palette()
df.plot(style=".", x="date", y="net_payment_count", figsize=(10,5), ms=2, color=color_pal[0], title="Payment")
plt.show()

color_pal = sns.color_palette()
df.plot(style=".", x="working_type", y="net_payment_count", figsize=(10,5), ms=2, color=color_pal[0], title="Payment")
plt.show()

df["working_type"].value_counts()
df["merchant_source_name"].value_counts()
df["settlement_period"].value_counts()
df["merchant_segment"].value_counts()

def mevsimler(ay):
    if ay in [12,1,2]:
        return "Kis"
    elif ay in[3,4,5]:
        return "Ilkbahar"
    elif ay in [6,7,8]:
        return "Yaz"
    else:
        return"Sonbahar"

def create_features(dataframe):
    dataframe["ceyreklik"] = dataframe['date'].dt.quarter
    dataframe["ay"] = dataframe['date'].dt.month
    dataframe['yil'] = dataframe['date'].dt.year
    dataframe["mevsim"] = dataframe["ay"].apply(mevsimler)
create_features(df)
df.head()
df.isnull().sum()
df[df["net_payment_count"].isna()]

# segment_ratio
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df['merchant_segment'] == "Segment - 2"), "segment_ratio"] = "highlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df['merchant_segment'] == "Segment - 3"), "segment_ratio"] = "highlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df['merchant_segment'] == "Segment - 4"), "segment_ratio"] = "highlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df['merchant_segment'] == "Segment - 1"), "segment_ratio"] = "highlevel4"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df['merchant_segment'] == "Segment - 2"), "segment_ratio"] = "midlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df['merchant_segment'] == "Segment - 3"), "segment_ratio"] = "midlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df['merchant_segment'] == "Segment - 4"), "segment_ratio"] = "midlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df['merchant_segment'] == "Segment - 1"), "segment_ratio"] = "midlevel4"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df['merchant_segment'] == "Segment - 2"), "segment_ratio"] = "midlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df['merchant_segment'] == "Segment - 2"), "segment_ratio"] = "lowlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df['merchant_segment'] == "Segment - 3"), "segment_ratio"] = "lowlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df['merchant_segment'] == "Segment - 4"), "segment_ratio"] = "lowlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df['merchant_segment'] == "Segment - 1"), "segment_ratio"] = "lowlevel1"

df.head()
df["segment_ratio"].isnull().sum()

# Source_ratio

df["merchant_source_name"].value_counts()
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["merchant_source_name"] == "Merchant Source - 3"), "source_ratio"] = "sourceHlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["merchant_source_name"] == "Merchant Source - 1"), "source_ratio"] = "sourceHlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["merchant_source_name"] == "Merchant Source - 2"), "source_ratio"] = "sourceHlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["merchant_source_name"] == "Merchant Source - 3"), "source_ratio"] = "sourceMlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["merchant_source_name"] == "Merchant Source - 1"), "source_ratio"] = "sourceMlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["merchant_source_name"] == "Merchant Source - 2"), "source_ratio"] = "sourceMlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["merchant_source_name"] == "Merchant Source - 3"), "source_ratio"] = "sourceLlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["merchant_source_name"] == "Merchant Source - 1"), "source_ratio"] = "sourceLlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["merchant_source_name"] == "Merchant Source - 2"), "source_ratio"] = "sourceLlevel3"

df["source_ratio"].isnull().sum()

# working type ratio

df["working_type"].value_counts()

df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["working_type"] == "Working Type - 2"), "type_ratio"] = "typeHlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["working_type"] == "Working Type - 5"), "type_ratio"] = "typeHlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["working_type"] == "Working Type - 1"), "type_ratio"] = "typeHlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["working_type"] == "Working Type - 3"), "type_ratio"] = "typeHlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["working_type"] == "Working Type - 4"), "type_ratio"] = "typeHlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 3") & (df["working_type"] == "Working Type - 6"), "type_ratio"] = "typeHlevel3"

df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["working_type"] == "Working Type - 2"), "type_ratio"] = "typeMlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["working_type"] == "Working Type - 5"), "type_ratio"] = "typeMlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["working_type"] == "Working Type - 1"), "type_ratio"] = "typeMlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["working_type"] == "Working Type - 3"), "type_ratio"] = "typeMlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["working_type"] == "Working Type - 4"), "type_ratio"] = "typeMlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 2") & (df["working_type"] == "Working Type - 6"), "type_ratio"] = "typeMlevel3"

df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["working_type"] == "Working Type - 2"), "type_ratio"] = "typeLlevel1"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["working_type"] == "Working Type - 5"), "type_ratio"] = "typeLlevel2"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["working_type"] == "Working Type - 1"), "type_ratio"] = "typeLlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["working_type"] == "Working Type - 3"), "type_ratio"] = "typeLlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["working_type"] == "Working Type - 4"), "type_ratio"] = "typeLlevel3"
df.loc[(df['settlement_period'] == "Settlement Period - 1") & (df["working_type"] == "Working Type - 6"), "type_ratio"] = "typeLlevel3"

month_list = []
month_list2 = []
df["month_id"].value_counts()
def check_stats1(dataframe, col):

    IQR_tran = dataframe[col].value_counts().quantile(0.75) - dataframe[col].value_counts().quantile(0.25)
    value = dataframe[col].value_counts()

    for idx, count in value.items():
        if count >= dataframe[col].value_counts().quantile(0.75) + IQR_tran:
            month_list.append(idx)
        elif count < dataframe[col].value_counts().quantile(0.75) + IQR_tran and count > dataframe[col].value_counts().quantile(0.25):
            month_list2.append(idx)

check_stats1(df,"month_id")

df["month_stats"] = df["month_id"].apply(lambda x: "month_level1" if x in month_list else ("month_level2" if x in month_list2 else "month_level3"))
df.head()

merchant_list = []
merchant_list2 = []
def check_stats(dataframe, col):

    IQR_tran = dataframe[col].value_counts().quantile(0.75) - dataframe[col].value_counts().quantile(0.25)
    value = dataframe[col].value_counts()

    for idx, count in value.items():
        if count >= dataframe[col].value_counts().quantile(0.75) + IQR_tran:
            merchant_list.append(idx)
        elif count < dataframe[col].value_counts().quantile(0.75) + IQR_tran and count > dataframe[col].value_counts().quantile(0.25):
            merchant_list2.append(idx)
check_stats(df, "mcc_id")

df["mcc_stats"] = df["mcc_id"].apply(lambda x: "mcc_level1" if x in merchant_list else ("mcc_level2" if x in merchant_list2 else "mcc_level3"))


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe["count_lag_" + str(lag)] = dataframe.groupby["net_payment_count"].transfrom(lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

def lag_feature(dataframe, lag):
    dataframe["count_lag_" + str(lag)] = dataframe["net_payment_count"].shift(lag) + random_noise(dataframe)
    return dataframe

df = lag_feature(df,78180)
df = lag_feature(df, 78187)
df = lag_feature(df, 78194)
df = lag_feature(df, 78201)
df = lag_feature(df, 78208)
df.isnull().sum()
df.head()

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_cols.remove("mcc_id")
cat_cols.remove("merchant_id")


num_cols = [col for col in df.columns if df[col].dtypes != "O"]

num_cols.remove("net_payment_count")
num_cols.remove("date")
# num_cols.remove("month_id")
# num_cols.remove("mcc_id")


rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

fig, ax = plt.subplots(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns,
                    yticklabels=corr.columns, ax=ax)

df.head()

df["merchant_id"] = df["merchant_id"].apply(lambda x: str(x)[9:])
df["merchant_id"] = df["merchant_id"].astype("int")
df["mcc_id"] = df["mcc_id"].astype("int")
df.info()

df_sub = df[df["net_payment_count"].isna()]
df_train = df[df["net_payment_count"].notna()]

df_train.sort_values(by='date', ascending=False).head(40)
len(df_train)
df_train.dropna(inplace=True)
df.head()
df["mcc_id"] = df["mcc_id"].apply(lambda x: str(x)[4:])

### Model XGBoost

split_date = "2023-01-01"

df_train1_tr = df_train.loc[(df_train.date <= split_date)].copy()
df_train1_te = df_train.loc[df_train.date > split_date].copy()

y_train = df_train1_tr["net_payment_count"]
X_train = df_train1_tr.drop(["net_payment_count", "date"], axis=1)
y_test = df_train1_te["net_payment_count"]
X_test = df_train1_te.drop(["net_payment_count", "date"], axis=1)

xgb_model = XGBRegressor().fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mean_absolute_error(y_test, y_pred)
# 539.3251196614293

df_sub
y1 = df_sub["net_payment_count"]
X1 = df_sub.drop(["net_payment_count", "date"], axis=1)

df_sub["net_payment_count"] = xgb_model.predict(X1)
df_sub.head(30)
df_sub.reset_index(inplace=True)
df_submission = pd.read_csv("/Users/yunusemreturkoglu/Desktop/BootCamp/izyco/sample_submission.csv")
df_submission["net_payment_count"] = df_sub["net_payment_count"]
df_submission.index = df_submission["id"]
df_submission.drop("id", inplace=True, axis=1)
df_submission.head()
df_submission.to_csv("submission1_sample_XGB1.csv")


# Model LightGBM

lgb_model = LGBMRegressor().fit(X_train,y_train)
y_pred = lgb_model.predict(X_test)
mean_absolute_error(y_test, y_pred)
# 613.0434488494333

df_sub
y1 = df_sub["net_payment_count"]
X1 = df_sub.drop(["net_payment_count", "date"], axis=1)

df_sub["net_payment_count"] = lgb_model.predict(X1)
df_sub.head(30)
df_sub.reset_index(inplace=True)
df_submission = pd.read_csv("/Users/yunusemreturkoglu/Desktop/BootCamp/izyco/sample_submission.csv")
df_submission["net_payment_count"] = df_sub["net_payment_count"]
df_submission.index = df_submission["id"]
df_submission.drop("id", inplace=True, axis=1)
df_submission.tail()
df_submission.to_csv("submission1_sample_LBM.csv")

# Model Catboost

cat_model = CatBoostRegressor().fit(X_train, y_train)
y_pred = cat_model.predict(X_test)
mean_absolute_error(y_test, y_pred)
# 565.8143550929944

# KNN Model

knn_model = KNeighborsRegressor().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
mean_absolute_error(y_test, y_pred)

# bagg model

bagg_model = BaggingRegressor(bootstrap_features=True).fit(X_train,y_train)
y_pred = bagg_model.predict(X_test)
mean_absolute_error(y_test, y_pred)

# DecisionTRee Model

split_date = "2023-02-01"

df_train1_tr = df_train.loc[(df_train.date <= split_date)].copy()
df_train1_te = df_train.loc[df_train.date > split_date].copy()

y_train = df_train1_tr["net_payment_count"]
X_train = df_train1_tr.drop(["net_payment_count", "date"], axis=1)
y_test = df_train1_te["net_payment_count"]
X_test = df_train1_te.drop(["net_payment_count", "date"], axis=1)

tree_model = DecisionTreeRegressor().fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
mean_absolute_error(y_test, y_pred)
# 478.1694099311436
fig, ax = plt.subplots(figsize=(10,10))
Importance = pd.DataFrame({"Importance": tree_model.feature_importances_*100},
                         index = X_train.columns)

Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind ="barh", color = "r", ax=ax)

plt.xlabel("Variable Importance Level ")
plt.show()

df_sub.drop("index", inplace=True, axis=1)
y1 = df_sub["net_payment_count"]
X1 = df_sub.drop(["net_payment_count", "date"], axis=1)

df_sub["net_payment_count"] = tree_model.predict(X1)
df_sub.head(30)
df_sub.reset_index(inplace=True)
df_submission = pd.read_csv("/Users/yunusemreturkoglu/Desktop/BootCamp/izyco/sample_submission.csv")
df_submission["net_payment_count"] = df_sub["net_payment_count"]
df_submission.index = df_submission["id"]
df_submission.drop("id", inplace=True, axis=1)
df_submission.head()
df_submission.to_csv("submission1_sample_tree2.csv")
# Random Forest Model

rf_model = RandomForestRegressor().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mean_absolute_error(y_test, y_pred)
# 511.60707508148795

df_sub["net_payment_count"] = rf_model.predict(X1)
df_sub.head(30)
df_sub.reset_index(inplace=True)
df_submission = pd.read_csv("/Users/yunusemreturkoglu/Desktop/BootCamp/izyco/sample_submission.csv")
df_submission["net_payment_count"] = df_sub["net_payment_count"]
df_submission.index = df_submission["id"]
df_submission.drop("id", inplace=True, axis=1)
df_submission.head()
df_submission.to_csv("submission1_sample_rf.csv")
# Keras Model

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


split_date = "2023-06-01"

df_train1_tr = df_train.loc[(df_train.date <= split_date)].copy()
df_train1_te = df_train.loc[df_train.date > split_date].copy()


y_train = df_train1_tr["net_payment_count"]
X_train = df_train1_tr.drop(["net_payment_count", "date", "merchant_id"], axis=1)
y_test = df_train1_te["net_payment_count"]
X_test = df_train1_te.drop(["net_payment_count", "date", "merchant_id"], axis=1)

X_train.shape
y_train.shape
X_test.shape

model1 = Sequential()
model1.add(InputLayer((23,1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))
model1.summary()

cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanAbsoluteError(), optimizer=Adam(learning_rate=0.0001), metrics=[MeanAbsolutePercentageError()])
model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[cp1])

from tensorflow.keras.models import load_model
model1 = load_model('model1/')

test_predictions = model1.predict(y_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
test_results

