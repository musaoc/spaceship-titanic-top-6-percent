"""
Spaceship Titanic Classification — Kaggle Top 6% Solution
A competitive machine learning solution achieving a Top 6% standing in Kaggle's Spaceship Titanic competition by engineering domain-specific features and benchmarking gradient-boosted tree models.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/spaceship-titanic-top-6-for-beginners
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 1 ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- Cell 2 ---
df_train = pd.read_csv('../input/spaceship-titanic/train.csv')
df_test = pd.read_csv('../input/spaceship-titanic/test.csv')

df_train.head()

# --- Cell 3 ---
r1,c1 = df_train.shape
print('The training data has {} rows and {} columns'.format(r1,c1))
r2,c2 = df_test.shape
print('The validation data has {} rows and {} columns'.format(r2,c2))

# --- Cell 4 ---
df_train.info()

# --- Cell 5 ---
df_train.describe()

# --- Cell 6 ---
df_test.describe()

# --- Cell 7 ---
# To see the quantity of null vaues in all the columns.
# c1 stands for the number of columns in the training data.


print('MISSING VALUES IN TRAINING DATASET:')
print(df_train.isna().sum().nlargest(c1))
print('')
print('MISSING VALUES IN VALIDATION DATASET:')
print(df_test.isna().sum().nlargest(c2))

# --- Cell 8 ---
df_train.set_index('PassengerId',inplace=True)
df_test.set_index('PassengerId',inplace=True)

# --- Cell 9 ---
df_train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = df_train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)
df_test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = df_test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)

df_train['Age'] =df_train['Age'].fillna(df_train['Age'].median())
df_test['Age'] =df_test['Age'].fillna(df_test['Age'].median())

df_train['VIP'] =df_train['VIP'].fillna(False)
df_test['VIP'] =df_test['VIP'].fillna(False)

df_train['HomePlanet'] =df_train['HomePlanet'].fillna('Mars')
df_test['HomePlanet'] =df_test['HomePlanet'].fillna('Mars')

df_train['Destination']=df_train['Destination'].fillna("PSO J318.5-22")
df_test['Destination']=df_test['Destination'].fillna("PSO J318.5-22")

df_train['CryoSleep'] =df_train['CryoSleep'].fillna(False)
df_test['CryoSleep'] =df_test['CryoSleep'].fillna(False)

df_train['Cabin'] =df_train['Cabin'].fillna('T/0/P')
df_test['Cabin'] =df_test['Cabin'].fillna('T/0/P')


# --- Cell 10 ---
plt.figure(figsize=(15,18))
sns.heatmap(df_train.corr(), annot=True);

# --- Cell 11 ---
plt.pie(df_train.Transported.value_counts(), shadow=True, explode=[.1,.1], autopct='%.1f%%')
plt.title('Transported ', size=18)
plt.legend(['False', 'True'], loc='best', fontsize=12)
plt.show()

# --- Cell 12 ---
sns.countplot(df_train.Transported);

# --- Cell 13 ---
sns.countplot(df_train.HomePlanet,hue=df_train.Transported);
# Dude, Europa is gone

# --- Cell 14 ---
sns.countplot(df_train.VIP,hue=df_train.Transported);

# --- Cell 15 ---
sns.countplot(df_train.CryoSleep,hue=df_train.Transported);

# --- Cell 16 ---
sns.countplot(df_train.Destination,hue=df_train.Transported)
plt.xticks(rotation=90);

# --- Cell 17 ---
sns.boxplot(y=df_train.Age,x=df_train.Transported);
#Age is not affecting much. But I have a plan XD

# --- Cell 18 ---
# Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
df_train[['Deck','Num','Side']] = df_train.Cabin.str.split('/',expand=True)
df_test[['Deck','Num','Side']] = df_test.Cabin.str.split('/',expand=True)

# --- Cell 19 ---
sns.countplot(df_train.Deck,hue=df_train.Transported);

# --- Cell 20 ---
plt.figure(figsize=(10,5))
sns.histplot(data=df_train, x='Num', hue='Transported',bins=14);

# --- Cell 21 ---
sns.countplot(df_train.Side,hue=df_train.Transported);

# --- Cell 22 ---
sns.countplot(df_test.Side);

# --- Cell 23 ---
df_train['total_spent']= df_train['RoomService']+ df_train['FoodCourt']+ df_train['ShoppingMall']+ df_train['Spa']+ df_train['VRDeck']
df_test['total_spent']=df_test['RoomService']+df_test['FoodCourt']+df_test['ShoppingMall']+df_test['Spa']+df_test['VRDeck']

# --- Cell 24 ---
df_train['AgeGroup'] = 0
for i in range(6):
    df_train.loc[(df_train.Age >= 10*i) & (df_train.Age < 10*(i + 1)), 'AgeGroup'] = i
# Same for test data
df_test['AgeGroup'] = 0
for i in range(6):
    df_test.loc[(df_test.Age >= 10*i) & (df_test.Age < 10*(i + 1)), 'AgeGroup'] = i

# --- Cell 25 ---
sns.countplot(y=df_train['AgeGroup'],hue=df_train['Transported']);

# --- Cell 26 ---
from sklearn.preprocessing import LabelEncoder

categorical_cols= ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side','Num']
for i in categorical_cols:
    print(i)
    le=LabelEncoder()
    arr=np.concatenate((df_train[i], df_test[i])).astype(str)
    le.fit(arr)
    df_train[i]=le.transform(df_train[i].astype(str))
    df_test[i]=le.transform(df_test[i].astype(str))

# --- Cell 27 ---
df_train.head()

# --- Cell 28 ---
df_train= df_train.drop(['Name','Cabin'],axis=1)
df_test= df_test.drop(['Name','Cabin'],axis=1)

# --- Cell 29 ---
plt.figure(figsize=(30,10))
sns.heatmap(df_train.corr(), annot=True);

# --- Cell 30 ---
df_train['Transported']=df_train['Transported'].replace({True:1,False:0})

# --- Cell 31 ---
X=df_train.drop('Transported',axis=1)
y = df_train['Transported']

# --- Cell 32 ---
X.columns

# --- Cell 33 ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)

# --- Cell 34 ---

from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=1500,
                         eval_metric='Accuracy',
                        verbose=0)

# --- Cell 35 ---
model.fit(X_train,y_train)

# --- Cell 36 ---
pred_y=model.predict(X_val)

pred=model.predict(X_train)
    
print(accuracy_score(y_train.values,pred))
print(accuracy_score(y_val.values,pred_y))

# --- Cell 37 ---
from sklearn.model_selection import GridSearchCV
gcv=GridSearchCV(CatBoostClassifier(),param_grid={'iterations': range(200,2000,200), 'eval_metric': ['Accuracy'],'verbose':[0]},cv=3)
gcv.fit(X_train,y_train)
pred_y=gcv.predict(X_val)

pred=gcv.predict(X_train)
    
print(accuracy_score(y_train.values,pred))
print(accuracy_score(y_val.values,pred_y))

# --- Cell 38 ---
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(random_state=1,n_estimators=250,learning_rate=0.15,max_depth=3)
gb.fit(X_train,y_train)

# --- Cell 39 ---
pred_y=gb.predict(X_val)
pred=gb.predict(X_train)
    
print(accuracy_score(y_train.values,pred))
print(accuracy_score(y_val.values,pred_y))

# --- Cell 40 ---
# lets re fit the model on the entire data
gcv.fit(X,y)

# --- Cell 42 ---
y_pred = gcv.predict(df_test)

sub=pd.DataFrame({'Transported':y_pred.astype(bool)},index=df_test.index)

sub.head()

# --- Cell 43 ---
sub.to_csv('submission')

# --- Cell 44 ---
140/24458100



if __name__ == "__main__":
    print("Pipeline execution complete.")
