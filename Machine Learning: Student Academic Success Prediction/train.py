print("Script started")
from ucimlrepo import fetch_ucirepo 
import pandas as pd 
import numpy as np
import pandas as pd

import pickle

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def clean_string_fields(df):
    strings = list(df.dtypes[df.dtypes == 'object'].index)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    return df

def split_dataset(data, seed):
    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state = seed)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state = seed)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    return df_full_train, df_train, df_val, df_test


print("Fetching Dataset")
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
metadata = predict_students_dropout_and_academic_success.metadata
  
# variable information 
var_info = predict_students_dropout_and_academic_success.variables

df = predict_students_dropout_and_academic_success.data.original

print("Cleaning data")
df.columns = df.columns.str.lower().str.replace(' ', '_')

df = clean_string_fields(df)

df = df[(df["mother's_occupation"]!= 99) & (df["father's_occupation"]!= 99)]

categorical_fields = ['marital_status', 'application_mode', 
                   'application_order', 'course',
                    'daytime/evening_attendance', 'previous_qualification',
                    'nacionality',
                    "mother's_qualification", "father's_qualification",
                    "mother's_occupation", "father's_occupation", 
                    'displaced', 'educational_special_needs', 'debtor',
                   'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
                    'international', 'target'
                   ]

for field in categorical_fields:
    df[field] = df[field].astype('category')
    
df_full_train, df_train, df_val, df_test = split_dataset(df, 1)

y_train = df_train.target
y_val = df_val.target
y_test = df_test.target

del df_train['target']
del df_val['target']
del df_test['target']

print("Training model")
# Training the model
train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

rf = RandomForestClassifier()


params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 15, 20, 50, 60, None],
              'min_samples_leaf' : [1, 5, 10, 15, 20, 500, 100, 200],
              'criterion' : ['gini', 'entropy', 'log_loss'],
              'n_estimators' : [100, 120, 140, 160, 50, 70, 30], 
              'random_state': [47]
              }

grid_search_rf = GridSearchCV(estimator=rf, param_grid=params, n_jobs=-1, cv=5)

grid_search_rf.fit(X_train, y_train)

with open('FinalModel.bin', 'wb') as f_out:
    pickle.dump((dv, grid_search_rf), f_out)
    
print("The model is saved to 'FinalModel.bin'")
