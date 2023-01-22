# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix
# Workbook settings
pd.set_option('display.max_columns', None)
random.seed(110)







##Multi-Nomial Regression with Distance Group/ EXCEPTIONAL WORK

df3= flight_data_df2

df4 = df3.select_dtypes(include = ['float','integer'])

#Removing NaN Values
for column in df4.columns:
    if df4[column].isnull().any():
        count = df4[column].isnull().sum()
        print(column + " has " +str(count)+" NaN values")
       
df3_reduced = df4.dropna()

# Dropping columns with less than 2 classes

df3_new = df3_reduced.loc[:, df3_reduced.apply(pd.Series.nunique) > 1]


X = df3_new.drop('DistanceGroup', axis = 1)
y = df3_new.DistanceGroup


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 50)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


#Scaling the data
#Credit to: https://scikit-learn.org/stable/modules/preprocessing.html
scaler = preprocessing.StandardScaler().fit(X_train)

X_scaled = scaler.transform(X)

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Credit to https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
pipe = make_pipeline(StandardScaler(), LogisticRegression(multi_class = 'multinomial',solver = 'lbfgs'))
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
pipe.fit(X_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False)),
                ('logisticregression', LogisticRegression())])


final = pipe.score(X_test, y_test)
print('Accuracy with No L2 Penalty')
print('Accuracy with Standardization and Cross Validation: %.3f' % (mean(n_scores)))
print('Accuarcy with Standardization and not Cross Validation: %.3f' % (final))
#Creating Confusion Matrix
#Credit to https://realpython.com/logistic-regression-python/
confusion_matrix(y,pipe.predict(X))
cm = confusion_matrix(y, pipe.predict(X))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


#Creating Output from Models
print(classification_report(y, pipe.predict(X)))

#With L2 Penalty
pipe = make_pipeline(StandardScaler(), LogisticRegression(multi_class = 'multinomial',solver = 'lbfgs', penalty = 'l2',C=.05))
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty = 'l2',C=.05)
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
pipe.fit(X_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False)),
                ('logisticregression', LogisticRegression())])


final = pipe.score(X_test, y_test)
print('Accuracy with L2 Penalty')
print('Accuracy with Standardization and Cross Validation: %.3f' % (mean(n_scores)))
print('Accuarcy with Standardization and not Cross Validation: %.3f' % (final))

#Creating Confusion Matrix
#Credit to https://realpython.com/logistic-regression-python/
confusion_matrix(y,pipe.predict(X))
cm = confusion_matrix(y, pipe.predict(X))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Distance Group 0', 'Predicted Distance Group 1'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Distance Group 0', 'Actual Distance Group 1'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


#Creating Output from Models
print(classification_report(y, pipe.predict(X)))



# From the output above we can see the logistic regression for classifying Distance group is highly accurate. Looking at the Confusion Matrix below, we see there is a difference not only with the implementation of K-Fold Cross Validation, but also with the addition of the L2 Penalty. We are more confident in the models without the L2 penalty as these classify more accurately as well as provide enough room to provide post model adjustments if need be. 
#This is our first attempt at modeling with our Flight dataset and in the future will bring about more robust methods to our modeling. Upon further review of the model, there are many more directions available for us to entertain, but would require a more in depth understanding of modeling within Pandas.
#References

# 6.3. Preprocessing data. (n.d.). Scikit-learn. https://scikit-learn.org/stable/modules/preprocessing.html
#Real Python. (2022a, September 1). Logistic Regression in Python. https://realpython.com/logistic-regression-python/
#Machine Learning Mastery (n.d.). https://machinelearningmastery.com/multinomial-logistic-regression-with-python/