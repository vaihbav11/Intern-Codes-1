#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df=pd.read_csv("IRIS.csv")


# In[6]:


df.info()
df.head(5)


# In[7]:


import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn._oldcore")
pair_plot = sns.pairplot(df, hue="species", diag_kind="auto", palette='pink')
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()


# In[8]:


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.boxplot(df['sepal_width'])
plt.title('Boxplot of Sepal Width')
plt.ylabel('Sepal Width')

plt.subplot(2, 2, 2)
plt.boxplot(df['sepal_length'])
plt.title('Boxplot of Sepal Length')
plt.ylabel('Sepal Length')

plt.subplot(2, 2, 3)
plt.boxplot(df['petal_length'])
plt.title('Boxplot of Petal Length')
plt.ylabel('Petal Length')

plt.subplot(2, 2, 4)
plt.boxplot(df['petal_width'])
plt.title('Boxplot of Petal Width')
plt.ylabel('Petal Width')

plt.tight_layout()
plt.show()


# In[9]:


#SELECT KBEST FEATURES METHOD
X1=df.drop(columns='species',axis=1)
Y1=df['species']
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3)
mutual_info=mutual_info_classif(X1_train,Y1_train)
mutual_info=pd.Series(mutual_info)
mutual_info.index=X1_train.columns
mutual_info.sort_values(ascending=False).plot.bar(figsize=(12,10),color='purple')


# In[10]:


from sklearn.feature_selection import SelectKBest
cols=SelectKBest(mutual_info_classif,k=3)
cols.fit(X1_train,Y1_train)
X1_train.columns[cols.get_support()]


# In[11]:


encode=LabelEncoder()
df['species']=encode.fit_transform(df['species'])


# In[12]:


cor=df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[13]:


x_selected=df[['sepal_length','petal_width']]
y_target=df['species']
x_train,x_test,y_train,y_test=train_test_split(x_selected,y_target,test_size=0.3)


# In[14]:


param_grid = {
    'n_estimators': [10,20,40,60,80,100],
    'min_samples_split': [3,5,7,9],
    'max_features': ['sqrt','log2'],
    'max_depth': [3,5,7,9],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1,error_score='raise')

grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Score: {best_score}")

rf_best = grid_search.best_estimator_

y_pred = rf_best.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {round(accuracy, 2)}')
print('Classification Report:')
print(class_report)


# In[15]:


from sklearn.ensemble import RandomForestClassifier

# Now you can use RandomForestClassifier
model = RandomForestClassifier()


# In[16]:


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'], yticklabels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


# In[17]:


pred_data = pd.DataFrame([[1.4, 1.2]], columns=x_selected.columns)

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='.*does not have valid feature names.*')
    pred_new = rf_best.predict(pred_data)


if pred_new[0] == 0:
    print("Iris-setosa")
elif pred_new[0] == 1:
    print("Iris-versicolor")
else:
    print("Iris-virginica")


# In[ ]:




