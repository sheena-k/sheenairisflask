import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_excel("iris_python.xlsx")

#classifications = df["Classification"].value_counts()

#df["Classification"].unique()

y = df["Classification"]
x = df.drop(['Classification'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

model = lr.fit(x_train,y_train)

y_predictions = model.predict(x_test)

from sklearn.metrics import accuracy_score

y_predictions

print("accuracy =",accuracy_score(y_test,y_predictions))

import pickle

filename="savemodel.pickle"

with open(filename,"wb") as file:
  pickle.dump(model,file)





