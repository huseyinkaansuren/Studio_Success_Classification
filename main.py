import pulldata
import datapreprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



pd.set_option("display.max_columns", None)
# pulldata.pull_data()
df = pd.read_csv("Data.csv")

print(df.head())

studio_df = datapreprocessing.preprocessing(df)

print(studio_df.describe())
print(studio_df.info())
print(studio_df.corr())

sns.catplot(x="Score", data=studio_df)
# sns.catplot(x="Ranked", data=studio_df)
# sns.catplot(x="Popularity", data=studio_df)
plt.show()

x = studio_df.iloc[:,1:4]
y = studio_df.iloc[:, 5:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



"""
from sklearn.linear_model import LogisticRegression
log_r = LogisticRegression(random_state=0)
log_r.fit(X_train, y_train)

y_pred = log_r.predict(X_test)
"""


#Two algorithms are working too with %100 success ***

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)