import pandas as pd
import matplotlib.pyplot as plt

from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix

df= pd.read_csv('logatta.csv')
categories= df[['BusinessTravel','MaritalStatus','OverTime', 'Gender']]
encoder= OrdinalEncoder()
df_encoded= encoder.fit_transform(categories)
df_encoded= pd.DataFrame(df_encoded, columns= categories.columns)
df[categories.columns]= df_encoded

X= df.drop('accepted for the interview', axis=1)
y= df['accepted for the interview']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
acceptance_counts = y.value_counts()
acceptance_labels = ['Not Accepted', 'Accepted']

#grouped_data= df.groupby(['Gender'])['accepted for the interview'].value_counts(normalize=True).unstack()
#plt.figure(figsize=(12, 6))
#grouped_data.plot(kind='bar', stacked=False, color=['skyblue', 'green'], width=0.8)
#plt.title("Acceptance Rates")
#plt.xlabel("Male      Female")
#plt.ylabel("Acceptance")
#plt.xticks(rotation=45, ha='right')
#plt.legend(title='Acceptance Status', labels=acceptance_labels)
#plt.tight_layout()
#plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred_nb = naive_bayes_model.predict(X_test)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

models = [logistic_model, naive_bayes_model, knn_model]
for model in models:
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)
  matrix = confusion_matrix(y_test, y_pred)
  print(f"Model: {model.__class__.__name__}")
  print(f"Accuracy: {accuracy:.2f}")
  print(f"F1 Score: {f1:.2f}")
  print(class_report)
  print()
  print(matrix)

