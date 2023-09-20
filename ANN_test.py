# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

# One Hot Encoding the "Geography" column, 010 100 001로. 상관관계가 없으므로 원핫으로.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
#위에서 Xsms 4열부터 시작, 그러므로 국가는 2열이 되므로 [1]
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
# ann으로 변수설정
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer with Dense class
# Dense units은 레이어 수로, 적절 수는 직접 돌려보고 숫자 조절
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# 0 또는 1중에서 판단하니 units = 1, 하지만 3개면 3개, 그 이상이면 그 숫자에 맞춰서.

# Part 3 - Training the ANN

# Compiling the ANN with method compile
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# adam은 확률적 경사하강최적화 가능
# 만약 0과 1이 아닌, 그 이상 분류한다면 로스함수로 'categorical_crossentropy'

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

# # """
# Homework:
# Use our ANN model to predict if the customer with the following informations will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000
# So, should we say goodbye to that customer?
#
# Solution:
# """
#
print(ann.predict(sc.transform([[1., 0., 0., 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
#50%보다 낮으면 해지할 확률이 낮다고 판단한다면, 우리는 0.5보다 높은 값만 받으면 된다. 0.5보다 안크면 0으로 처리, >0.5를 지우면 상세한 확률이 출력
# """
# Therefore, our ANN model predicts that this customer stays in the bank!
# Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
# Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
# """
#
# # Predicting the Test set results, logistic 범주형 데이터 예측
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#
# # Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))