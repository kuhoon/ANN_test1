# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Data.csv') #('.csv', header=None) can show first Row
X = dataset.iloc[:, :-1].values #iloc - sucht Info[Row, Column]
y = dataset.iloc[:, -1].values #kaufen oder nie, Ziel, last Column
print(X)
print(y)

# Taking care of missing data, replace mean Value with imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #imputer ist Instant fur SimpleImputer, np.nan = missing data, wir nuzten mit Mean value
imputer.fit(X[:, 1:3])
#.의 의미는 inputer에서 fit기능을 부름, fit은 숫자만 인덱싱
#python doesnt account last number, just 1 und 2 Column consider
X[:, 1:3] = imputer.transform(X[:, 1:3])
#nan tranfer to X
#imputer에 전환된 값은 다시 X[:, 1:3] 에 저장
print(X)

# Encoding categorical data
# Encoding the Independent Variable (입력값)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder #문자열을 100, 010, 001 등과같은 벡터로 변형해서 분류
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#ct는 오브젝트이름, 원하는데로 바꾸면 돼
#transformers=[(인코딩(인코딩할거니 인코더이름), 인코딩유형(클래스명), 열의인덱스)]
#remainder = 우리는 문자열인 국가 1열만 바꿀꺼고, 나머지열은 인코딩하면 안되니까 passthrough 1,2,3열모두 변환하지만, 1열만 백터화 labelencoder와의 차이점
X = np.array(ct.fit_transform(X))
#ct.fit_transform(X)할경우 넘파이 배열이 아니므로, np.array를 붙여준다. 자동으로 X로 업데이트
#출력을 하면 1.0 0.0 0.0 이런식의 통 백터로 문자열을 바꿔서 분류함
print(X)

# Encoding the Dependent Variable (출력값)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#우리는 결과값만 할거기 때문에 아무것도 입력안해도됨
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# splitsize = traing 80, test 20 무작위로 선택됨,
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling #평균 표춘편차
# standardisation : -3~3까지, 평균값과 표준편차로 (그외에는 평소에는 이게 더 좋음)
# Normalisation : -1~1Rkwl, 최소값과 최대최소차이값으로 (정규분포를 따른다면 N이 좋음)
# 오로지 숫자데이터에만 사용할 것
# 하지만 OneHotEncoder인코더같은 가변형에는 이미 문자열을 숫자로 바꿔서 0과 1에 있는데 설정하면, 스케일링되어버려서 원래 문자열이랑 매칭안되는 정보손실현상(여기선 국적 100, 010, 001 .... 등등)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# 3열부터 전부라는 뜻, 여기서 fit은 평균값고 표준편차를 가져오고, transform에서는 이 데이타로 SC를 계산
X_test[:, 3:] = sc.transform(X_test[:, 3:])
# train을 통해 이미 스케일링되어있으므로, fit없이 transform만 사용. 절때 한번 더 fit 하면 안돼!!!
print(X_train)
print(X_test)