import pandas as pd

data = pd.read_csv('GPA.csv') #pandas로 엑셀파일 불러오기, 전처리 과정, data는 데이터프레임
#print(data.isnull().sum())#빈 데이터를 검색
data = data.dropna() #빈 행이 있다면 제거
#data.fillna(100) #빈 행을 100으로 채워라. 이건 내가 임의 설정
y_data = data['admit'].values # admit 세로열에 있는 데이터 출력
x_data = []
for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank'] ]) #x_data에 추가하기


import tensorflow as tf
import numpy as np

model  = tf.keras.models.Sequential([
    #딥러닝 모델 디자인
    tf.keras.layers.Dense(64, activation='tanh'),
    # hidden layer 갯수, ()는 노드 갯수로 실험으로 파악해야함
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    # activation='sigmoid' 는 0과 1사이에서의 확률을 뱉음
])

model. compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(x_data),np.array(y_data),epochs = 1000)
#x 정답 예측에 필요한 인풋, y실제 정답데이터, 모델 학습

#예측
predict = model.predict([[750, 3.70, 3], [400, 2.2, 1]]) #2명분 학습
print(predict)