import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing, model_selection
import pandas as pd
import numpy as np

data_set=pd.read_csv('/home/ankit/data set/diabetes.csv')

datax=data_set.loc[:,'Pregnancies':'Age']
datay=data_set.loc[:,'Outcome']
print (datax.shape)
print (datay.shape)
datax=preprocessing.scale(datax)

train_x, test_x, train_y, test_y=model_selection.train_test_split(datax,datay,test_size=0.2)

train_y=keras.utils.to_categorical(train_y)
test_y=keras.utils.to_categorical(test_y)

model=Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_x,train_y,epochs=250,batch_size=25)


_,test_acc=model.evaluate(test_x,test_y,verbose=1)
print('Test Accuracy:',test_acc)



