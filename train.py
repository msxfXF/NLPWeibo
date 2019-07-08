import numpy as np
import random
import word2vec
import jieba
from keras import Sequential
from keras.layers import LSTM,Bidirectional,Activation,Dense,Flatten
from keras_preprocessing import sequence
w2v = word2vec.load("D:\Baidu Net Disk Downloads\sgns.weibo.word\Chinese-Word-Vectors-master\evaluation\sgns.weibo.word.txt")

# list = []
strs = []
strs_label = []
with open("data/train_0.txt","r",encoding="utf-8") as f:
	lines = f.readlines()
	for line in lines:
		strs.append(line)
		strs_label.append(0)
with open("data/train_1.txt",encoding="utf-8") as f:
	lines = f.readlines()
	for line in lines:
		strs.append(line)
		strs_label.append(1)
shuffle_index = np.random.permutation(np.arange(len(strs)))
strs = np.array(strs)[shuffle_index]
y = np.array(strs_label)[shuffle_index]

x = np.zeros(shape=(len(strs),128,300), dtype=np.float32)
for i,str in enumerate(strs):
	res_cuts = jieba.cut(str[:128])
	for j,res_cut in enumerate(res_cuts):
		if res_cut in w2v:
			x[i,j,:] = w2v[res_cut]

model = Sequential()
model.add(Bidirectional(LSTM(64,dropout=0.2,return_sequences=True),input_shape=(128,300)))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
print(model.summary())

model.fit(x,y,validation_split=0.1,batch_size=32, epochs=3)
model.save('bilstm.h5')