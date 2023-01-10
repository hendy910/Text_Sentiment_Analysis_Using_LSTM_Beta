# %%
import pandas as pd
import regex as re
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import json
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
import os
from keras.regularizers import L1L2, L1, L2
from keras.utils import plot_model

# %% 1. Data loading
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
df = pd.read_csv(URL)
# %% 2. Data Inspection
df.info()
df.head()
df.duplicated().sum()

# %%
df.duplicated().sum()
df.head()
df.info()
# %%
lengths = df['text'].str.len()
argmax = np.where(lengths == lengths.max())[0]
df.iloc[argmax]

# %% Check the data
print(df['text'][8])

# %% 3. Data Cleaning
# 1) Remove numbers
# 2) Remove HTML Tags
# 3) Remove punctuations
# 4) Change all to lowercase()

for index, data in enumerate(df['text']):
    df['text'][index] = re.sub('<.*?>','',data) # remove HTML Tags
    df['text'][index] = re.sub('[^a-zA-Z]',' ',df['text'][index]).lower()

# %%
# Recheck Data
print(df['text'][10])

# %% 4. Features Selection

category = df['category']
text = df['text']

# %% 5. Data Preprocessing

num_words = 5000 # find unique number of words in all the sentences
oov_token = 'Out of Vocab' # out of vocab

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

# to train the tokenizer 
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:50]))

#  to transform the text using tokenizer 
text = tokenizer.texts_to_sequences(text)

# %%
# Padding

padded_text = pad_sequences(text,maxlen=350,padding='pre',truncating='pre')

# %%
# One hot encoder
# To instantiate
ohe = OneHotEncoder(sparse=False)

category = ohe.fit_transform((category)[::,None])

# %%
# Train Test Split
# expand the dimension before feeding to train_test_split
padded_text = np.expand_dims(padded_text,axis=-1)

X_train,X_test,y_train,y_test = train_test_split(padded_text,category,test_size=0.2,random_state=123,shuffle=True)

# %% 6. Model Development
embedding_layer = 128

model = Sequential()
model.add(Embedding(num_words,embedding_layer))
model.add(LSTM(embedding_layer,return_sequences=True, kernel_regularizer='l2')) 
model.add(Dropout(0.3))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128, kernel_regularizer='l2'))
model.add(Dropout(0.3))
model.add(Dense(5,activation='softmax')) 
model.summary()


model.compile(optimizer= Adam(lr=0.0001), loss='categorical_crossentropy', metrics = ['acc'])
plot_model(model, show_shapes=True, show_dtype=True)

# %% TensorBoard Callbacks
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y&m%d-%H%M%S"))
ts_callback = TensorBoard(log_dir=LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True,)
history = model.fit(X_train,y_train,validation_data=(X_test,y_test), batch_size=64,epochs=50,callbacks=[ts_callback,es_callback])


# %% 7. Model Analysis

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training','validation'])
plt.show()


# %%
y_predicted = model.predict(X_test)

# %%
history.history.keys()
# %% Classification Report and Confusion Matrix
y_predicted = np.argmax(y_predicted,axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report (y_test,y_predicted))
print(confusion_matrix(y_test,y_predicted))

# %% Display Confusion Matrix 
cm = confusion_matrix(y_test,y_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()

"""
# %%
# Model Saving
# to save trained model
model.save('model.h5')

# %%
#  to save one hot encoder model
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)

# %%
# tokenizer 
token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json,f)

# %%
new_text = [NVIDIA‘s new flagship GPU has just launched. Revealed at CES 2023, the RTX 4070 Ti outperforms the previous gen flagship GPU — the RTX 3090 Ti — by three times while consuming half the power — all due to NVIDIA’s breakthrough Ada Lovelace architecture innovations and NVIDIA DLSS 3.
The GeForce RTX 4070 Ti features 7680 CUDA Cores and 12 GB of super-fast GDDR6X memory. Nvidia compares the RTX 4070 Ti to 12x the performance of the six-year-old GeForce GTX 1080 Ti. The Nvidia GeForce RTX 4070 Ti is available now for a price of $799 USD. Check out the CES presentation in the video above]

# Need to remove punctuations
for index, data in enumerate(new_text):
    new_text[index] = re.sub('<.*?>','',data) # remove HTML Tags
    new_text[index] = re.sub('[^a-zA-Z]',' ',new_text[index]).lower() # remove punctuations,numbers,lower

new_text = tokenizer.texts_to_sequences(new_text)
padded_new_text = pad_sequences(new_text,maxlen=200,padding='post',truncating='post')

output = model.predict(padded_new_text) # for predict model only recognise number, so use tokenizer

print(ohe.get_feature_names_out())
print(output)
print(ohe.inverse_transform(output))    
# %%
"""