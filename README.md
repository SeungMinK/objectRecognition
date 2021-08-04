(20.11.03~4)[SW Week]HBC-5 딥러닝 해커톤<br>
5개 건물 사진을 촬영/수집하고, 수집한 사진을 분류하는 딥러닝 모델 개발 대회

~~~
python 버전을 맞추기 위한 코드
In [ ]:
!pip uninstall tensorflow
!pip install tensorflow==1.15
In [ ]:
import tensorflow as tf 
print(tf.__version__)
In [ ]:
pip uninstall keras
In [ ]:
pip install keras==2.2.4
In [ ]:
!pip uninstall numpy
In [ ]:
!pip install numpy==1.16.1
코드 시작
In [ ]:
"""
from PIL import Image,ImageFilter,ImageEnhance
import os, glob, numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split

caltech_dir = "/content/drive/My Drive/trainig" #트레이닝 데이터 경로
categories = ["0", "1", "2", "3","4"] #라벨 이름
nb_classes = len(categories) 

#이미지 resize 크기 설정
image_w = 64 
image_h = 64


X = []
y = []


for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.*")
    print(cat, " 파일 길이 : ", len(files))
    

    gaussianBlur = ImageFilter.GaussianBlur(5)#가우시안 필터
    contour = ImageFilter.CONTOUR()#윤곽선 적용
    enhance = ImageFilter.EDGE_ENHANCE()#엣지
    mean = ImageFilter.MedianFilter()
    max = ImageFilter.MaxFilter()
    min = ImageFilter.MinFilter()
    for i, f in enumerate(files):
        
        img = Image.open(f)#이미지 불러오기
        img = img.convert("LA")#2채널 회색조로 변형

        img = img.resize((image_w, image_h))#이미지 resize
        #img = img.filter(gaussianBlur)#가우시안필터적용
        #img = img.filter(contour)
        img = img.filter(enhance)
        img = img.filter(max)
        #img = img.filter(mean)


        data = np.asarray(img)#img를 np형태로 저장

        X.append(data)#appned로 데이터 이어 붙이기
        y.append(label)#라벨 이어붙이기

        if i % 700 == 0:
            print(cat, " : ", f)
      
X = np.array(X)
y = np.array(y)


img.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)
xy = (X_train, X_test, y_train, y_test)
np.save("/content/drive/My Drive/numpy_data/multi_image_data.npy", xy)

print("이미지 총 갯수 : ", len(y))
"""
In [ ]:
"""
import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

X_train, X_test, y_train, y_test = np.load('/content/drive/My Drive/numpy_data/multi_image_data.npy')
print(X_train.shape)
print(X_train.shape[0])
"""
In [ ]:
"""
categories = ["0", "1", "2", "3", "4"]
nb_classes = len(categories)

#일반화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255
"""
In [ ]:
"""
with K.tf_ops.device('/device:GPU:0'):#gpu 사용
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu')) #2d convolution 적용 activition은 relu 사용
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3,3), padding="same", activation='relu')) #2d convolution 적용 activition은 relu 사용
    model.add(MaxPooling2D(pool_size=(2,2))) #max pooling 필터 영역별 최고값 조절
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = './model'
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    model_path = model_dir + '/multi_img_classification.model'
    checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    """
In [ ]:
#history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
In [ ]:
#print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]*100))
In [ ]:
"""
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()
"""
In [ ]:
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
In [ ]:
from PIL import Image,ImageFilter,ImageEnhance
import os, glob, numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split


image_w = 64
image_h = 64
categories = ["0", "1", "2", "3", "4"]
nb_classes = len(categories)


enhance = ImageFilter.EDGE_ENHANCE()#엣지
max = ImageFilter.MaxFilter()

#gaussianBlur = ImageFilter.GaussianBlur(5)
X = []
y = []
filenames = []


for building_index in categories:
  idx = int(building_index)
  print("Reading building: ", idx)

  caltech_dir = "/content/drive/My Drive/SW-WEEK/HBC-5-2020/FinalTestDataset" + '/' + building_index
  print(caltech_dir)

  #files = glob.glob(caltech_dir+"/*/*.*")
  files = glob.glob(caltech_dir+"/*.*")

  for i, f in enumerate(files):
      img = Image.open(f)
      img = img.convert("LA")   
      img = img.filter(enhance)
      img = img.filter(max)
      img = img.resize((image_w, image_h))
      data = np.asarray(img)
      filenames.append(f)
      X.append(data)

      #one-hot 돌리기.
      label = [0 for i in range(nb_classes)]
      label[idx] = 1
      y.append(label)#라벨 이어붙이기


X_train1, X_test1, y_train1, y_test1 = train_test_split(np.array(X), np.array(y))
# X_train2, X_test2, y_train2, y_test2 = train_test_split(np.array(X), np.array(y))
# xy = (X_train, X_test, y_train, y_test)
# np.save("/content/drive/My Drive/numpy_data/multi_image_data.npy", xy)
# 불러오고
# 일반화
X_train1 = np.array(X_train1).astype(float) / 255
X_test1 = np.array(X_test1).astype(float) / 255
#X_train2 = np.array(X_train2).astype(float) / 255
#X_test2 = np.array(X_test2).astype(float) / 255

print('done')
Reading building:  0
/content/drive/My Drive/SW-WEEK/HBC-5-2020/FinalTestDataset/0
Reading building:  1
/content/drive/My Drive/SW-WEEK/HBC-5-2020/FinalTestDataset/1
Reading building:  2
/content/drive/My Drive/SW-WEEK/HBC-5-2020/FinalTestDataset/2
Reading building:  3
/content/drive/My Drive/SW-WEEK/HBC-5-2020/FinalTestDataset/3
Reading building:  4
/content/drive/My Drive/SW-WEEK/HBC-5-2020/FinalTestDataset/4
done
In [ ]:
model = load_model('/content/drive/My Drive/SW-WEEK/HBC-5-2020/Team-1-Model/multi_img_classification.model')

# 우리가 가진 데이터 
#X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y)

loss1, acc1 = model.evaluate(X_train1, y_train1)
loss2, acc2 = model.evaluate(X_test1, y_test1)

size1 = len(X_train1)
size2 = len(X_test1)
size_sum = size1 + size2
final_accuracy = (acc1 * size1 / size_sum) + (acc2 * size2 / size_sum)

print('Final accuracy : ', (acc1 + acc2) / 2.0 )
59/59 [==============================] - 3s 56ms/step - loss: 7.0180 - accuracy: 0.3547
20/20 [==============================] - 1s 52ms/step - loss: 6.6255 - accuracy: 0.3696
Final accuracy :  0.36213333904743195
In [ ]:
print(acc1, acc2)

size1 = len(X_train1)
print(size1)

size2 = len(X_test1)
print(size2)
0.35466668009757996 0.36959999799728394
1875
625

~~~
