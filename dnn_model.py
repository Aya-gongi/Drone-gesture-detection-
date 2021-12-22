#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
TRAIN_DIR = 'C:/Users/user/Documents/drone_gesture_dataset/train/images'
TEST_DIR = 'C:/Users/user/Documents/drone_gesture_dataset/test/images'


# In[2]:


#create a function that extract the order from the image label 
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'follow': return [1, 0]
    elif word_label == 'land': return [0, 1]
    elif word_label =='take_off' : return [1, 1]
    
#create the training data 
def create_train_data():
    training_data = []
    IMG_SIZE = 50
    for img in tqdm(os.listdir(TRAIN_DIR)):
  
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
        try:
            img = cv2.resize(img, (50, 150,1), interpolation=cv2.INTER_AREA)
            print(img.shape)
        except:
            break
        height, width  = img.shape
        IMG_SIZE=(width,height)
        print(IMG_SIZE)
  
    print(img)   
    training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# In[3]:


create_train_data


# In[4]:


#crete the testing data
def process_test_data():
    testing_data = []
    #IMG_SIZE=50
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 150))
        testing_data.append([np.array(img), img_num])
          
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# In[5]:


train_data = create_train_data()
test_data = process_test_data()


# In[6]:


train_data = np.load('train_data.npy',allow_pickle=True)
test_data = np.load('test_data.npy',allow_pickle=True)


# In[7]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf


# In[8]:


#create the DNN model
LR = 1e-3
tf.compat.v1.reset_default_graph()
convnet = input_data(shape =[None, 50, 150, 1], name ='input')
  
convnet = conv_2d(convnet, 32, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
  
convnet = conv_2d(convnet, 64, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
  
convnet = conv_2d(convnet, 128, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
  
convnet = conv_2d(convnet, 64, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
  
convnet = conv_2d(convnet, 32, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
  
convnet = fully_connected(convnet, 1024, activation ='relu')
convnet = dropout(convnet, 0.8)
  
convnet = fully_connected(convnet, 2, activation ='softmax')
convnet = regression(convnet, optimizer ='adam', learning_rate = LR,
      loss ='categorical_crossentropy', name ='targets')
  
model = tflearn.DNN(convnet, tensorboard_dir ='log')
  


# In[9]:


train = train_data[:-500]
test = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1, 50, 150, 1)
Y = [i[1] for i in train]


# In[10]:


test_x = np.array([i[0] for i in test])#.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]


# In[11]:


#train the model
MODEL_NAME = 'dronegesturecontrol-{}-{}.model'.format(LR, '6conv-basic')
model.fit({'input': X}, {'targets': Y}, n_epoch = 5, 
    validation_set =({'input': test_x}, {'targets': test_y}), 
    snapshot_step = 500, show_metric = True, run_id = MODEL_NAME)
model.save(MODEL_NAME)


# In[12]:


#plot the result consisting on showing images with their control 
import matplotlib.pyplot as plt
test_data = np.load('test_data.npy',allow_pickle=True)
fig = plt.figure()
for num, data in enumerate(test_data[:20]):
      
    img_num = data[1]
    img_data = data[0]
      
    y = fig.add_subplot(4, 5, num + 1)
    orig = img_data
    data = img_data.reshape(50, 150, 1)
  
    model_out = model.predict([data])[0]
      
    if np.argmax(model_out) == 1: str_label ='follow'
    elif np.argmax(model_out) == 0: str_label ='land'
    else: str_label ='take_off'
    
    
    y.imshow(orig, cmap ='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


# In[13]:


data[0].shape


# In[14]:


#save the model on a h5 file
model.save('model.h5') 


# In[15]:


#Take a photo as an order and save it 
#retrain the model to predict the image order 
from PIL import Image  
import PIL 
import cv2
import time 
cam_port = 0
cam = VideoCapture(cam_port)
print('Make an order')
time.sleep(5)
result, image = cam.read()
 if result:
    imshow("Order", image)
    imwrite("C:/Users/user/Documents/drone_gesture_dataset/orders/Order.png", image)
    waitKey(0)
    time.sleep(2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            destroyWindow("Order")
else:
    print("No image detected. Please! try again")

image = Image.open(r'C:/Users/user/Documents/drone_gesture_dataset/orders/Order.png')  
img = cv2.imread('C:/Users/user/Documents/drone_gesture_dataset/orders/Order.png')
img = cv2.resize(img, (50 ,50))
img=img.reshape((-1, 50,150,1))
img.shape


# In[ ]:





# In[17]:


model_out_gest = model.predict(img)


# In[18]:


print(model_out_gest)


# In[35]:


img=img.reshape(50,150)
if np.argmax(model_out_gest) == 1: str_label ='follow'
elif np.argmax(model_out_gest) == 0: str_label ='land'
else: str_label ='take_off'

plt.imshow(img, cmap ='gray')
plt.title(str_label)
plt.show()


# In[ ]:




