import numpy as np
import sys
import os
import pickle
import random
from PIL import Image
DATADIR = '/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/train'
CATEGORIES = ["Dog", "Cat"]
train_batch=list(range(0, 12500))
val_batch=[]
x_train, y_train, x_val, y_val=[], [], [], []
vals=list(range(12500))
random.shuffle(vals)

for i in range(3125):
    train_batch.remove(vals[i])
    val_batch.append(vals[i])
for i in train_batch:
    print(i)
    path=os.path.join(DATADIR,  "cat." + str(i) + ".jpg")
    img=Image.open(path, 'r').convert("RGB")
    img=img.resize((200, 200))
    x_train.append(np.asarray(img))
    y_train.append(0)
for i in train_batch:
    print(i)
    path=os.path.join(DATADIR,  "dog." + str(i) + ".jpg")
    img=Image.open(path, 'r').convert("RGB")
    img=img.resize((200, 200))
    x_train.append(np.asarray(img))
    y_train.append(1)

x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
x_train_file = open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/x_train', "wb")
y_train_file = open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/y_train', "wb")
pickle.dump(x_train, x_train_file)
pickle.dump(y_train, y_train_file)
x_train_file.close()
y_train_file.close()

del x_train
del y_train
for i in val_batch:
    print(i)
    path=os.path.join(DATADIR,  "cat." + str(i) + ".jpg")
    img=Image.open(path, 'r').convert("RGB")
    img=img.resize((200, 200))
    x_val.append(np.asarray(img))
    y_val.append(0)
for i in val_batch:
    print(i)
    path=os.path.join(DATADIR,  "dog." + str(i) + ".jpg")
    img=Image.open(path, 'r').convert("RGB")
    img=img.resize((200, 200))
    x_val.append(np.asarray(img))
    y_val.append(1)

x_val=np.asarray(x_val)
y_val=np.asarray(y_val)
x_val_file = open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/x_val', "wb")
y_val_file = open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/y_val', "wb")
pickle.dump(x_val, x_val_file)
pickle.dump(y_val, y_val_file)
x_val_file.close()
y_val_file.close()
