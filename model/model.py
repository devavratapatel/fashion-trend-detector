import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from utils import *
from tensorflow.keras.initializers import glorot_uniform

def identity_block(x,f,filters,stage,block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1,F2,F3 = filters

    x_shortcut = x
    x = Conv2D(filters=F1, kernel_size=(1,1), strides = (1,1), padding='valid',name=conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = F2, kernel_size=(f,f), strides=(1,1),padding='same',name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',name = conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)

    x = Add()([x_shortcut,x])
    x = Activation('relu')(x)

    return x

def convolutional_block(x,f,filters,stage,block,s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    x_shortcut = x

    x = Conv2D(F1,(1,1),strides=(s,s),name=conv_name_base+'2a',padding='valid',kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b',padding='same', kernel_initializer = glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis = 3, name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c',padding='valid', kernel_initializer = glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis = 3, name = bn_name_base + '2c')(x)

    x_shortcut = Conv2D(F3,(1,1),strides=(s,s), name =  conv_name_base+'1',padding='valid',kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = BatchNormalization(axis=3,name=bn_name_base+'1')(x_shortcut)

    x = Add()([x_shortcut,x])
    x = Activation('relu')(x)

    return x

def model(input_shape = (64,64,3),classes=6):
    x_input = Input(input_shape)

    x = ZeroPadding2D((3,3))(x_input)

    x = Conv2D(64,(7,7),strides = (2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3,name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = convolutional_block(x,f=3,filters=[64,64,256], stage=2,block='a',s=1)
    x = identity_block(x,3,[64,64,256],stage=2,block='b')
    x = identity_block(x,3,[64,64,256],stage=2,block='c')

    x = convolutional_block(x,f=3,filters=[128,128,512],stage=3,block='a',s=2)
    x = identity_block(x,3,[128,128,512],stage=3,block='b')
    x = identity_block(x,3,[128,128,512], stage=3, block='c')
    x = identity_block(x,3,[128,128,512],stage=3,block='d')

    x = convolutional_block(x, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = convolutional_block(x, f = 3, filters =  [512, 512, 2048], stage = 5, block='a', s = 2)
    x = identity_block(x, 3,  [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3,  [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D(pool_size=(2,2),name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax',name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(x)

    model = Model(inputs = x_input, outputs = x, name = 'resnet')

    return model

model = model(input_shape=(64,64,3), classes=6)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = load_dataset()
x_train = x_train_orig/255.
x_test = x_test_orig/255.
y_train = convert_to_one_hot(y_train_orig,6).T
y_test = convert_to_one_hot(y_test_orig, 6).T

model.fit(x_train,y_train,epochs=20,batch_size=32)

preds = model.evaluate(x_test,y_test)
print("Loss:" + str(preds[0]))
print("Test Accuracy: " + str(preds[1]))

model.save('ResNet.keras')