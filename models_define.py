
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Reshape,Flatten,Activation, BatchNormalization

def get_Dense_model():

    model = Sequential([
        Dense(1024,activation='relu',input_shape=(81,)),
        Dense(1024,activation='relu'),
        Dense(1024,activation='relu'),
        Dense(81*9),
        Reshape((-1,9)),
        Activation('softmax')
    ])

    return model

def get_Conv_model():

    model = Sequential([
        Reshape((9,9,1),input_shape=(81,)),
        Conv2D(64,kernel_size=(3,3),activation = 'relu', padding='same'),
        BatchNormalization(),
        Conv2D(64,kernel_size=(3,3),activation = 'relu', padding='same'),
        BatchNormalization(),
        Conv2D(128,kernel_size=(1,1),activation = 'relu', padding='same'),
        Flatten(),
        Dense(81*9),
        Reshape((-1,9)),
        Activation('softmax')
    ])

    return model