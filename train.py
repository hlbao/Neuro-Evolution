import sys, os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.models import model_from_json
import logging


# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_class():
    """Retrieve the dataset and process the data."""
    
    nb_features = 64
    nb_classes = 7
    batch_size = 64
    width, height = 48, 48
    input_shape=(width, height, 1)
    epochs     = 100
    x = np.load('./fdataX.npy')
    y = np.load('./flabels.npy')
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    #for xx in range(10):
    #    plt.figure(xx)
    #    plt.imshow(x[xx].reshape((48, 48)), interpolation='none', cmap='gray')
    #plt.show()
    #splitting into training testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    #saving the test samples to be used later
    np.save('modXtest', x_test)
    np.save('modytest', y_test)
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

#Compliling the model 

def compile_model_mlp(network, nb_classes, input_shape):
    
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def compile_model_cnn(genome, nb_classes, input_shape):
    
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(0,nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(nb_neurons[i], kernel_size = (3, 3), activation = activation, padding='same', input_shape = input_shape))
        else:
            model.add(Conv2D(nb_neurons[i], kernel_size = (3, 3), activation = activation))
        
        if i < 2: 
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(nb_neurons[len(nb_neurons) - 1], activation = activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model



def train_and_score(network, dataset):
   
    nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_class()
    
    model = compile_model_mlp(network, nb_classes, input_shape)
    #  model = compile_model_cnn(network, nb_classes, input_shape)

    model.fit(np.array(x_train), np.array(y_train),
              batch_size=batch_size,
              epochs=epochs
              verbose=0,
              #verbose=1,
              validation_data=(np.array(x_test), np.array(y_test)),
              callbacks=[early_stopper]#,shuffle=True
             )

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
