import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('./fer2013.csv')

width, height = 48, 48

datapoints = data['pixels'].tolist()

#getting features for training
x = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    x.append(xx.astype('float32'))

x = np.asarray(x)
x = np.expand_dims(x, -1)

#getting labels for training
y = pd.get_dummies(data['emotion']).as_matrix()

#storing them using numpy
np.save('fdataX', x)
np.save('flabels', y)

print("Preprocessing Done")
print("Number of Features: "+str(len(x[0])))
print("Number of Labels: "+ str(len(y[0])))
print("Number of examples in dataset:"+str(len(x)))
print("x,y stored in fdataX.npy and flabels.npy respectively")
