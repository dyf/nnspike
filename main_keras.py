import keras.models as km
import keras.layers as kl
import numpy as np
import train 
import h5py
import matplotlib.pyplot as plt

patch_size = 200
out_size = 170

def loader():
    with h5py.File("training_data.h5", "r") as f:
        cats = f["cats"].value
        patches = f["patches"].value
        
    n_samples = patches.shape[0]

    s = patches.shape
    
    cats = cats[:,0,:] # just threshold
    
    border = (cats.shape[1] - out_size) // 2
    
    cats = cats[:,border:-border]

    yield cats[:,:,np.newaxis], patches[:,:,np.newaxis]


model = km.Sequential([
        kl.Conv1D(filters=10, kernel_size=11, activation='relu', input_shape=(None,1)),
        kl.Conv1D(filters=10, kernel_size=11, activation='relu', input_shape=(None,1)),
        #kl.Conv1D(filters=20, kernel_size=25, activation='relu'),
        kl.Conv1D(filters=1, kernel_size=11, activation='sigmoid'),
        #kl.Dense((200,1), input_shape=(None,1))
        #kl.Dense(out_size, activation='sigmoid')
        ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


def train_model():
    for cats, patches in loader():
        print(cats.shape, patches.shape)
        model.fit(patches, cats, batch_size=100, epochs=1000)

    model.save('output/model_thresh.h5')


train_model()

