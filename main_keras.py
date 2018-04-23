import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
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
    
    border = (cats.shape[2] - out_size) // 2
  
    cats = cats[:,:,border:-border]
    bg = cats.sum(axis=1) == 0
    ocats = np.zeros((cats.shape[0],  cats.shape[1]+1, cats.shape[2]))
    ocats[:,:-1,:] = cats
    ocats[:,-1,:] = bg
    ocats = np.swapaxes(ocats, 1, 2)

    yield ocats, patches[:,:,np.newaxis]


model = km.Sequential([
        kl.Conv1D(filters=10, kernel_size=11, activation='relu', input_shape=(None,1)),
        kl.Conv1D(filters=20, kernel_size=11, activation='relu', input_shape=(None,1)),
        kl.Conv1D(filters=4, kernel_size=11, activation='softmax'),
        #kl.Dense((200,1), input_shape=(None,1))
        ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def train_model():
    for cats, patches in loader():
        model.fit(patches, cats, 
                  batch_size=100, epochs=1000, 
                  callbacks=[kc.ModelCheckpoint('output/model.h5')])

    model.save('output/model_thresh.h5')


train_model()

