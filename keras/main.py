import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import numpy as np
import h5py
import matplotlib.pyplot as plt

patch_size = 200
out_size = 160

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
        kl.Conv1D(filters=10, kernel_size=21, activation='relu', input_shape=(None,1)),
        kl.Conv1D(filters=40, kernel_size=21, activation='relu'),
        kl.Dense(4, activation='softmax'),
        ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def train_model():
    for cats, patches in loader():
        model.fit(patches, cats, 
                  batch_size=100, epochs=500, 
                  callbacks=[kc.ModelCheckpoint('output/model.h5')])


def vis_model():
    model = km.load_model('output/model.h5')

    with h5py.File('training_data.h5','r') as f:
        patches = f['patches'].value
        cats = f['cats'].value

    output = model.predict(patches[:,:,np.newaxis])

    print("p", patches.shape)
    print("c", cats.shape)
    print("o", output.shape)

    buffer = (patch_size - out_size) // 2

    for i in range(10):
        pi = np.random.randint(0,patches.shape[0])

        plt.plot(patches[pi,buffer:-buffer], label='v')
        
        for channel, label in zip((0,1,2), ("threshold", "peak", "trough")):
            c = cats[pi,channel,buffer:-buffer]
            o = output[pi,:,channel]

            plt.plot(c, label='train_'+label)
            plt.plot(o+0.1, label='output_'+label)

        plt.legend()
        plt.show()
    

#train_model()
vis_model()

