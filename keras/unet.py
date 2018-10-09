import numpy as np
import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras.utils as ku
import keras.optimizers as ko
import h5py
import matplotlib.pyplot as plt

def unet(input_shape, output_cats):
    inputs = kl.Input(input_shape)
    c1 = kl.Conv1D(8, 9, activation='relu', padding='same')(inputs)
    c1 = kl.Conv1D(8, 9, activation='relu', padding='same')(c1)
    p1 = kl.MaxPooling1D(pool_size=2)(c1)

    c2 = kl.Conv1D(16, 9, activation='relu', padding='same')(p1)
    c2 = kl.Conv1D(16, 9, activation='relu', padding='same')(c2)
    p2 = kl.MaxPooling1D(pool_size=2)(c2)

    c3 = kl.Conv1D(32, 9, activation='relu', padding='same')(p2)
    c3 = kl.Conv1D(32, 9, activation='relu', padding='same')(c3)
    p3 = kl.MaxPooling1D(pool_size=2)(c3)

    c4 = kl.Conv1D(64, 9, activation='relu', padding='same')(p3)
    c4 = kl.Conv1D(64, 9, activation='relu', padding='same')(c4)
    d4 = kl.Dropout(0.5)(c4)
    p4 = kl.MaxPooling1D(pool_size=2)(d4)

    c5 = kl.Conv1D(128, 9, activation='relu', padding='same')(p4)
    c5 = kl.Conv1D(128, 9, activation='relu', padding='same')(c5)
    d5 = kl.Dropout(0.5)(c5)

    u6 = kl.Conv1D(64, 4, activation='relu', padding='same')(kl.UpSampling1D(size=2)(d5))
    m6 = kl.Concatenate(axis=2)([d4,u6])
    c6 = kl.Conv1D(64, 9, activation = 'relu', padding = 'same')(m6)
    c6 = kl.Conv1D(64, 9, activation = 'relu', padding = 'same')(c6)

    u7 = kl.Conv1D(32, 4, activation='relu', padding='same')(kl.UpSampling1D(size=2)(c6))
    m7 = kl.Concatenate(axis=2)([c3,u7])
    c7 = kl.Conv1D(32, 9, activation = 'relu', padding = 'same')(m7)
    c7 = kl.Conv1D(32, 9, activation = 'relu', padding = 'same')(c7)

    u8 = kl.Conv1D(16, 4, activation='relu', padding='same')(kl.UpSampling1D(size=2)(c7))
    m8 = kl.Concatenate(axis=2)([c2,u8])
    c8 = kl.Conv1D(16, 9, activation = 'relu', padding = 'same')(m8)
    c8 = kl.Conv1D(16, 9, activation = 'relu', padding = 'same')(c8)

    u9 = kl.Conv1D(8, 4, activation='relu', padding='same')(kl.UpSampling1D(size=2)(c8))
    m9 = kl.Concatenate(axis=2)([c1,u9])
    c9 = kl.Conv1D(8, 9, activation = 'relu', padding = 'same')(m9)
    c9 = kl.Conv1D(8, 9, activation = 'relu', padding = 'same')(c9)

    c10 = kl.Dense(output_cats, activation = 'softmax')(c9)

    model = km.Model(inputs=inputs, output=c10)

    model.compile(optimizer = ko.Adam(lr = 1e-4), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])

    print(model.summary())
    return model

def vis(N=10):
    model = km.load_model("unet.weights")
    print(model.summary())
    cats, patches = load()
    #pad = (patches.shape[1] - out_size) // 2
    
    idxs = np.random.randint(0, patches.shape[0], N)

    patches = patches[idxs,:,:]
    cats = cats[idxs,:,:]


    output = model.predict(patches)

    print("p", patches.shape)
    print("c", cats.shape)
    print("o", output.shape)

    print(output.shape)

    for pi in range(N):
        plt.plot(patches[pi,:,0], label='v')
        plt.plot(output[pi,:,0], label='c0')
        plt.plot(output[pi,:,1], label='c1')
        plt.plot(output[pi,:,2], label='c2')
        plt.plot(output[pi,:,3], label='c3')
        
        #train_idxs = np.where(cats[pi,:,1:] > 0)
        #predict_idxs = np.where(output[pi,:,1:] > .5)
        
        #plt.scatter(train_idxs[0], np.ones(len(train_idxs[0]))*0, label='train')
        #plt.scatter(predict_idxs[0], np.ones(len(predict_idxs[0]))*(-0.1), label='predict')

        plt.legend()
        plt.show()
        plt.close()

def load():
    with h5py.File("training_data.h5", "r") as f:
        cats = f['cats'][:,:4096]
        patches = f['patches'][:,:4096]

    cats = ku.to_categorical(cats)
    patches = patches[:,:,np.newaxis]

    return cats, patches

def train():
    cats, patches = load()

    batch_size=100
    epochs=100
    model = unet(patches.shape[1:], cats.shape[-1])
    output_file = "unet.weights"
    
    model.fit(patches, cats, 
              batch_size=batch_size, epochs=epochs, 
              validation_split=.3,
              callbacks=[kc.ModelCheckpoint(output_file), kc.EarlyStopping()])

#train()
vis()
