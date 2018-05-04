import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras.utils as ku

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import argparse

out_size = 440

model = km.Sequential([
        kl.Conv1D(filters=10, kernel_size=31, activation='relu', input_shape=(None,1)),
        kl.Conv1D(filters=10, kernel_size=31, activation='relu'),
    # kl.Conv1D(filters=10, kernel_size=21, activation='relu'),
        kl.Dense(4, activation='softmax'),
        ])

def loader(training_file):
    with h5py.File(training_file, "r") as f:
        cats = f["cats"].value
        patches = f["patches"].value

    # peaks only
    #cats = cats == 2
    #for i in np.random.randint(0, patches.shape[0], 5):
    #    plt.plot(patches[i])
    #    plt.plot(cats[i])
    #    plt.show()
    #    plt.close()

    cats = ku.to_categorical(cats)
        
    n_samples = patches.shape[0]

    s = patches.shape
    
    border = (cats.shape[1] - out_size) // 2

  
    return cats[:,border:-border], patches[:,:,np.newaxis]


def train_model(training_file, batch_size, epochs, output_file):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    cats, patches = loader(training_file)
    print(cats.shape, patches.shape)
    model.fit(patches, cats, 
              batch_size=batch_size, epochs=epochs, 
              validation_split=.3,
              callbacks=[kc.ModelCheckpoint(output_file)])


def vis_model(model_file, training_file, N):
    print(model_file, training_file, N)
    model = km.load_model(model_file)

    cats, patches = loader(training_file)
    pad = (patches.shape[1] - out_size) // 2
    
    idxs = np.random.randint(0, patches.shape[0], N)

    patches = patches[idxs,:,:]
    cats = cats[idxs,:,:]


    output = model.predict(patches)

    print("p", patches.shape)
    print("c", cats.shape)
    print("o", output.shape)

    for pi in range(N):
        plt.plot(patches[pi,pad:-pad,0], label='v')
        plt.plot(cats[pi,:,1:].sum(axis=1), label="train")
        plt.plot(-output[pi,:,1:].sum(axis=1), label="predict")
        plt.legend()
        plt.show()
        plt.close()
    

def main():
    command = sys.argv[1]

    parser = argparse.ArgumentParser()

    if command == "train":
        parser.add_argument("--training_file", default="training_data.h5")
        parser.add_argument("--batch_size", type=int, default=100)
        parser.add_argument("--epochs", type=int, default=1500)
        parser.add_argument("--output_file", default="output/model.h5")
        args = parser.parse_args(sys.argv[2:])

        train_model(**vars(args))
    elif command == "vis":
        parser.add_argument("--N", type=int, default=10)
        parser.add_argument("--model_file", default="output/model.h5") 
        parser.add_argument("--training_file", default="training_data.h5")
        args = parser.parse_args(sys.argv[2:])
        
        vis_model(**vars(args))

        
        

    

if __name__ == "__main__": main()
    

