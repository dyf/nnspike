import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras.utils as ku

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import argparse

PAD = 50
model = km.Sequential([
        kl.Conv1D(filters=10, kernel_size=51, activation='relu', input_shape=(None,1)),
        kl.Conv1D(filters=10, kernel_size=51, activation='relu'),
    # kl.Conv1D(filters=10, kernel_size=21, activation='relu'),
        kl.Dense(4, activation='softmax'),
        ])

def loader(training_file):
    with h5py.File(training_file, "r") as f:
        cats = f["cats"].value
        patches = f["patches"].value

    cats = ku.to_categorical(cats)

    return cats, patches[:,:,np.newaxis]


def train_model(training_file, batch_size, epochs, output_file):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    cats, patches = loader(training_file)

    cats = cats[:,PAD:-PAD,:]

    print(cats.shape, patches.shape)
    model.fit(patches, cats, 
              batch_size=batch_size, epochs=epochs, 
              validation_split=.3,
              callbacks=[kc.ModelCheckpoint(output_file), kc.EarlyStopping()])


def vis_model(model_file, training_file, N):
    print(model_file, training_file, N)
    model = km.load_model(model_file)

    cats, patches = loader(training_file)
    idxs = np.random.randint(0, patches.shape[0], N)

    patches = patches[idxs,:,:]
    cats = cats[idxs,:,:]


    output = model.predict(patches)

    print("p", patches.shape)
    print("c", cats.shape)
    print("o", output.shape)

    for pi in range(N):
        plt.plot(patches[pi,PAD:-PAD,0], label='v')
        
        train_idxs = np.where(cats[pi,PAD:-PAD,1:] > 0)
        predict_idxs = np.where(output[pi,:,1:] > .5)
        
        plt.scatter(train_idxs[0], np.ones(len(train_idxs[0]))*0, label='train')
        plt.scatter(predict_idxs[0], np.ones(len(predict_idxs[0]))*(-0.1), label='predict')

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
        parser.add_argument("--output_file", default="cnn.weights")
        args = parser.parse_args(sys.argv[2:])

        train_model(**vars(args))
    elif command == "vis":
        parser.add_argument("--N", type=int, default=10)
        parser.add_argument("--model_file", default="cnn.weights") 
        parser.add_argument("--training_file", default="training_data.h5")
        args = parser.parse_args(sys.argv[2:])
        
        vis_model(**vars(args))

        
        

    

if __name__ == "__main__": main()
    

