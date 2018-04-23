import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import argparse

patch_size = 1000
out_size = 960

model = km.Sequential([
        kl.Conv1D(filters=10, kernel_size=21, activation='relu', input_shape=(None,1)),
        kl.Conv1D(filters=40, kernel_size=21, activation='relu'),
        kl.Dense(4, activation='softmax'),
        ])

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


def train_model(batch_size, epochs, output_file):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    for cats, patches in loader():
        model.fit(patches, cats, 
                  batch_size=batch_size, epochs=epochs, 
                  callbacks=[kc.ModelCheckpoint(output_file)])


def vis_model(model_file, training_file, N):
    print(model_file, training_file, N)
    model = km.load_model(model_file)

    with h5py.File(training_file, 'r') as f:
        patches = f['patches'].value
        cats = f['cats'].value

    output = model.predict(patches[:,:,np.newaxis])

    print("p", patches.shape)
    print("c", cats.shape)
    print("o", output.shape)

    buffer = (patch_size - out_size) // 2

    for i in range(N):
        pi = np.random.randint(0,patches.shape[0])

        plt.plot(patches[pi,buffer:-buffer], label='v')
        
        for channel, label in zip((0,1,2), ("threshold", "peak", "trough")):
            c = cats[pi,channel,buffer:-buffer]
            o = output[pi,:,channel]

            plt.plot(c, label='train_'+label)
            plt.plot(o+0.1, label='output_'+label)

        plt.legend()
        plt.show()
        plt.close()
    

def main():
    command = sys.argv[1]

    parser = argparse.ArgumentParser()

    if command == "train":
        parser.add_argument("--batch_size", type=int, default=100)
        parser.add_argument("--epochs", type=int, default=500)
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
    

