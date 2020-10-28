import numpy as np
import keras
from skimage.io import imread

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_paths, input_size, output_size, input_channels, output_channels, num_classes, batch_size=32, processing_classes = None, output_function = None, shuffle=True):
        'Initialization'
        self.image_paths = image_paths
        self.input_size = input_size
        self.output_size = output_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.n_channels = len(self.input_channels)
        self.processing_classes = processing_classes
        self.output_function = output_function
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_images_temp = [self.image_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_images_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_images_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.input_size[0], self.input_size[1], self.n_channels))
        y = np.empty((self.batch_size, self.output_size[0], self.output_size[1]), dtype=np.uint16)

        # Generate data
        for i, image in enumerate(list_images_temp):
            # Store sample
            image = imread(image)
            for fnc in self.processing_classes:
                image = fnc(image)
            image, label = self.output_function(image)

            X[i, :, :, :] = image[:, :, self.input_channels]

            # Store class
            y[i, :, :] = label[:, :, self.output_channels].squeeze()

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)
