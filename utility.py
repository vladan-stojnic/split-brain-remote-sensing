from skimage import transform, color
import numpy as np
from sklearn.neighbors import NearestNeighbors
import keras.backend as K
import os
from sklearn.externals import joblib
import pickle

EPS = 1e-8

class RandomCrop(object):
    def __init__(self, output_shape):
        super(RandomCrop, self).__init__()
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, output_shape)
        else:
            self.output_shape = output_shape

    def __call__(self, image):
        shape_x, shape_y, _ = image.shape
        if (shape_x<self.output_shape[0] or shape_y<self.output_shape[1]):
            raise ValueError('Image is smaller than desired ouput shape!')
        
        #Defines top left point of the crop
        dx = np.random.randint(0, shape_x-self.output_shape[0])
        dy = np.random.randint(0, shape_y-self.output_shape[1])

        return image[dx:dx+self.output_shape[0], dy:dy+self.output_shape[1], :]

class ResizeImage(object):
    def __init__(self, output_shape, preserve_range = False, anti_aliasing = True):
        super(ResizeImage, self).__init__()
        self.anti_aliasing = anti_aliasing
        self.preserve_range = preserve_range
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, output_shape)
        else:
            self.output_shape = output_shape

    def __call__(self, image):
        return transform.resize(image, self.output_shape, preserve_range = self.preserve_range, anti_aliasing = self.anti_aliasing)

class ConvertToLAB(object):
    def __init__(self, illuminant = 'D50'):
        super(ConvertToLAB, self).__init__()
        self.illuminant = illuminant

    def __call__(self, image):
        return color.rgb2lab(image, illuminant = self.illuminant)

class PrepareOutputZhang(object):
    def __init__(self, output_shape, anti_aliasing = True):
        super(PrepareOutputZhang, self).__init__()
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, output_shape)
        else:
            self.output_shape = output_shape
        self.anti_aliasing = anti_aliasing
        pts_in_hull = None
        with open('resources/pts_in_hull.npy', 'rb') as f:
            pts_in_hull = np.load(f)
        self.nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(pts_in_hull)

    def __call__(self, image):
        resizer = ResizeImage(self.output_shape, anti_aliasing = self.anti_aliasing)
        converter = ConvertToLAB()

        small = resizer(image)

        image = converter(image)
        small = converter(small)

        image[:, :, 0] -= 50

        output = np.zeros(self.output_shape + (2,), dtype = np.uint8)
        #Add one to have 100 bins
        output[:, :, 0] = np.digitize(small[:, :, 0], np.linspace(0, 100+EPS, num = 101)) - 1

        query = np.reshape(small[:, :, 1:3], (self.output_shape[0]*self.output_shape[1], 2))
        indices = self.nbrs.kneighbors(X = query, return_distance = False)
        output[:, :, 1] = np.reshape(indices, self.output_shape)

        return image, output

def pixelwise_accuracy(target, output):
    return 100*K.mean(K.mean(K.mean(K.equal(K.argmax(target, axis = 3), K.argmax(output, axis = 3)), axis = 2), axis = 1), axis = 0)

def read_data(dataset_path):
    image_paths = []
    labels = []
    current_class = -1
    classes = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for cls in classes:
        current_class += 1
        for image in os.listdir(cls):
            image_paths.append(os.path.join(cls, image))
            labels.append(current_class)

    return image_paths, labels

def get_parser():
    """Get parser object."""

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data",
                        dest="data",
                        help="Path to the dataset.",
                        default="/home/research/vladan/data/AID",
                        required=False)
    parser.add_argument("-w", "--weights",
                        dest="weights",
                        help="Path to the weights file.",
                        default="weights.h5",
                        required=False)
    parser.add_argument("-f", "--features",
                        dest="features",
                        help="Path to the features file.",
                        default="features.h5",
                        required=False)
    parser.add_argument("-r", "--results",
                        dest="results",
                        help="Path to the results file",
                        default="./results/results.txt",
                        required=False)
    parser.add_argument("--input-channels",
                        dest="input_channels",
                        help="Indices of input channels",
                        default='[0]',
                        required=False)
    parser.add_argument("--output-channels",
                        dest="output_channels",
                        help="Indices of output channels",
                        default='[1]',
                        required=False)
    parser.add_argument("--input-size",
                        dest="input_size",
                        help="Input image size",
                        default="(180, 180)",
                        required=False)
    parser.add_argument("--output-size",
                        dest="output_size",
                        help="Output image size",
                        default="(12, 12)",
                        required=False)
    parser.add_argument("--batch-size",
                        dest="batch_size",
                        help="Batch size",
                        default=64,
                        type = int,
                        required=False)
    parser.add_argument("--dataset-size",
                        dest="dataset_size",
                        help="Training dataset size",
                        default=30000,
                        type = int,
                        required=False)
    parser.add_argument("--num-classes",
                        dest="num_classes",
                        help="Training dataset sizeNumber of classes",
                        default=100,
                        type = int,
                        required=False)
    parser.add_argument("--algorithm",
                        dest="algorithm",
                        help="Type of training algorithm",
                        default="Zhang",
                        required=False)
    parser.add_argument("--output-type",
                        dest="output_type",
                        help="Type of output (single/double)",
                        default="single",
                        required=False)
    parser.add_argument("--num-pixels",
                        dest="num_pixels",
                        help="Number of pixels used",
                        default=100,
                        type = int,
                        required=False)
    parser.add_argument("--single-clusters",
                        dest="single_clusters",
                        help="Number of single clusters",
                        default=100,
                        type = int,
                        required=False)
    parser.add_argument("--double-clusters",
                        dest="double_clusters",
                        help="Number of double clusters",
                        default=313,
                        type = int,
                        required=False)
    parser.add_argument("--initializer",
                        dest="initializer",
                        help="Kernel initializer",
                        default='glorot_uniform',
                        required=False)

    return parser

class PrepareOutputPCA(object):
    def __init__(self, output_shape, single_clusters, double_clusters, anti_aliasing = True):
        super(PrepareOutputPCA, self).__init__()
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, output_shape)
        else:
            self.output_shape = output_shape
        self.anti_aliasing = anti_aliasing
        self.pca = joblib.load('resources/pca.pkl')
        with open('resources/bounds.pkl', 'rb') as f:
            self.bounds = pickle.load(f)

        with open(single_clusters, 'rb') as f:
            clusters = np.load(f)
            self.single_clusters = clusters['arr_0']

        with open(double_clusters, 'rb') as f:
            clusters = np.load(f)
            self.double_clusters = clusters['arr_0']

        self.single_clusters = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(self.single_clusters)
        self.double_clusters = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(self.double_clusters)

    def __call__(self, image):
        resizer = ResizeImage(self.output_shape, anti_aliasing = self.anti_aliasing)

        #Small is [0, 1] so we need to return it to [0, 255]
        small = resizer(image)*255

        dx, dy, dz = image.shape
        image = np.reshape(image, (dx*dy,dz))
        image = self.pca.transform(image)
        image = np.reshape(image, (dx,dy,dz))
        dx, dy, dz = small.shape
        small = np.reshape(small, (dx*dy,dz))
        small = self.pca.transform(small)
        small = np.reshape(small, (dx,dy,dz))

        image[:, :, 0] = np.clip(image[:, :, 0], self.bounds[0], self.bounds[1])
        small[:, :, 0] = np.clip(small[:, :, 0], self.bounds[0], self.bounds[1])

        image[:, :, 1] = np.clip(image[:, :, 1], self.bounds[2], self.bounds[3])
        small[:, :, 1] = np.clip(small[:, :, 1], self.bounds[2], self.bounds[3])

        image[:, :, 2] = np.clip(image[:, :, 2], self.bounds[4], self.bounds[5])
        small[:, :, 2] = np.clip(small[:, :, 2], self.bounds[4], self.bounds[5])

        output = np.zeros(self.output_shape + (2,), dtype = np.uint16)

        query = np.reshape(small[:, :, 0], (self.output_shape[0]*self.output_shape[1], 1))
        indices = self.single_clusters.kneighbors(X = query, return_distance = False)
        output[:, :, 0] = np.reshape(indices, self.output_shape)

        query = np.reshape(small[:, :, 1:3], (self.output_shape[0]*self.output_shape[1], 2))
        indices = self.double_clusters.kneighbors(X = query, return_distance = False)
        output[:, :, 1] = np.reshape(indices, self.output_shape)

        return image, output

class PrepareOutputLAB(object):
    def __init__(self, output_shape, single_clusters, double_clusters, anti_aliasing = True):
        super(PrepareOutputLAB, self).__init__()
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, output_shape)
        else:
            self.output_shape = output_shape
        self.anti_aliasing = anti_aliasing

        with open(single_clusters, 'rb') as f:
            clusters = np.load(f)
            self.single_clusters = clusters['arr_0']

        with open(double_clusters, 'rb') as f:
            clusters = np.load(f)
            self.double_clusters = clusters['arr_0']

        self.single_clusters = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(self.single_clusters)
        self.double_clusters = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(self.double_clusters)

    def __call__(self, image):
        resizer = ResizeImage(self.output_shape, anti_aliasing = self.anti_aliasing)
        converter = ConvertToLAB()

        small = resizer(image)

        image = converter(image)
        small = converter(small)

        image[:, :, 0] -= 50

        output = np.zeros(self.output_shape + (2,), dtype = np.uint16)

        query = np.reshape(small[:, :, 0], (self.output_shape[0]*self.output_shape[1], 1))
        indices = self.single_clusters.kneighbors(X = query, return_distance = False)
        output[:, :, 0] = np.reshape(indices, self.output_shape)

        query = np.reshape(small[:, :, 1:3], (self.output_shape[0]*self.output_shape[1], 2))
        indices = self.double_clusters.kneighbors(X = query, return_distance = False)
        output[:, :, 1] = np.reshape(indices, self.output_shape)

        return image, output
