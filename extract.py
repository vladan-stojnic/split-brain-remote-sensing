import utility
from ast import literal_eval
from generator import DataGenerator
from networks import AlexNet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

args = utility.get_parser().parse_args()

DATASET = args.data
BATCH_SIZE = args.batch_size

if args.algorithm == "Zhang":
    #Vratiti na 180, 180
    INPUT_SIZE = (227, 227)
    OUTPUT_SIZE = (12, 12)
    PROCESSING_CLASSES = [utility.ResizeImage(INPUT_SIZE)]
    OUTPUT_FUNCTION = utility.PrepareOutputZhang(OUTPUT_SIZE)
    if args.output_type == "single":
        INPUT_CHANNELS = [1, 2]
        OUTPUT_CAHNNELS = [0]
        NUM_CLASSES = 100
    elif args.output_type == "double":
        INPUT_CHANNELS = [0]
        OUTPUT_CAHNNELS = [1]
        NUM_CLASSES = 313
    else:
        raise ValueError("Invalid output type")
elif args.algorithm == 'PCAKmeans':
    INPUT_SIZE = (227, 227)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.ResizeImage(INPUT_SIZE, preserve_range = True)]
    OUTPUT_FUNCTION = utility.PrepareOutputPCA(OUTPUT_SIZE, 'resources/clusters_pca_single_'+str(SINGLE_CLUSTERS)+'.npy',
                                               'resources/clusters_pca_double_'+str(DOUBLE_CLUSTERS)+'.npy')
    if args.output_type == "single":
        INPUT_CHANNELS = [1, 2]
        OUTPUT_CAHNNELS = [0]
        NUM_CLASSES = SINGLE_CLUSTERS
    elif args.output_type == "double":
        INPUT_CHANNELS = [0]
        OUTPUT_CAHNNELS = [1]
        NUM_CLASSES = DOUBLE_CLUSTERS
    else:
        raise ValueError("Invalid output type")
elif args.algorithm == 'PCAGrid':
    INPUT_SIZE = (227, 227)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.ResizeImage(INPUT_SIZE, preserve_range = True)]
    OUTPUT_FUNCTION = utility.PrepareOutputPCA(OUTPUT_SIZE, 'resources/clusters_pca_grid_single_'+str(SINGLE_CLUSTERS)+'.npy',
                                               'resources/clusters_pca_grid_double_'+str(DOUBLE_CLUSTERS)+'.npy')
    if args.output_type == "single":
        INPUT_CHANNELS = [1, 2]
        OUTPUT_CAHNNELS = [0]
        NUM_CLASSES = SINGLE_CLUSTERS
    elif args.output_type == "double":
        INPUT_CHANNELS = [0]
        OUTPUT_CAHNNELS = [1]
        NUM_CLASSES = DOUBLE_CLUSTERS
    else:
        raise ValueError("Invalid output type")
elif args.algorithm == 'LABKmeans':
    INPUT_SIZE = (227,227)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.ResizeImage(INPUT_SIZE)]
    OUTPUT_FUNCTION = utility.PrepareOutputLAB(OUTPUT_SIZE, 'resources/clusters_lab_single_'+str(SINGLE_CLUSTERS)+'.npy',
                                               'resources/clusters_lab_double_'+str(DOUBLE_CLUSTERS)+'.npy')
    if args.output_type == "single":
        INPUT_CHANNELS = [1, 2]
        OUTPUT_CAHNNELS = [0]
        NUM_CLASSES = SINGLE_CLUSTERS
    elif args.output_type == "double":
        INPUT_CHANNELS = [0]
        OUTPUT_CAHNNELS = [1]
        NUM_CLASSES = DOUBLE_CLUSTERS
    else:
        raise ValueError("Invalid output type")
elif args.algorithm == 'LABGrid':
    INPUT_SIZE = (227,227)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.ResizeImage(INPUT_SIZE)]
    OUTPUT_FUNCTION = utility.PrepareOutputLAB(OUTPUT_SIZE, 'resources/clusters_lab_grid_single_'+str(SINGLE_CLUSTERS)+'.npy',
                                               'resources/clusters_lab_grid_double_'+str(DOUBLE_CLUSTERS)+'.npy')
    if args.output_type == "single":
        INPUT_CHANNELS = [1, 2]
        OUTPUT_CAHNNELS = [0]
        NUM_CLASSES = SINGLE_CLUSTERS
    elif args.output_type == "double":
        INPUT_CHANNELS = [0]
        OUTPUT_CAHNNELS = [1]
        NUM_CLASSES = DOUBLE_CLUSTERS
    else:
        raise ValueError("Invalid output type")
else:
    raise ValueError("Invalid algorithm")

data, labels = utility.read_data(DATASET)

params = {
    'input_size': INPUT_SIZE,
    'output_size': OUTPUT_SIZE,
    'input_channels': INPUT_CHANNELS,
    'output_channels': OUTPUT_CAHNNELS,
    'num_classes': NUM_CLASSES,
    'batch_size': BATCH_SIZE,
    'processing_classes': PROCESSING_CLASSES,
    'output_function': OUTPUT_FUNCTION,
    'shuffle': False
}

generator = DataGenerator(data, **params)

base_model = AlexNet(INPUT_SIZE+(len(INPUT_CHANNELS), ), NUM_CLASSES)

base_model.load_weights(args.weights)

model = Model(inputs = base_model.input, outputs = GlobalAveragePooling2D()(base_model.get_layer('act4').output))

features = model.predict_generator(generator = generator, use_multiprocessing = True, workers = 8, verbose = 1)

with open(args.features, 'wb') as f:
    np.savez(f, features, labels)

