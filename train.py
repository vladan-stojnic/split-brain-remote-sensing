import utility
from sklearn.model_selection import train_test_split
from ast import literal_eval
from generator import DataGenerator
from networks import AlexNet
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = utility.get_parser().parse_args()

DATASET = args.data
TRAINSET_SIZE = args.dataset_size
#INPUT_SIZE = literal_eval(args.input_size)
#OUTPUT_SIZE = literal_eval(args.output_size)
#INPUT_CHANNELS = literal_eval(args.input_channels)
#OUTPUT_CHANNELS = literal_eval(args.output_channels)
#NUM_CLASSES = args.num_classes
BATCH_SIZE = args.batch_size
INITIALIZER = args.initializer

if args.algorithm == "Zhang":
    INPUT_SIZE = (180, 180)
    OUTPUT_SIZE = (12, 12)
    PROCESSING_CLASSES = [utility.RandomCrop(INPUT_SIZE)]
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
    INPUT_SIZE = (180, 180)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.RandomCrop(INPUT_SIZE)]
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
    INPUT_SIZE = (180, 180)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.RandomCrop(INPUT_SIZE)]
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
    INPUT_SIZE = (180, 180)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.RandomCrop(INPUT_SIZE)]
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
    INPUT_SIZE = (180, 180)
    OUTPUT_SIZE = (12, 12)
    SINGLE_CLUSTERS = args.single_clusters
    DOUBLE_CLUSTERS = args.double_clusters
    PROCESSING_CLASSES = [utility.RandomCrop(INPUT_SIZE)]
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

data, _ = utility.read_data(DATASET)
train_data, val_data = train_test_split(data, test_size = 1500, random_state = 42)
if TRAINSET_SIZE != 30000:
    _, train_data = train_test_split(train_data, test_size = TRAINSET_SIZE, random_state = 42)

params = {
    'input_size': INPUT_SIZE,
    'output_size': OUTPUT_SIZE,
    'input_channels': INPUT_CHANNELS,
    'output_channels': OUTPUT_CAHNNELS,
    'num_classes': NUM_CLASSES,
    'batch_size': BATCH_SIZE,
    'processing_classes': PROCESSING_CLASSES,
    'output_function': OUTPUT_FUNCTION
}

train_generator = DataGenerator(train_data, **params)
validation_generator = DataGenerator(val_data, **params)

model = AlexNet(INPUT_SIZE+(len(INPUT_CHANNELS), ), NUM_CLASSES, INITIALIZER)

model.summary()

optimizer = Adam(lr = 1e-4)
tensorboard = TensorBoard(log_dir = './Graph', histogram_freq = 0, write_grads = False, batch_size = BATCH_SIZE, write_images = False)
reduce_on_plateau = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 5, mode = 'min', min_delta = 1e-4)
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 15, mode = 'min')
model_checkpoint = ModelCheckpoint('models/'+args.algorithm+'_'+args.output_type+'_'+str(NUM_CLASSES)+'_'+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_weights_only=True, mode='min', period=15)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = [utility.pixelwise_accuracy])

model.fit_generator(generator = train_generator, epochs = 200, callbacks = [tensorboard, reduce_on_plateau, early_stopping, model_checkpoint],
                    validation_data = validation_generator, use_multiprocessing = True, workers = 8)

model.save_weights('models/'+args.algorithm+'_'+args.output_type+'_'+str(NUM_CLASSES)+'_'+'final.hdf5')
