from sklearn.cluster import MiniBatchKMeans
from skimage.io import imread
from sklearn.externals import joblib
import pickle
import utility
import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split

args = utility.get_parser().parse_args()

DATASET = args.data
NUM_PIXELS = args.num_pixels
SINGLE_CLUSTERS = args.single_clusters
DOUBLE_CLUSTERS = args.double_clusters

image_paths, _ = utility.read_data(DATASET)
image_paths, _ = train_test_split(image_paths, test_size = 1500, random_state = 42)

kmeans_single = MiniBatchKMeans(n_clusters = SINGLE_CLUSTERS)
kmeans_double = MiniBatchKMeans(n_clusters = DOUBLE_CLUSTERS)

pca = joblib.load('resources/pca.pkl')
with open('resources/bounds.pkl', 'rb') as f:
    bounds = pickle.load(f)

batch = np.zeros((NUM_PIXELS, 3))

for i, img in enumerate(image_paths):
    image = imread(img)
    dx, dy, dz = image.shape
    image = np.reshape(image, (dx*dy, dz))
    image = pca.transform(image)
    image = np.reshape(image, (dx, dy, dz))
    image[:, :, 0] = np.clip(image[:, :, 0], bounds[0], bounds[1])
    image[:, :, 1] = np.clip(image[:, :, 1], bounds[2], bounds[3])
    image[:, :, 2] = np.clip(image[:, :, 2], bounds[4], bounds[5])

    dims = image.shape[0], image.shape[1]
    indices = sample_without_replacement(np.prod(dims), NUM_PIXELS)
    indices = np.vstack(np.unravel_index(indices, dims)).T
    for j in range(NUM_PIXELS):
        batch[j, :] = image[indices[j, 0], indices[j, 1], :]

    kmeans_single.partial_fit(np.expand_dims(batch[:, 0], axis = 1))
    kmeans_double.partial_fit(batch[:, 1:3])

with open('resources/clusters_pca_single_'+str(SINGLE_CLUSTERS)+'.npy', 'wb') as f:
    np.savez(f, kmeans_single.cluster_centers_)

with open('resources/clusters_pca_double_'+str(DOUBLE_CLUSTERS)+'.npy', 'wb') as f:
    np.savez(f, kmeans_double.cluster_centers_)
