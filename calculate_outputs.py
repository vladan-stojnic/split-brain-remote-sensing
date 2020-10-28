from sklearn.cluster import MiniBatchKMeans
from skimage.io import imread
from skimage.color import rgb2lab
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

single_data = np.zeros((SINGLE_CLUSTERS, 1))
double_data = np.zeros((DOUBLE_CLUSTERS, 1))

prep = utility.PrepareOutputPCA((12, 12), 'resources/clusters_pca_single_'+str(SINGLE_CLUSTERS)+'.npy', 'resources/clusters_pca_double_'+str(DOUBLE_CLUSTERS)+'.npy')

for i, img in enumerate(image_paths):
    #print(i)
    image = imread(img)    

    img, out = prep(image)

    for j in range(12):
        for k in range(12):
             single_data[out[j, k, 0]] += 1
             double_data[out[j, k, 1]] += 1

print (single_data)
print (double_data)

with open('counts/clusters_pca_single_'+str(SINGLE_CLUSTERS)+'.npy', 'wb') as f:
    np.savez(f, single_data)

with open('counts/clusters_pca_double_'+str(DOUBLE_CLUSTERS)+'.npy', 'wb') as f:
    np.savez(f, double_data)
