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
SINGLE_CLUSTERS = args.single_clusters
DOUBLE_CLUSTERS = args.double_clusters

with open('resources/bounds.pkl', 'rb') as f:
    bounds = pickle.load(f)

single_points = np.linspace(bounds[0], bounds[1], num = SINGLE_CLUSTERS + 1)
single_clusters = np.zeros((SINGLE_CLUSTERS, 1))

for i in range(single_points.size-1):
    single_clusters[i, :] = 0.5*(single_points[i] + single_points[i+1])

x_points = np.linspace(bounds[2], bounds[3], num = int(np.sqrt(DOUBLE_CLUSTERS)) + 1)
y_points = np.linspace(bounds[4], bounds[5], num = int(np.sqrt(DOUBLE_CLUSTERS)) + 1)

x_clusters = np.zeros((int(np.sqrt(DOUBLE_CLUSTERS)), 1))
y_clusters = np.zeros((int(np.sqrt(DOUBLE_CLUSTERS)), 1))

for i in range(x_points.size-1):
    x_clusters[i, :] = 0.5*(x_points[i] + x_points[i+1])
    y_clusters[i, :] = 0.5*(y_points[i] + y_points[i+1])

X, Y = np.meshgrid(x_clusters, y_clusters)

double_clusters = np.array([X.flatten(), Y.flatten()]).T

print (single_clusters.shape)
print (double_clusters.shape)

with open('resources/clusters_pca_grid_single_'+str(SINGLE_CLUSTERS)+'.npy', 'wb') as f:
    np.savez(f, single_clusters)

with open('resources/clusters_pca_grid_double_'+str(DOUBLE_CLUSTERS)+'.npy', 'wb') as f:
    np.savez(f, double_clusters)
