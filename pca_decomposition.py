from sklearn.decomposition import PCA
import numpy as np
import utility
from skimage.io import imread
from sklearn.utils.random import sample_without_replacement
from sklearn.externals import joblib
import pickle
from sklearn.model_selection import train_test_split

args = utility.get_parser().parse_args()

DATASET = args.data
NUM_PIXELS = args.num_pixels

image_paths, _ = utility.read_data(DATASET)
image_paths, _ = train_test_split(image_paths, test_size = 1500, random_state = 42)

X = np.zeros((len(image_paths)*NUM_PIXELS, 3))

for i, img in enumerate(image_paths):
    image = imread(img)
    dims = image.shape[0], image.shape[1]
    indices = sample_without_replacement(np.prod(dims), NUM_PIXELS)
    indices = np.vstack(np.unravel_index(indices, dims)).T
    for j in range(NUM_PIXELS):
        X[i*NUM_PIXELS + j] = image[indices[j, 0], indices[j, 1], :]

pca = PCA(n_components = 3)
output = pca.fit_transform(X)
mins = np.min(output, axis = 0)
maxs = np.max(output, axis = 0)
bounds = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

print(bounds)

joblib.dump(pca, 'resources/pca.pkl')
with open('resources/bounds.pkl', 'wb') as f:
    pickle.dump(bounds, f)
