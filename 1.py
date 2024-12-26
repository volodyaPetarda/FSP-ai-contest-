import os
import pickle
from typing import List
import numpy as np

from sklearn.cluster import KMeans

class WheatClassifier:
    def __init__(self):
        self.classifier_ = None
        model_fname = os.path.join(os.path.dirname(__file__), 'wheat_classifier.pkl') #os.path.dirname(__file__)
        if not os.path.isfile(model_fname):
            raise IOError(f'The file "{model_fname}" does not exist!')
        with open(model_fname, mode='rb') as fp:
            self.classifier_ = pickle.load(fp)
        if not isinstance(self.classifier_, KMeans):
            raise IOError(f'The file "{model_fname}" contains a wrong classifier!')

    def predict(self, features: np.array) -> int:
        features = features.reshape(1, -1)
        return self.classifier_.predict(features)[0]

    def predict_batch(self, features_array: np.array) -> np.array:
        return self.classifier_.predict(features_array)