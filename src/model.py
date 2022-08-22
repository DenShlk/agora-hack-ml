import pickle
from copy import deepcopy

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier

from . import preprocess
from .product import Product
from .stop_words import STOP_WORDS


class ProductMatchingModel:
    def __init__(self,
                 reference_classifier: RidgeClassifier = RidgeClassifier(alpha=0.4),
                 vectorizer: CountVectorizer = TfidfVectorizer(stop_words=STOP_WORDS),
                 class2id: [str] = None):

        self.threshold = 0
        self.vectorizer = vectorizer
        self.reference_classifier = reference_classifier
        self.class2id = class2id

    def fit(self, all_products: [Product]):
        all_products = deepcopy(all_products)
        refs, prods = preprocess.separate_references(all_products)

        self.class2id = [r.product_id for r in refs]

        # remove products without reference
        all_products = list(filter(lambda p: p.reference_id is not None or p.is_reference, all_products))

        corpus = preprocess.products2corpus(all_products)
        x = self.vectorizer.fit_transform(corpus).toarray()
        y_ref = preprocess.build_reference_target(refs, all_products)

        self.reference_classifier.fit(x, y_ref)

    def predict(self, products: [Product]):
        products = deepcopy(products)
        assert self.class2id is not None

        corpus = preprocess.products2corpus(products)
        x = self.vectorizer.transform(corpus).toarray()
        y = self.reference_classifier.decision_function(x)

        # TODO: describe what's up
        results = []
        for res in y:
            res = np.exp(res) / np.sum(np.exp(res))
            c = res.argmax()
            m = res.mean()
            if m == 0:
                m += 0.01

            if res[c] / m < self.threshold:
                results.append(None)
            else:
                results.append(self.class2id[c])

        return results

    def dump(self, path: str):
        with open(path, 'wb') as storage_file:
            pickle.dump(self, storage_file)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as storage_file:
            return pickle.load(storage_file)
