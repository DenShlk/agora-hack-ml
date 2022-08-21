import pickle
from copy import deepcopy

import numpy
import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from typing import Protocol

from .product import Product
from . import preprocess

from .stop_words import STOP_WORDS


class ScikitClassifier(Protocol):
    def fit(self, X, y): ...

    def predict(self, X) -> npt.NDArray: ...

    def score(self, X, y, sample_weight=None) -> float: ...

    def set_params(self, **params): ...


class ProductMatchingModel:
    def __init__(self,
                 reference_classifier: ScikitClassifier = RidgeClassifier(alpha=0.4, normalize=True),
                 unknown_classifier_1: ScikitClassifier = RidgeClassifier(alpha=0.1, normalize=True),
                 unknown_classifier_2: ScikitClassifier = RidgeClassifier(alpha=0.1, normalize=True),
                 vectorizer: CountVectorizer = TfidfVectorizer(stop_words=STOP_WORDS, sublinear_tf=True),
                ):

        self.threshold = 0

        self.unknown_classifier_1 = unknown_classifier_1
        self.unknown_classifier_2 = unknown_classifier_2
        self.vectorizer = vectorizer
        self.reference_classifier = reference_classifier

        self.class2id = None

    def _fit_unknown_classifier(self, predictor: ScikitClassifier, refs_in: [Product], refs_out: [Product],
                                prods: [Product]):
        refs_out_ids = set([ref.reference_id for ref in refs_out])

        prods = deepcopy(prods)
        for p in prods:
            if p.reference_id in refs_out_ids:
                p.reference_id = None

        all_prods = refs_out + refs_in + prods

        corpus = preprocess.products2corpus(all_prods)
        x = self.vectorizer.transform(corpus).toarray()
        y = preprocess.build_unknowns_target(all_prods, refs_out)

        predictor.fit(x, y)

    def set_threshold(self, val):
        self.threshold = val

    def fit(self, all_products: [Product]):
        refs, prods = preprocess.separate_references(all_products)

        self._fit_reference_classifier(refs, all_products)

        refs_half_1, refs_half_2 = train_test_split(refs, test_size=0.5, random_state=42)
        self._fit_unknown_classifier(self.unknown_classifier_1, refs_half_1, refs_half_2, prods)
        self._fit_unknown_classifier(self.unknown_classifier_2, refs_half_2, refs_half_1, prods)

    def _fit_reference_classifier(self, refs, all_products):
        all_products = deepcopy(all_products)

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

        not_in_first_half = self.unknown_classifier_1.predict(x)
        not_in_second_half = self.unknown_classifier_2.predict(x)

        unknowns = np.multiply(not_in_first_half, not_in_second_half)

        known_indices = np.nonzero(unknowns == 0)[0]  # like np.zero
        known_x = x[known_indices]

        result = [None] * len(products)

        if known_x.size:
            y = self.reference_classifier.predict(known_x)
            y_ids = [self.class2id[c] for c in y]
            for idx, predicted_id in zip(known_indices, y_ids):
                result[idx] = predicted_id

        # result = [None] * len(products)

        # if known_x.size:
        #     y = self.reference_classifier.decision_function(known_x)
        #     results = []
        #     for res in y:
        #         res = np.exp(res) / np.sum(np.exp(res))
        #         c = res.argmax()
        #         m = res.mean()
        #         if m == 0:
        #             m += 0.01
        #         print('threshold:', self.threshold)
        #         if res[c] / m < self.threshold:
        #             results.append(None)
        #         else:
        #             results.append(self.class2id[c])
        #     for idx, predicted_id in zip(known_indices, results):
        #         result[idx] = predicted_id

        return result

    def dump(self, path: str):
        with open(path, 'wb') as storage_file:
            pickle.dump(self, storage_file)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as storage_file:
            return pickle.load(storage_file)
