import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import CountVectorizer

from product import Product
import preprocess
from typing import Protocol


class ScikitClassifier(Protocol):
    def fit(self, X, y): ...

    def predict(self, X) -> npt.NDArray: ...

    def score(self, X, y, sample_weight=None) -> float: ...

    def set_params(self, **params): ...


class ProductMatchingModel:
    def __init__(self,
                 reference_classifier: ScikitClassifier,
                 unknown_classifier: ScikitClassifier,
                 vectorizer: CountVectorizer,
                 class2id: [str] = None):
        self.vectorizer = vectorizer
        self.class2id = class2id
        self.unknown_classifier = unknown_classifier
        self.reference_classifier = reference_classifier

    def fit(self, all_products: [Product]):
        # all_products = [Product(**p) for p in all_products] most likely to happen in server code
        refs, prods = preprocess.separate_references(all_products)
        self.class2id = [r.product_id for r in refs]

        corpus = preprocess.products2corpus(prods)
        x = self.vectorizer.fit_transform(corpus).toarray()

        y_ref = preprocess.build_reference_target(refs, prods)
        y_unk = preprocess.build_unknowns_target(prods)

        self.reference_classifier.fit(x, y_ref)
        self.unknown_classifier.fit(x, y_unk)

    def predict(self, products: [Product]):
        assert self.class2id is not None

        corpus = preprocess.products2corpus(products)
        x = self.vectorizer.transform(corpus).toarray()

        unknowns = self.unknown_classifier.predict(x)

        known_indices = np.nonzero(unknowns)
        known_x = x[known_indices]

        result = [None] * len(products)

        y = self.reference_classifier.predict(known_x)
        y_ids = [self.class2id[c] for c in y]

        for idx, predicted_id in zip(known_indices, y_ids):
            result[idx] = predicted_id

        return result
