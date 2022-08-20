import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax
from copy import deepcopy

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
                 vectorizer: CountVectorizer,
                 threshold: float = 0.1,
                 class2id: [str] = None):

        self.threshold = threshold
        self.vectorizer = vectorizer
        self.class2id = class2id
        self.reference_classifier = reference_classifier

    def fit(self, all_products: [Product]):
        all_products = deepcopy(all_products)
        # all_products = [Product(**p) for p in all_products] most likely to happen in server code
        refs, prods = preprocess.separate_references(all_products)
        refs.append(Product(product_id="null",
                            name="null",
                            props=[],
                            is_reference=True,
                            reference_id=""))
        for p in prods:
            if p.reference_id is None:
                p.reference_id = "null"

        self.class2id = [r.product_id for r in refs]

        corpus = preprocess.products2corpus(prods)
        x = self.vectorizer.fit_transform(corpus).toarray()

        y_ref = preprocess.build_reference_target(refs, prods)

        self.reference_classifier.fit(x, y_ref)

    def predict(self, products: [Product]):
        products = deepcopy(products)
        assert self.class2id is not None

        corpus = preprocess.products2corpus(products)
        x = self.vectorizer.transform(corpus).toarray()

        # d = self.reference_classifier.decision_function(x)  # for ridge classifier
        # self.reference_classifier.classes_[np.argmax(d, axis=-1)]
        y = self.reference_classifier.predict(x)

        y_ids = [self.class2id[c] for c in y]
        result = [None if predicted_id == "null" else predicted_id for predicted_id in y_ids]

        return result
