import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

from product import Product


def separate_references(products: List[Product]) -> Tuple[List[Product], List[Product]]:
    references = [p for p in products if p.is_reference]
    references_id_set = set([ref.product_id for ref in references])

    referrers = [p for p in products if p.product_id not in references_id_set]
    return references, referrers


def products2corpus(products: List[Product]) -> List[str]:
    return [p.name + ' ' + ' '.join(p.props) for p in products]


def build_reference_target(references: List[Product], products: List[Product]) -> npt.NDArray[int]:
    # Построим таргет сет, номером класса будет индекс эталона в изначальном сете
    ref_id2ind = {
        ref.product_id: i for i, ref in enumerate(references)
    }

    return np.array([
        ref_id2ind[p.product_id if p.is_reference else p.reference_id] for p in products
    ])


def build_unknowns_target(products: List[Product], outlier_refs: List[Product]) -> npt.NDArray[int]:
    outlier_ref_ids = set(map(lambda p: p.product_id, outlier_refs))
    result = [0] * len(products)

    for i, p in enumerate(products):
        if p.is_reference:
            if p.product_id in outlier_ref_ids:
                result[i] = 1
        elif p.reference_id is None:
            result[i] = 1
    return np.array(list(result))
