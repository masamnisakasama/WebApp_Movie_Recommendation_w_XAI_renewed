# lime_utils.py
from lime.lime_text import LimeTextExplainer
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# SentenceTransformer は重いので1回だけロードして高速化されないかね
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL

# 日本語の文/句の分割パターン
_SENT_SPLIT   = r'(?<=[。．\.！!？\?…])\s*'          # 文末で区切ることでsentenceに分割
_CLAUSE_SPLIT = r'(?<=[。．\.！!？\?…、，,；;])\s*'    # 句読点も区切ることでclauseに分割

def _split_expr_from_unit(unit: str) -> str:
    return _CLAUSE_SPLIT if unit == "clause" else _SENT_SPLIT

class BERTSimilarityExplainer:
    def __init__(self, base_description, token_unit="sentence", model=None, **kwargs):
        self.base_description = base_description
        self.model = model if model is not None else _get_model()

        # ベース埋め込み
        self.base_embedding = self.model.encode([base_description], convert_to_numpy=True)[0]

        def similarity_func(texts):
            emb = self.model.encode(texts, convert_to_numpy=True)
            sim = np.dot(emb, self.base_embedding) / (
                np.linalg.norm(emb, axis=1) * np.linalg.norm(self.base_embedding) + 1e-10
            )
            # LIME用2クラス確率 [Not Similar, Similar]
            return np.vstack([1 - sim, sim]).T

        self.similarity_score = similarity_func

        # 文/句単位の分割　
        self.explainer = LimeTextExplainer(
            class_names=["Not Similar", "Similar"],
            bow=True,
            verbose=False,
            random_state=1,
            feature_selection='none',
            kernel_width=25,
            split_expression=_split_expr_from_unit(token_unit),
        )

    def explain(self, target_description, num_features=10, num_samples=100):
        return self.explainer.explain_instance(
            target_description,
            self.similarity_score,
            labels=[1],
            num_features=num_features,
            num_samples=num_samples,   
        )
