import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ManualSoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: list):
        self.estimators = estimators

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_")
        probas = []
        for name, estimator in self.estimators:
            try:
                p = estimator.predict_proba(X)
                probas.append(p)
            except Exception as exc:
                from src.utils.logger import logger

                logger.warning("predict_proba failed for %s: %s — skipping.", name, exc)
        if not probas:
            raise RuntimeError("No base model produced valid predict_proba output.")
        return np.mean(np.stack(probas, axis=0), axis=0)

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        indices = np.argmax(avg_proba, axis=1)
        return self.classes_[indices]
