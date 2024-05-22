from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.mapping = {}

    def fit(self, X, y=None):
        for col in self.column:
            self.mapping[col] = y.groupby(X[col]).mean()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.column:
            X_copy[col] = X_copy[col].map(self.mapping[col])
        return X_copy