from sklearn.base import BaseEstimator,TransformerMixin

class BinaryMapper(BaseEstimator,TransformerMixin):
    def __init__(self,mappings):
        super().__init__()
        self.mappings=mappings

    def fit(self,X,y=None):
        return self
    def transform(self,X):
        copy_X=X.copy(deep=True)
        for key,value in self.mappings.items():
            copy_X[key]=copy_X[key].map(value)
        return copy_X
    def get_feature_names_out(self,input_features=None):
        return input_features