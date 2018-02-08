from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


class ModelWrapper:
    """
    Wrapper to simplify the process of
    loading data, parameter tuning, etc
    """
    def __init__(self, csv_name, *args, debug=False, **kwargs):
        df = pd.read_csv(csv_name)
        self.X = self.preprocess_X(df.iloc[:, :-1])
        
        # when debugging only use a tiny subset of the data
        if debug:
            self.X = self.X[:10]

        self.Y = df.iloc[:,-1]
        
    def preprocess_X(self, X):
        return X

    def predict(self, X):
        """
        Return a list of predictions
        and error associated with the model
        """
        raise NotImplemented
    
    def train(self, X, Y):
        """
        Train on a subset of the data, returning
        a model
        """
        raise NotImplemented

    def build_model(self):
        """
        Builds a model
        """
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y)
        print ("ASDASD")
        self.train(X_train, Y_train)
        Y_pred = self.predict(X_test)
        print (mean_squared_error(Y_pred, Y_test), "ERROR")


class SKLearnModelWrapper(ModelWrapper):
    """
    Specifically built for SKLearn models
    """
    def get_model(self):
        """
        Return the SKLearn model class
        """
        raise NotImplemented

    def train(self, X, Y):
        self.model = self.get_model()
        assert (self.model is not None)
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)


class CombinedModelWrapper(ModelWrapper):
    """
    Combines the predictions for several models
    MAKE SURE THAT THE DATAFRAMES ARE ORDERED
    THE SAME WAY!!
    """
    def __init__(self, models, *args, **kwargs):
        self.models = models
        self.X = list(zip(model.X for model in models))
        self.Y = models[0].Y[:len(self.X)]
        
    def apply_models(self, X):
        return [
            [
                model.predict(x)
                for model, x in zip(self.models, dataset)
            ]
            for dataset in X
        ]

    def train(self, X, Y):
        X_transformed = self.apply_models(X)
        return super(CombinedModelWrapper, self).train(X_transformed, Y)

    def predict(self, X):
        X_transformed = self.apply_models(X)
        return super(CombinedModelWrapper, self).predict(X_transformed)

    def build_model(self):
        for model in self.models:
            model.build_model()

        return super(CombinedModelWrapper, self).build_model()