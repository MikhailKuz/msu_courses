import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from timeit import default_timer as timer


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                **trees_parameters): 
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features
            
        y_val : numpy ndarray
            Array of size n_val_objects           
        """
        add_data = (not X_val is None) and (not y_val is None)
        n_est = self.n_estimators
        if add_data:
            self.n_estimators = 0
            start = timer()
            scores = []
            times = []
        for _ in range(n_est):
            dtr = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                        **self.trees_parameters)
            idx = np.random.choice(list(range(X.shape[0])), X.shape[0], replace=True)
            dtr.fit(X[idx], y[idx])
            self.trees.append(dtr)
            if add_data:
                self.n_estimators += 1
                end = timer()
                delta = end - start
                times.append(delta)
                scores.append(rmse(y_val, self.predict(X_val)))
        if add_data:
            return dict([('scores', scores), ('time', times)])

        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects,    n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            ans += self.trees[i].predict(X)

        return ans / self.n_estimators


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        self.gamma = []
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        add_data = (not X_val is None) and (not y_val is None)
        n_est = self.n_estimators
        if add_data:
            self.n_estimators = 0
            start = timer()
            scores = []
            times = []
        ans = np.zeros(X.shape[0])
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 'sqrt'
        
        for _ in range(n_est):
            dtr = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                        **self.trees_parameters)
            dtr.fit(X, 2 * (ans - y) / X.shape[0])
            preds = dtr.predict(X)
            self.gamma.append(minimize_scalar(lambda x: mse(y, ans + x * preds)).x)
            self.trees.append(dtr)
            ans += self.lr * self.gamma[-1] * preds
            if add_data:
                self.n_estimators += 1
                end = timer()
                delta = end - start
                times.append(delta)
                scores.append(rmse(y_val, self.predict(X_val)))
        if add_data:
            return dict([('scores', scores), ('time', times)])

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            ans += self.gamma[i] * self.lr * self.trees[i].predict(X)
        return ans
    
def mse(y, ans):
    return ((y - ans) ** 2).mean()

def rmse(y, ans):
    return np.sqrt(mse(y, ans))
