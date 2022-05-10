#%%
from lib2to3.pytree import Base
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import gurobi as gb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import fuzzy_operators as fo
import importlib
importlib.reload(fo)
from sklearn.preprocessing import LabelEncoder


############ SCALERS #############

def check_if_binary(feature):
    eps = 1e-5
    zero_cond = np.logical_and(feature <  eps, feature > -eps)
    one_cond = np.logical_and(feature <  1+eps, feature > 1-eps)
    return np.all(np.logical_or(zero_cond, one_cond))

class DistanceScaler(BaseEstimator, TransformerMixin):

    def __init__(self, nominal_param = 1/np.sqrt(2)):

        self.nominal_param = nominal_param
        self.numerical_scaler = RobustScaler(quantile_range=(1, 99))
        self.nominal_scaler = MinMaxScaler(feature_range=(0, self.nominal_param))
    
    def fit(self, X):
        nominal_features = np.apply_along_axis(check_if_binary, 0, X)
        X_nominal = X[:,nominal_features]
        X_numerical = X[:, np.logical_not(nominal_features)]
        self.nominal_fitted = False
        self.numerical_fitted = False
        if X_numerical.shape[1] > 0:
            self.numerical_scaler.fit(X_numerical)
            self.numerical_fitted = True
        if X_nominal.shape[1] > 0:
            self.nominal_scaler.fit(X_nominal)
            self.nominal_fitted = True

    def transform(self, X):
        nominal_features = np.apply_along_axis(check_if_binary, 0, X)
        X_nominal = X[:,nominal_features]
        X_numerical = X[:, np.logical_not(nominal_features)]

        if self.numerical_fitted:
            X_numerical = self.numerical_scaler.transform(X_numerical)
        if self.nominal_fitted:
            X_nominal = self.nominal_scaler.transform(X_nominal)
        return np.concatenate((X_numerical, X_nominal), axis=1)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

class NumericalScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.numerical_scaler = RobustScaler(quantile_range=(1, 99))
    
    def fit(self, X):
        nominal_features = np.apply_along_axis(check_if_binary, 0, X)
        #X_nominal = X[:,nominal_features]
        X_numerical = X[:, np.logical_not(nominal_features)]
        self.fitted = False
        if X_numerical.shape[1] > 0:
            self.numerical_scaler.fit(X_numerical)
            self.fitted = True

    def transform(self, X):
        nominal_features = np.apply_along_axis(check_if_binary, 0, X)
        X_nominal = X[:,nominal_features]
        X_numerical = X[:, np.logical_not(nominal_features)]
        if self.fitted:
            X_numerical = self.numerical_scaler.transform(X_numerical)
        return np.concatenate((X_numerical, X_nominal), axis=1)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)



############# REGRESSION ######################

#%%


def minus_generator(x, *args):
    if len(args) == 0:
        c=1.
    else:
        c = args[0]
    return c*(1-x)
def log_generator(x, *args):
    
    return -np.log(x)


def get_consistent_relabeling_quantile(rel_matrix_x, decision_label, generator = minus_generator, nn_approx = -1, p=.5):
    model = gb.Model("granular_approximation")
    model.modelSense = gb.GRB.MINIMIZE
    x = []
    y = []
    alphas = []
    n_samples = rel_matrix_x.shape[0]
    
    if nn_approx == -1:
        nn_approx = int(n_samples)
    
    decision_range = np.max(decision_label) - np.min(decision_label)
    for i in range(n_samples):
        x.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, obj=p))
        y.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, obj=1 - p))
        alphas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=-float('inf'), obj= 0))
    
    for i in range(n_samples):
        inds = np.argpartition(-rel_matrix_x[i], kth=nn_approx-1)[:nn_approx]
        for j in inds:
            model.addConstr(alphas[i] - alphas[j] <= generator(rel_matrix_x[i][j], decision_range))
            model.addConstr(alphas[j] - alphas[i] <= generator(rel_matrix_x[i][j], decision_range)) 
    
    for i in range(n_samples):
        model.addConstr(decision_label[i] - alphas[i] == x[i] - y[i])
    

    model.setParam("OutputFlag", 0)
    model.optimize()

    alphas_x = np.zeros(n_samples)
    
    for i in range(n_samples):
        alphas_x[i] = alphas[i].x

    return alphas_x


def get_consistent_relabeling_expectation(rel_matrix_x, decision_label, generator = minus_generator, nn_approx = -1):
    model = gb.Model("granular_approximation")
    model.modelSense = gb.GRB.MINIMIZE
    alphas = []
    n_samples = rel_matrix_x.shape[0]
    if nn_approx == -1:
        nn_approx = int(n_samples)
    #decision_range = np.quantile(decision_label, .99) - np.quantile(decision_label, .01)
    decision_range = np.max(decision_label) - np.min(decision_label)
    for i in range(n_samples):
        alphas.append(model.addVar(vtype=gb.GRB.CONTINUOUS, lb=-float('inf'), obj= 0))
    
    for i in range(n_samples):
        inds = np.argpartition(-rel_matrix_x[i], kth=nn_approx-1)[:nn_approx]
        for j in inds:  
            model.addConstr(alphas[i] - alphas[j] <= generator(rel_matrix_x[i][j], decision_range))
            model.addConstr(alphas[j] - alphas[i] <= generator(rel_matrix_x[i][j], decision_range)) 
    
    model.setObjective(gb.quicksum((alphas[i] - decision_label[i])**2 for i in range(n_samples)))
    

    model.setParam("OutputFlag", 0)
    model.optimize()    

    alphas_x = np.zeros(n_samples)
    
    for i in range(n_samples):
        alphas_x[i] = alphas[i].x

    return alphas_x



#%%

class consistent_regression(BaseEstimator):

    def __init__(self, type = 'quantile', gamma = 1., generator = log_generator, nn_approx=-1, p=.5):
        self.gamma = gamma
        self.type = type
        self.generator = generator
        self.p = p
        self.nn_approx = nn_approx
    

    def fit(self, X, y):
        self.X_train = X
        rel_matrix_x = np.minimum(fo.gaussian_similarity(X, X, arg=self.gamma) + 1e-7, 1)
        self.decision_range = np.max(y) - np.min(y)
        if self.type=='quantile':
            self.relabeling = get_consistent_relabeling_quantile(rel_matrix_x=rel_matrix_x, decision_label=y, 
            generator=self.generator, nn_approx=self.nn_approx, p=self.p)
        if self.type=='expectation':
            self.relabeling = get_consistent_relabeling_expectation(rel_matrix_x=rel_matrix_x, decision_label=y, 
            generator=self.generator, nn_approx=self.nn_approx)

    def predict(self, X):
        rel_matrix = fo.gaussian_similarity(X, self.X_train)
        lower_bound = np.max(self.relabeling - self.generator(rel_matrix, self.decision_range), axis=1)
        upper_bound = np.min(self.relabeling + self.generator(rel_matrix, self.decision_range), axis=1)
        return (lower_bound + upper_bound)/2



############ BINARY CLASSIFICATION ###################

#%%
def triangular_similarity_nominal(x, y, arg):
    nominal_features = np.apply_along_axis(check_if_binary, 0, x)
    X_nominal = x[:,nominal_features]
    X_numerical = x[:, np.logical_not(nominal_features)]
    Y_nominal = y[:,nominal_features]
    Y_numerical = y[:, np.logical_not(nominal_features)]
    rel_matrix_numerical = fo.triangular_similarity(X_numerical, Y_numerical, arg)
    rel_matrix_nominal = fo.triangular_similarity(X_nominal, Y_nominal, 1)
    return np.minimum(rel_matrix_numerical, rel_matrix_nominal)

#%%

#id = lambda x: x

class FuzzyGranularBinaryClassifier(BaseEstimator):

    def __init__(self, loss='quantile', p=.5, arg = 1., nn_approx=-1):
        self.loss = loss
        self.p = p
        self.arg = arg
        self.nn_approx = nn_approx


    def fit(self, X, y):
        self.X_train = X
        rel_matrix = fo.triangular_similarity(self.X_train, self.X_train, arg=self.arg)
        fset = LabelEncoder().fit_transform(y)
        
        n_samples = self.X.shape[0]
        if self.nn_approx > 0 and self.nn_approx < 1:
            self.nn_approx = int(self.nn_approx * n_samples)

        if self.loss == 'quantile':
            self.lambdas = fo.get_granular_approximation_quantile_lukasiewicz(rel_matrix, fset, p=self.p, nn_approx = self.nn_approx)
        elif self.loss == 'mse':
            self.lambdas = fo.get_granular_approximation_expectation_lukasiewicz(rel_matrix, fset, nn_approx = self.nn_approx)
    



    def __get_lambdas_pred(self, X):
        rel_matrix = fo.triangular_similarity(X, self.X_train, arg =self.arg)
        lambdas_left = np.max(fo.lukasiewicz_t_norm (rel_matrix, self.lambdas), axis=1)  
        lambdas_right = np.min(fo.lukasiewicz_implicator (rel_matrix, self.lambdas), axis=1)  
        lambdas_pred = (lambdas_left + lambdas_right)/2
        #scores1 = np.max(fo.lukasiewicz_t_norm (rel_matrix, self.lambdas), axis=1)  
        #scores0 = np.max(fo.lukasiewicz_t_norm (rel_matrix, 1- self.lambdas), axis=1)  
        #prediction = np.argmax(np.stack((scores0,  scores1), axis = 1), axis=1)
        return lambdas_pred
    
    def decision_function(self, X):
        return self.__get_lambdas_pred(X) - 0.5
    
    def predict(self, X):
        return 1*(self.__get_lambdas_pred(X) - 0.5 > 0)
    
    def get_params(self, deep=True):
        return {"loss": self.loss, 
                "p": self.p,
                "arg": self.arg,
                "nn_approx": self.nn_approx}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)     
        return self




class FuzzyGranularBinaryClassifierOWA(BaseEstimator):

    def __init__(self, loss='quantile', p=.5, rel_arg = 1.):
        self.loss = loss
        self.p = p
        self.rel_arg = rel_arg

    def fit(self, X, y):
        self.X_train = X
        rel_matrix = fo.gaussian_similarity(self.X_train, self.X_train, arg=self.rel_arg)
        #fset = LabelEncoder().fit_transform(y)
        fset = y
        if self.loss == 'quantile':
            self.lambdas = fo.get_granular_approximation_quantile_lukasiewicz(rel_matrix, fset, p=self.p)
        elif self.loss == 'mse':
            self.lambdas = fo.get_granular_approximation_expectation_lukasiewicz(rel_matrix, fset)
    

    def predict(self, X):
        rel_matrix = fo.gaussian_similarity(X, self.X_train, arg =self.rel_arg)
        scores1 = fo.owa_invadd_max(fo.lukasiewicz_t_norm (rel_matrix, self.lambdas), axis=1)
        scores0 = fo.owa_invadd_max(fo.lukasiewicz_t_norm (rel_matrix, 1- self.lambdas), axis=1)  
        prediction = np.argmax(np.stack((scores0,  scores1), axis = 1), axis=1)
        return prediction
    
    def get_params(self, deep=True):
        return {"loss": self.loss, 
                "p": self.p,
                "rel_arg": self.rel_arg}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)     
        return self

#%%

###### MUTLI-CLASS CLASSIFICATION ########

class FuzzyGranularMultiClassClassifier(BaseEstimator):

    def __init__(self, loss = 'mse', relation_type = 'triangular', arg = 1., owa_weights = 'strict', nn_approx=-1):
        
        '''
        if relation_type not in ['triangular', 'quadratic']:
            raise ValueError("relation_type parameter can take only values 'triangular' or 'quadratic'")
        else:
        '''
        self.relation_type = relation_type
  

        if loss not in ['mse', 'mae']:
            raise ValueError("loss parameter can take only values 'mae' or 'mse'")
        else:
            self.loss = loss
        self.arg = arg
        self.nn_approx = nn_approx
        self.owa_weights = owa_weights


    
    def fit(self, X, y, sample_weight=None):    
        

        self.X = np.array(X)
        self.y = np.array(y)
        
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        if self.nn_approx > 0 and self.nn_approx < 1:
            self.nn_approx = int(self.nn_approx * self.n_samples)

        self.classes = np.unique(self.y)
        self.num_classes = len(self.classes)
        if self.relation_type == 'triangular':
            rel_matrix_x = fo.triangular_similarity(self.X, self.X, self.arg)
        elif self.relation_type == 'quadratic':
            rel_matrix_x = fo.quadratic_similarity(self.X, self.X, self.arg*np.sqrt(self.n_features))
        elif self.relation_type == 'manhattan':
            rel_matrix_x = fo.manhattan_similarity(self.X, self.X, self.arg*self.n_features)
        
        rel_matrix_y = fo.discernibility_matrix(self.y, self.y)
        if self.loss == 'mse':
            self.lambdas = fo.get_multiclass_granular_approx_mse(rel_matrix_x, rel_matrix_y, sample_weight, nn_approx=self.nn_approx)
        elif self.loss == 'mae':
            self.lambdas = fo.get_multiclass_granular_approx_mae(rel_matrix_x, rel_matrix_y, sample_weight, nn_approx=self.nn_approx)


    def predict(self, X):
        test_size = X.shape[0]
        scores = np.zeros((self.num_classes, test_size))
        for i in range(self.num_classes):
            class_ = self.classes[i]
            positive_inds = np.where(self.y == class_)[0]
            negative_inds = np.where(self.y != class_)[0]   
            X_train_positive = self.X[positive_inds]
            X_train_negative = self.X[negative_inds]
            lambdas_positive = self.lambdas[positive_inds]
            if self.relation_type == 'triangular':
                tmp_rel_matrix = fo.triangular_similarity(X_train_negative, X_train_positive, self.arg)
            elif self.relation_type == 'quadratic':
                tmp_rel_matrix = fo.quadratic_similarity(X_train_negative, X_train_positive, self.arg*np.sqrt(self.n_features))
            elif self.relation_type == 'manhattan':
                tmp_rel_matrix = fo.manhattan_similarity(X_train_negative, X_train_positive, self.arg*self.n_features)
            lambdas_negative = np.max(fo.lukasiewicz_t_norm(tmp_rel_matrix, lambdas_positive), axis=1)
            class_lambdas = np.zeros(self.n_samples)
            class_lambdas[positive_inds] = lambdas_positive
            class_lambdas[negative_inds] = lambdas_negative
            if self.relation_type == 'triangular':
                rel_matrix = fo.triangular_similarity(X, self.X, self.arg)
            elif self.relation_type == 'quadratic':
                rel_matrix = fo.quadratic_similarity(X, self.X, self.arg*np.sqrt(self.n_features))
            elif self.relation_type == 'manhattan':
                rel_matrix = fo.manhattan_similarity(X, self.X, self.arg*self.n_features)
            
            if self.owa_weights == 'strict':
                weights = np.concatenate((np.ones(1), np.zeros(self.n_samples-1)))    
            elif self.owa_weights == 'add':
                weights = fo.get_add_weights(self.n_samples)
            elif self.owa_weights == 'exp':
                weights = fo.get_exp_weights(self.n_samples)
            elif self.owa_weights == 'invadd':
                weights = fo.get_invadd_weights(self.n_samples)
            
            lambdas_left = fo.owa_max(fo.lukasiewicz_t_norm (rel_matrix, class_lambdas), weighs=weights, axis=1)  
            lambdas_right = fo.owa_min(fo.lukasiewicz_implicator(rel_matrix, class_lambdas), weighs=weights, axis=1)
            
            scores[i] = (lambdas_left + lambdas_right)/2
            inds = np.argmax(scores, axis=0)
        return self.classes[inds]
    
    def get_params(self, deep=True):
        return {"loss": self.loss, 
                "relation_type": self.relation_type,
                "arg": self.arg,
                "nn_approx" : self.nn_approx,
                "owa_weights" : self.owa_weights,
                }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)     
        return self

#%%



class kFROWANNClassifier(BaseEstimator):
    def __init__(self, n_neighbors=-1, owa_type='add', relation_type = 'triangular'):
        self.n_neighbors = n_neighbors
        self.owa_type = owa_type
        self.relation_type = relation_type


    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

        if self.n_neighbors == -1:
            self.n_neighbors = n_samples
        
        if self.owa_type == 'add':
            if n_samples > self.n_neighbors:
                first_part = fo.get_add_weights(self.n_neighbors)
                second_part = np.zeros(n_samples - self.n_neighbors)
                self.weights = np.concatenate((first_part, second_part))
            else:
                self.weights = fo.get_add_weights(n_samples)
        elif self.owa_type == 'exp':
            if n_samples > self.n_neighbors:
                first_part = fo.get_exp_weights(self.n_neighbors)
                second_part = np.zeros(n_samples - self.n_neighbors)
                self.weights = np.concatenate((first_part, second_part))
            else:
                self.weights = fo.get_exp_weights(n_samples)
        elif self.owa_type == 'invadd':
            if n_samples > self.n_neighbors:
                first_part = fo.get_invadd_weights(self.n_neighbors)
                second_part = np.zeros(n_samples - self.n_neighbors)
                self.weights = np.concatenate((first_part, second_part))
            else:
                self.weights = fo.get_invadd_weights(n_samples)
        elif self.owa_type == 'strict':
            self.weights = np.concatenate((np.ones(1), np.zeros(n_samples - 1)))
        

    def predict(self, X):
        classes = list(set(self.y))
        approx_values = []
        for class_ in classes:
            tmp_lambdas = 1*(self.y == class_)
            if self.relation_type == 'triangular':
                rel_matrix = fo.triangular_similarity(X, self.X, arg=1)
            elif self.relation_type == 'quadratic':
                rel_matrix= fo.quadratic_similarity(X, self.X, arg=np.sqrt(self.n_features))
            low_approx = fo.owa_min(fo.lukasiewicz_implicator(rel_matrix, tmp_lambdas), weighs=self.weights, axis=1)
            upp_approx = fo.owa_max(fo.lukasiewicz_t_norm(rel_matrix, tmp_lambdas), weighs=self.weights, axis=1)
            approx_values.append((low_approx + upp_approx)/2)
        
        return np.argmax(approx_values, axis = 0)
    
    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, 
                "owa_type": self.owa_type,
                "relation_type": self.relation_type,
                }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)     
        return self
        
# %%
import scipy as sp
class CustomKNeighborsClassifier(BaseEstimator):
    
    def __init__ (self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    
    def predict(self, X):
        similarity_mat = fo.quadratic_similarity(X, self.X)
        closest_inds = np.argsort(-similarity_mat, axis=1)[:,:self.n_neighbors]
        categ = self.y[closest_inds.reshape(1,-1)[0]].reshape(-1, self.n_neighbors)
        pred = sp.stats.mode(categ, axis = 1).mode.T[0]
        return pred

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, 
                }
    
    def kneighbors(self, X, n_neighbors):
        similarity_mat = fo.quadratic_similarity(X, self.X)
        closest_inds = np.argsort(-similarity_mat, axis=1)[:,:n_neighbors]
        return closest_inds

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)     
        return self





        



