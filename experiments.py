#%%

import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
import importlib
import scipy as sp
import fuzzy_operators as fo
import fuzzy_consistent_learning as fcl
importlib.reload(fo)
importlib.reload(fcl)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
import copy
import sklvq
import pandas as pd
import os
import warnings
import time
warnings.filterwarnings('ignore')


#%%
rand_state = 10
custom_scaler = fcl.DistanceScaler()
custom_scaler2 = fcl.NumericalScaler()
aux_model = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=rand_state)
#feature_selection = SelectFromModel(aux_model, threshold="0.1*mean")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand_state)
ros = RandomOverSampler(random_state=rand_state)
n_jobs=1
decimals = 3

# %%

########## OUR GRANULAR CLASSIFIERS ##############

granular_model_mae_triangular_nnall = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=-1, relation_type='triangular', loss='mae')
granular_model_mae_triangular_nn20 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.2, relation_type='triangular', loss='mae')
granular_model_mae_triangular_nn2 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular', loss='mae')
granular_model_mae_quadratic_nnall = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=-1, relation_type='quadratic', loss='mae')
granular_model_mae_quadratic_nn20 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.2, relation_type='quadratic', loss='mae')
granular_model_mae_quadratic_nn2 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic', loss='mae')
granular_model_mse_triangular_nnall = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=-1, relation_type='triangular')
granular_model_mse_triangular_nn20 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.2, relation_type='triangular')
granular_model_mse_triangular_nn2 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular')
granular_model_mse_quadratic_nnall = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=-1, relation_type='quadratic')
granular_model_mse_quadratic_nn20 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.2, relation_type='quadratic')
granular_model_mse_quadratic_nn2 = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic')


granular_models = ['granular_model_mae_triangular_nnall', 'granular_model_mae_triangular_nn20', 'granular_model_mae_triangular_nn2', 'granular_model_mae_quadratic_nnall',
                    'granular_model_mae_quadratic_nn20', 'granular_model_mae_quadratic_nn2', 'granular_model_mse_triangular_nnall', 'granular_model_mse_triangular_nn20',
                    'granular_model_mse_triangular_nn2', 'granular_model_mse_quadratic_nnall', 'granular_model_mse_quadratic_nn20', 'granular_model_mse_quadratic_nn2']

granular_pipeline_mae_triangular_nnall = make_pipeline(ros, custom_scaler2,   granular_model_mae_triangular_nnall)
granular_pipeline_mae_triangular_nn20 = make_pipeline(ros, custom_scaler2, granular_model_mae_triangular_nn20)
granular_pipeline_mae_triangular_nn2 = make_pipeline(ros, custom_scaler2, granular_model_mae_triangular_nn2)
granular_pipeline_mae_quadratic_nnall = make_pipeline(ros, custom_scaler, granular_model_mae_quadratic_nnall)
granular_pipeline_mae_quadratic_nn20 = make_pipeline(ros, custom_scaler, granular_model_mae_quadratic_nn20)
granular_pipeline_mae_quadratic_nn2 = make_pipeline(ros, custom_scaler, granular_model_mae_quadratic_nn2)
granular_pipeline_mse_triangular_nnall = make_pipeline(ros, custom_scaler2, granular_model_mse_triangular_nnall)
granular_pipeline_mse_triangular_nn20 = make_pipeline(ros, custom_scaler2, granular_model_mse_triangular_nn20)
granular_pipeline_mse_triangular_nn2 = make_pipeline(ros, custom_scaler2, granular_model_mse_triangular_nn2)
granular_pipeline_mse_quadratic_nnall = make_pipeline(ros, custom_scaler, granular_model_mse_quadratic_nnall)
granular_pipeline_mse_quadratic_nn20 = make_pipeline(ros, custom_scaler, granular_model_mse_quadratic_nn20)
granular_pipeline_mse_quadratic_nn2 = make_pipeline(ros, custom_scaler, granular_model_mse_quadratic_nn2)

granular_params = {'fuzzygranularmulticlassclassifier__arg': [1/10, 1/7, 1/5, 1/4, 1/3, 1/2, 1/1.5,  1., 1/.8, 1/.7,1/.5]}

granular_search_mae_triangular_nnall = GridSearchCV(granular_pipeline_mae_triangular_nnall, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_triangular_nn20 = GridSearchCV(granular_pipeline_mae_triangular_nn20, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_triangular_nn2 = GridSearchCV(granular_pipeline_mae_triangular_nn2, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_quadratic_nnall = GridSearchCV(granular_pipeline_mae_quadratic_nnall, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_quadratic_nn20 = GridSearchCV(granular_pipeline_mae_quadratic_nn20, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_quadratic_nn2 = GridSearchCV(granular_pipeline_mae_quadratic_nn2, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_triangular_nnall = GridSearchCV(granular_pipeline_mse_triangular_nnall, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_triangular_nn20 = GridSearchCV(granular_pipeline_mse_triangular_nn20, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_triangular_nn2 = GridSearchCV(granular_pipeline_mse_triangular_nn2, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_quadratic_nnall = GridSearchCV(granular_pipeline_mse_quadratic_nnall, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_quadratic_nn20 = GridSearchCV(granular_pipeline_mse_quadratic_nn20, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_quadratic_nn2 = GridSearchCV(granular_pipeline_mse_quadratic_nn2, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)

#%%

######## OWA-BASED GRANULAR CLASSIFIER ################


granular_model_mae_triangular_add = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular', loss='mae', owa_weights='add')
granular_model_mae_quadratic_add = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic', loss='mae', owa_weights='add')
granular_model_mse_triangular_add = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular', owa_weights='add')
granular_model_mse_quadratic_add = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic', owa_weights='add')
granular_model_mae_triangular_exp = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular', loss='mae', owa_weights='exp')
granular_model_mae_quadratic_exp = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic', loss='mae', owa_weights='exp')
granular_model_mse_triangular_exp = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular', owa_weights='exp')
granular_model_mse_quadratic_exp = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic', owa_weights='exp')
granular_model_mae_triangular_invadd = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular', loss='mae', owa_weights='invadd')
granular_model_mae_quadratic_invadd = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic', loss='mae', owa_weights='invadd')
granular_model_mse_triangular_invadd = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='triangular', owa_weights='invadd')
granular_model_mse_quadratic_invadd = fcl.FuzzyGranularMultiClassClassifier(arg=1., nn_approx=.02, relation_type='quadratic', owa_weights='invadd')

granular_owa_models = ['granular_model_mae_triangular_add', 'granular_model_mae_quadratic_add', 'granular_model_mse_triangular_add', 'granular_model_mse_quadratic_add',
            'granular_model_mae_triangular_exp', 'granular_model_mae_quadratic_exp', 'granular_model_mse_triangular_exp', 'granular_model_mse_quadratic_exp', 
            'granular_model_mae_triangular_invadd', 'granular_model_mae_quadratic_invadd', 'granular_model_mse_triangular_invadd', 'granular_model_mse_quadratic_invadd']

granular_pipeline_mae_triangular_add = make_pipeline(ros, custom_scaler2, granular_model_mae_triangular_add)
granular_pipeline_mae_quadratic_add = make_pipeline(ros, custom_scaler, granular_model_mae_quadratic_add)
granular_pipeline_mse_triangular_add = make_pipeline(ros, custom_scaler2, granular_model_mse_triangular_add)
granular_pipeline_mse_quadratic_add = make_pipeline(ros, custom_scaler, granular_model_mse_quadratic_add)
granular_pipeline_mae_triangular_exp = make_pipeline(ros, custom_scaler2, granular_model_mae_triangular_exp)
granular_pipeline_mae_quadratic_exp = make_pipeline(ros, custom_scaler, granular_model_mae_quadratic_exp)
granular_pipeline_mse_triangular_exp = make_pipeline(ros, custom_scaler2, granular_model_mse_triangular_exp)
granular_pipeline_mse_quadratic_exp = make_pipeline(ros, custom_scaler, granular_model_mse_quadratic_exp)
granular_pipeline_mae_triangular_invadd = make_pipeline(ros, custom_scaler2, granular_model_mae_triangular_invadd)
granular_pipeline_mae_quadratic_invadd = make_pipeline(ros, custom_scaler, granular_model_mae_quadratic_invadd)
granular_pipeline_mse_triangular_invadd = make_pipeline(ros, custom_scaler2, granular_model_mse_triangular_invadd)
granular_pipeline_mse_quadratic_invadd = make_pipeline(ros, custom_scaler, granular_model_mse_quadratic_invadd)

granular_search_mae_triangular_add = GridSearchCV(granular_pipeline_mae_triangular_add, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_quadratic_add = GridSearchCV(granular_pipeline_mae_quadratic_add, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_triangular_add = GridSearchCV(granular_pipeline_mse_triangular_add, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_quadratic_add = GridSearchCV(granular_pipeline_mse_quadratic_add, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_triangular_exp = GridSearchCV(granular_pipeline_mae_triangular_exp, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_quadratic_exp = GridSearchCV(granular_pipeline_mae_quadratic_exp, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_triangular_exp = GridSearchCV(granular_pipeline_mse_triangular_exp, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_quadratic_exp = GridSearchCV(granular_pipeline_mse_quadratic_exp, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_triangular_invadd = GridSearchCV(granular_pipeline_mae_triangular_invadd, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mae_quadratic_invadd = GridSearchCV(granular_pipeline_mae_quadratic_invadd, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_triangular_invadd = GridSearchCV(granular_pipeline_mse_triangular_invadd, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
granular_search_mse_quadratic_invadd = GridSearchCV(granular_pipeline_mse_quadratic_invadd, param_grid=granular_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)


light_granular_models = ['granular_model_mse_quadratic_nn2', 'granular_model_mse_quadratic_add', 'granular_model_mse_quadratic_exp', 'granular_model_mse_quadratic_invadd',
                        'granular_model_mae_quadratic_nn2', 'granular_model_mae_quadratic_add', 'granular_model_mae_quadratic_exp', 'granular_model_mae_quadratic_invadd',
                        'granular_model_mse_triangular_nn2', 'granular_model_mse_triangular_add', 'granular_model_mse_triangular_exp', 'granular_model_mse_triangular_invadd',
                        'granular_model_mae_triangular_nn2', 'granular_model_mae_triangular_add', 'granular_model_mae_triangular_exp', 'granular_model_mae_triangular_invadd' ]
#%%%

########### KNOWN ML METHODS ############

lr_model = LogisticRegression(C=1., penalty='l2', solver='liblinear', verbose=0)
knn_model = KNeighborsClassifier(n_neighbors=5)
lvq_model = sklvq.GLVQ(prototype_n_per_class=1.)
dt_model = DecisionTreeClassifier(max_depth=3)

lr_pipeline = make_pipeline(ros, custom_scaler, lr_model)
knn_pipeline = make_pipeline(ros, custom_scaler, knn_model)
lvq_pipeline = make_pipeline(ros, custom_scaler, lvq_model)   
dt_pipeline = make_pipeline(ros, custom_scaler, dt_model)

knn_params = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]}
lr_params = {'logisticregression__C': [ 100, 50, 30, 20, 10., 5., 1., .5, .1, .05, .01]}
lvq_params = {'glvq__prototype_n_per_class': [1,2,3,4,5,6,7,8, 9,10, 11]}
dt_params = {'decisiontreeclassifier__max_depth': [2,3,4,5,6,7,8,9, 10, 11, 12]}

lr_search = GridSearchCV(lr_pipeline, param_grid=lr_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
knn_search =  GridSearchCV(knn_pipeline, param_grid=knn_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
lvq_search = GridSearchCV(lvq_pipeline, param_grid=lvq_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
dt_search = GridSearchCV(dt_pipeline, param_grid=dt_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)

known_ml_models = ['lr_model', 'knn_model', 'lvq_model', 'dt_model']

#%%

######## FRNN METHODS ########

kFRNN_model_add_triangular = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='add')
kFRNN_model_exp_triangular = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='exp')
kFRNN_model_invadd_triangular = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='invadd')
kFRNN_model_strict_triangular = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='strict')
kFRNN_model_add_quadratic = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='add', relation_type='quadratic')
kFRNN_model_exp_quadratic = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='exp', relation_type='quadratic')
kFRNN_model_invadd_quadratic = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='invadd', relation_type='quadratic')
kFRNN_model_strict_quadratic = fcl.kFROWANNClassifier(n_neighbors=5, owa_type='strict', relation_type='quadratic')

kFRNN_pipeline_add_triangular = make_pipeline(ros, custom_scaler2, kFRNN_model_add_triangular)
kFRNN_pipeline_exp_triangular = make_pipeline(ros,custom_scaler2, kFRNN_model_exp_triangular)
kFRNN_pipeline_invadd_triangular = make_pipeline(ros, custom_scaler2, kFRNN_model_invadd_triangular)
kFRNN_pipeline_strict_triangular = make_pipeline(ros, custom_scaler2, kFRNN_model_strict_triangular)
kFRNN_pipeline_add_quadratic = make_pipeline(ros, custom_scaler, kFRNN_model_add_quadratic)
kFRNN_pipeline_exp_quadratic = make_pipeline(ros,custom_scaler, kFRNN_model_exp_quadratic)
kFRNN_pipeline_invadd_quadratic = make_pipeline(ros, custom_scaler, kFRNN_model_invadd_quadratic)
kFRNN_pipeline_strict_quadratic = make_pipeline(ros, custom_scaler, kFRNN_model_strict_quadratic)

kFRNN_params = {'kfrowannclassifier__n_neighbors': [-1, 1, 3, 5, 10, 15, 20, 25, 30, 40, 50]}

kFRNN_search_add_triangular = GridSearchCV(kFRNN_pipeline_add_triangular, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
kFRNN_search_exp_triangular = GridSearchCV(kFRNN_pipeline_exp_triangular, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
kFRNN_search_invadd_triangular = GridSearchCV(kFRNN_pipeline_invadd_triangular, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
kFRNN_search_strict_triangular = GridSearchCV(kFRNN_pipeline_strict_triangular, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
kFRNN_search_add_quadratic = GridSearchCV(kFRNN_pipeline_add_quadratic, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
kFRNN_search_exp_quadratic = GridSearchCV(kFRNN_pipeline_exp_quadratic, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
kFRNN_search_invadd_quadratic = GridSearchCV(kFRNN_pipeline_invadd_quadratic, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)
kFRNN_search_strict_quadratic = GridSearchCV(kFRNN_pipeline_strict_quadratic, param_grid=kFRNN_params, scoring='balanced_accuracy', cv=cv, n_jobs=n_jobs)

frnn_models = ['kFRNN_model_add_triangular', 'kFRNN_model_exp_triangular', 'kFRNN_model_invadd_triangular', 'kFRNN_model_strict_triangular',
                'kFRNN_model_add_quadratic', 'kFRNN_model_exp_quadratic', 'kFRNN_model_invadd_quadratic', 'kFRNN_model_strict_quadratic',]




#%%
def get_weights(y):
    res = np.array(np.unique(y, return_counts=True))
    map = pd.Series(data=res[1], index=res[0])
    return y.size/map[y].values

def get_dataset_name(ss):
    ss = ss[:-4]
    result = ''.join(c for c in ss if c.isalpha())
    return result

# %%
datasets = ["australian690(3_5_6)2.csv", "breast277(0_0_9)2.csv", "crx653(3_3_9)2.csv",
            "german1000(0_7_13)2.csv", "saheart462(5_3_1)2.csv", "ionosphere351(32_1_0)2.csv", 
            "mammographic830(0_5_0)2.csv", "pima768(8_0_0)2.csv", "wisconsin683(0_9_0)2.csv",
            "vowel990(10_3_0)11.csv", "wdbc569(30_0_0)2.csv", "balance625(4_0_0)3.csv", 
            "glass214(9_0_0)6.csv", "cleveland297(13_0_0)5.csv", "bupa345(1_5_0)2.csv", 
            "haberman306(0_3_0)2.csv", "heart270(1_12_0)2.csv", "spectfheart267(0_44_0)2.csv"]

datasets2 = ["german1000(0_7_13)2.csv"]


#datasets = ["australian690(3_5_6)2.csv", "breast277(0_0_9)2.csv", "crx653(3_3_9)2.csv", "flare1066(0_0_11)6.csv", 
 #           "german1000(0_7_13)2.csv", "saheart462(5_3_1)2.csv", ]


dataset_names = []
for dataset in datasets:
    dataset_names.append(get_dataset_name(dataset))

suspicious_models = [ 'monk', 'dermatology']

models = granular_models + granular_owa_models + known_ml_models + frnn_models
light_models = light_granular_models + known_ml_models + frnn_models
heavy_models = list(set(models) - set(light_models))

scores = pd.DataFrame(np.zeros((len(datasets), len(models))))
scores.columns = models
scores.index = dataset_names



oneHotEncoder = OneHotEncoder(sparse=False)

short_datasets = ["pima768(8_0_0)2.csv"]

#%%
models_to_run = models
scores = pd.DataFrame(np.zeros((len(datasets), len(models_to_run))))
scores.columns = models_to_run
scores.index = dataset_names


for data_name in datasets2:

    data = pd.read_csv(data_name)
    label_model = LabelEncoder()

    y = LabelEncoder().fit_transform(data.iloc[:, -1])
    X_df = data.iloc[:, :-1]
    n_samples = X_df.shape[0]  
    n_features = X_df.shape[1]

    feature_names = X_df.columns.values

    nominal_features = []
    for name in feature_names:
        if name[-3:] == '(N)':
            nominal_features.append(name)

    numerical_features = np.setdiff1d(feature_names, nominal_features)
    X_numerical = X_df[numerical_features].values
    X_nominal = X_df[nominal_features].values
    if X_nominal.shape[1] > 0:
        X_nominal = oneHotEncoder.fit_transform(X_nominal)

    X = np.concatenate((X_numerical, X_nominal), axis=1)



    #scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='balanced_accuracy')
    sample_weight = get_weights(y)
    granular_fit_params = {'fuzzygranularmulticlassclassifier__sample_weight': sample_weight}
    lr_fit_params = {'logisticregression__sample_weight': sample_weight}
    knn_fit_params = {'kneighborsclassifier__sample_weight': sample_weight}
    
    #model_gridsearch.fit(X, y, **fit_parameters)
    #model_gridsearch_lr.fit(X,y, **fit_parameters_lr)
    clear_name = get_dataset_name(data_name)
    print(clear_name)

    ##### GRANULAR MODELS ########

    if 'granular_model_mae_triangular_nnall' in models_to_run:
        start = time.time()
        granular_search_mae_triangular_nnall.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mae_triangular_nnall: ', granular_search_mae_triangular_nnall.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_triangular_nnall.best_params_)
        scores.loc[clear_name, 'granular_model_mae_triangular_nnall'] = np.round(granular_search_mae_triangular_nnall.best_score_, decimals=decimals)

    if 'granular_model_mae_triangular_nn20' in models_to_run:
        start = time.time()
        granular_search_mae_triangular_nn20.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mae_triangular_nn20: ', granular_search_mae_triangular_nn20.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_triangular_nn20.best_params_)
        scores.loc[clear_name, 'granular_model_mae_triangular_nn20'] = np.round(granular_search_mae_triangular_nn20.best_score_, decimals=decimals)
    
    if 'granular_model_mae_triangular_nn2' in models_to_run:
        start = time.time()
        granular_search_mae_triangular_nn2.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mae_triangular_nn2: ', granular_search_mae_triangular_nn2.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_triangular_nn2.best_params_)
        scores.loc[clear_name, 'granular_model_mae_triangular_nn2'] = np.round(granular_search_mae_triangular_nn2.best_score_, decimals=decimals)

    if 'granular_model_mae_quadratic_nnall' in models_to_run:
        start = time.time()
        granular_search_mae_quadratic_nnall.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mae_quadratic_nnall: ', granular_search_mae_quadratic_nnall.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_quadratic_nnall.best_params_)
        scores.loc[clear_name, 'granular_model_mae_quadratic_nnall'] = np.round(granular_search_mae_quadratic_nnall.best_score_, decimals=decimals)

    if 'granular_model_mae_quadratic_nn20' in models_to_run:
        start = time.time()
        granular_search_mae_quadratic_nn20.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mae_quadratic_nn20: ', granular_search_mae_quadratic_nn20.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_quadratic_nn20.best_params_)
        scores.loc[clear_name, 'granular_model_mae_quadratic_nn20'] = np.round(granular_search_mae_quadratic_nn20.best_score_, decimals=decimals)
    
    if 'granular_model_mae_quadratic_nn2' in models_to_run:
        start = time.time()
        granular_search_mae_quadratic_nn2.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mae_quadratic_nn2: ', granular_search_mae_quadratic_nn2.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_quadratic_nn2.best_params_)
        scores.loc[clear_name, 'granular_model_mae_quadratic_nn2'] = np.round(granular_search_mae_quadratic_nn2.best_score_, decimals=decimals)
    
    if 'granular_model_mse_triangular_nnall' in models_to_run:
        start = time.time()
        granular_search_mse_triangular_nnall.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mse_triangular_nnall: ', granular_search_mse_triangular_nnall.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_triangular_nnall.best_params_)
        scores.loc[clear_name, 'granular_model_mse_triangular_nnall'] = np.round(granular_search_mse_triangular_nnall.best_score_, decimals=decimals)

    if 'granular_model_mse_triangular_nn20' in models_to_run:
        start = time.time()
        granular_search_mse_triangular_nn20.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mse_triangular_nn20: ', granular_search_mse_triangular_nn20.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_triangular_nn20.best_params_)
        scores.loc[clear_name, 'granular_model_mse_triangular_nn20'] = np.round(granular_search_mse_triangular_nn20.best_score_, decimals=decimals)
    
    if 'granular_model_mse_triangular_nn2' in models_to_run:
        start = time.time()
        granular_search_mse_triangular_nn2.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mse_triangular_nn2: ', granular_search_mse_triangular_nn2.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_triangular_nn2.best_params_)
        scores.loc[clear_name, 'granular_model_mse_triangular_nn2'] = np.round(granular_search_mse_triangular_nn2.best_score_, decimals=decimals)

    if 'granular_model_mse_quadratic_nnall' in models_to_run:
        start = time.time()
        granular_search_mse_quadratic_nnall.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mse_quadratic_nnall: ', granular_search_mse_quadratic_nnall.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_quadratic_nnall.best_params_)
        scores.loc[clear_name, 'granular_model_mse_quadratic_nnall'] = np.round(granular_search_mse_quadratic_nnall.best_score_, decimals=decimals)

    if 'granular_model_mse_quadratic_nn20' in models_to_run:
        start = time.time()
        granular_search_mse_quadratic_nn20.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mse_quadratic_nn20: ', granular_search_mse_quadratic_nn20.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_quadratic_nn20.best_params_)
        scores.loc[clear_name, 'granular_model_mse_quadratic_nn20'] = np.round(granular_search_mse_quadratic_nn20.best_score_, decimals=decimals)
    
    if 'granular_model_mse_quadratic_nn2' in models_to_run:
        start = time.time()
        granular_search_mse_quadratic_nn2.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_model_mse_quadratic_nn2: ', granular_search_mse_quadratic_nn2.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_quadratic_nn2.best_params_)
        scores.loc[clear_name, 'granular_model_mse_quadratic_nn2'] = np.round(granular_search_mse_quadratic_nn2.best_score_, decimals=decimals)


    ########## OWA BASED GRANULAR ############


    if 'granular_model_mae_triangular_add' in models_to_run:
        start = time.time()
        granular_search_mae_triangular_add.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mae_triangular_add: ', granular_search_mae_triangular_add.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_triangular_add.best_params_)
        scores.loc[clear_name, 'granular_model_mae_triangular_add'] = np.round(granular_search_mae_triangular_add.best_score_, decimals=decimals)

    if 'granular_model_mae_quadratic_add' in models_to_run:  
        start = time.time()   
        granular_search_mae_quadratic_add.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mae_quadratic_add: ', granular_search_mae_quadratic_add.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_quadratic_add.best_params_)
        scores.loc[clear_name, 'granular_model_mae_quadratic_add'] = np.round(granular_search_mae_quadratic_add.best_score_, decimals=decimals)

    if 'granular_model_mse_triangular_add' in models_to_run:
        start = time.time()    
        granular_search_mse_triangular_add.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mse_triangular_add: ', granular_search_mse_triangular_add.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_triangular_add.best_params_)
        scores.loc[clear_name, 'granular_model_mse_triangular_add'] = np.round(granular_search_mse_triangular_add.best_score_, decimals=decimals)

    if 'granular_model_mse_quadratic_add' in models_to_run: 
        start = time.time()   
        granular_search_mse_quadratic_add.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mse_quadratic_add: ', granular_search_mse_quadratic_add.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_quadratic_add.best_params_)
        scores.loc[clear_name, 'granular_model_mse_quadratic_add'] = np.round(granular_search_mse_quadratic_add.best_score_, decimals=decimals)

    if 'granular_model_mae_triangular_exp' in models_to_run:
        start = time.time() 
        granular_search_mae_triangular_exp.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mae_triangular_exp: ', granular_search_mae_triangular_exp.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_triangular_exp.best_params_)
        scores.loc[clear_name, 'granular_model_mae_triangular_exp'] = np.round(granular_search_mae_triangular_exp.best_score_, decimals=decimals)

    if 'granular_model_mae_quadratic_exp' in models_to_run: 
        start = time.time()
        granular_search_mae_quadratic_exp.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mae_quadratic_exp: ', granular_search_mae_quadratic_exp.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_quadratic_exp.best_params_)
        scores.loc[clear_name, 'granular_model_mae_quadratic_exp'] = np.round(granular_search_mae_quadratic_exp.best_score_, decimals=decimals)

    if 'granular_model_mse_triangular_exp' in models_to_run: 
        start = time.time()
        granular_search_mse_triangular_exp.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mse_triangular_exp: ', granular_search_mse_triangular_exp.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_triangular_exp.best_params_)
        scores.loc[clear_name, 'granular_model_mse_triangular_exp'] = np.round(granular_search_mse_triangular_exp.best_score_, decimals=decimals)

    if 'granular_model_mse_quadratic_exp' in models_to_run: 
        start = time.time()
        granular_search_mse_quadratic_exp.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mse_quadratic_exp: ', granular_search_mse_quadratic_exp.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_quadratic_exp.best_params_)
        scores.loc[clear_name, 'granular_model_mse_quadratic_exp'] = np.round(granular_search_mse_quadratic_exp.best_score_, decimals=decimals)

    if 'granular_model_mae_triangular_invadd' in models_to_run: 
        start = time.time()
        granular_search_mae_triangular_invadd.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mae_triangular_invadd: ', granular_search_mae_triangular_invadd.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_triangular_invadd.best_params_)
        scores.loc[clear_name, 'granular_model_mae_triangular_invadd'] = np.round(granular_search_mae_triangular_invadd.best_score_, decimals=decimals)

    if 'granular_model_mae_quadratic_invadd' in models_to_run: 
        start = time.time()
        granular_search_mae_quadratic_invadd.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mae_quadratic_invadd: ', granular_search_mae_quadratic_invadd.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mae_quadratic_invadd.best_params_)
        scores.loc[clear_name, 'granular_model_mae_quadratic_invadd'] = np.round(granular_search_mae_quadratic_invadd.best_score_, decimals=decimals)

    if 'granular_model_mse_triangular_invadd' in models_to_run: 
        start = time.time()
        granular_search_mse_triangular_invadd.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mse_triangular_invadd: ', granular_search_mse_triangular_invadd.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_triangular_invadd.best_params_)
        scores.loc[clear_name, 'granular_model_mse_triangular_invadd'] = np.round(granular_search_mse_triangular_invadd.best_score_, decimals=decimals)

    if 'granular_model_mse_quadratic_invadd' in models_to_run:
        start = time.time()
        granular_search_mse_quadratic_invadd.fit(X, y)
        end = time.time()
        exec_time = end-start
        print('granular_search_mse_quadratic_invadd: ', granular_search_mse_quadratic_invadd.best_score_)
        print('exec time: ', exec_time)
        print(granular_search_mse_quadratic_invadd.best_params_)
        scores.loc[clear_name, 'granular_model_mse_quadratic_invadd'] = np.round(granular_search_mse_quadratic_invadd.best_score_, decimals=decimals)


    ############ KNOWN ML MODELS ####################

    if 'lr_model' in models_to_run:  
        lr_search.fit(X, y)
        print('lr: ', lr_search.best_score_)
        print(lr_search.best_params_)
        scores.loc[clear_name, 'lr_model'] = np.round(lr_search.best_score_, decimals=decimals)

    if 'knn_model' in models_to_run:  
        knn_search.fit(X, y)
        print('knn: ', knn_search.best_score_)
        print(knn_search.best_params_)
        scores.loc[clear_name, 'knn_model'] = np.round(knn_search.best_score_, decimals=decimals)

    if 'lvq_model' in models_to_run:  
        lvq_search.fit(X, y)
        print('lvq: ', lvq_search.best_score_)
        print(lvq_search.best_params_)
        scores.loc[clear_name, 'lvq_model'] = np.round(lvq_search.best_score_, decimals=decimals)

    if 'dt_model' in models_to_run:  
        dt_search.fit(X, y) 
        print('dt: ', dt_search.best_score_)
        print(dt_search.best_params_)
        scores.loc[clear_name, 'dt_model'] = np.round(dt_search.best_score_, decimals=decimals)

    ########## FRNN MODELS ######################

    if 'kFRNN_model_add_triangular' in models_to_run:  
        kFRNN_search_add_triangular.fit(X, y)
        print('kFRNN_model_add_triangular: ', kFRNN_search_add_triangular.best_score_)
        print(kFRNN_search_add_triangular.best_params_)
        scores.loc[clear_name, 'kFRNN_model_add_triangular'] = np.round(kFRNN_search_add_triangular.best_score_, decimals=decimals)

    if 'kFRNN_model_exp_triangular' in models_to_run:  
        kFRNN_search_exp_triangular.fit(X, y)
        print('kFRNN_model_exp_triangular: ', kFRNN_search_exp_triangular.best_score_)
        print(kFRNN_search_exp_triangular.best_params_)
        scores.loc[clear_name, 'kFRNN_model_exp_triangular'] = np.round(kFRNN_search_exp_triangular.best_score_, decimals=decimals)

    if 'kFRNN_model_invadd_triangular' in models_to_run:
        kFRNN_search_invadd_triangular.fit(X, y)
        print('kFRNN_model_invadd_triangular: ', kFRNN_search_invadd_triangular.best_score_)
        print(kFRNN_search_invadd_triangular.best_params_)
        scores.loc[clear_name, 'kFRNN_model_invadd_triangular'] = np.round(kFRNN_search_invadd_triangular.best_score_, decimals=decimals)

    if 'kFRNN_model_strict_triangular' in models_to_run:
        kFRNN_search_strict_triangular.fit(X, y)
        print('kFRNN_model_strict_triangular: ', kFRNN_search_strict_triangular.best_score_)
        print(kFRNN_search_strict_triangular.best_params_)
        scores.loc[clear_name, 'kFRNN_model_strict_triangular'] = np.round(kFRNN_search_strict_triangular.best_score_, decimals=decimals)
    
    ##################################

    if 'kFRNN_model_add_quadratic' in models_to_run:  
        kFRNN_search_add_quadratic.fit(X, y)
        print('kFRNN_model_add_quadratic: ', kFRNN_search_add_quadratic.best_score_)
        print(kFRNN_search_add_quadratic.best_params_)
        scores.loc[clear_name, 'kFRNN_model_add_quadratic'] = np.round(kFRNN_search_add_quadratic.best_score_, decimals=decimals)

    if 'kFRNN_model_exp_quadratic' in models_to_run:  
        kFRNN_search_exp_quadratic.fit(X, y)
        print('kFRNN_model_exp_quadratic: ', kFRNN_search_exp_quadratic.best_score_)
        print(kFRNN_search_exp_quadratic.best_params_)
        scores.loc[clear_name, 'kFRNN_model_exp_quadratic'] = np.round(kFRNN_search_exp_quadratic.best_score_, decimals=decimals)

    if 'kFRNN_model_invadd_quadratic' in models_to_run:
        kFRNN_search_invadd_quadratic.fit(X, y)
        print('kFRNN_model_invadd_quadratic: ', kFRNN_search_invadd_quadratic.best_score_)
        print(kFRNN_search_invadd_quadratic.best_params_)
        scores.loc[clear_name, 'kFRNN_model_invadd_quadratic'] = np.round(kFRNN_search_invadd_quadratic.best_score_, decimals=decimals)

    if 'kFRNN_model_strict_quadratic' in models_to_run:
        kFRNN_search_strict_quadratic.fit(X, y)
        print('kFRNN_model_strict_quadratic: ', kFRNN_search_strict_quadratic.best_score_)
        print(kFRNN_search_strict_quadratic.best_params_)
        scores.loc[clear_name, 'kFRNN_model_strict_quadratic'] = np.round(kFRNN_search_strict_quadratic.best_score_, decimals=decimals)

    scores.to_csv('results_rand10.csv')

# %%
