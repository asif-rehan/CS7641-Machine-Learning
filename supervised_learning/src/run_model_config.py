import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import time
import datetime
import os

from utils import *

this_dir =  os.path.dirname(__file__)


def vanilla_fit(X_train, y_train, pipe):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
        pipe.fit(X_train, y_train)
    train_score = pipe.score(X_train, y_train)
    print('train score before hyper-parameter tuning', train_score)
    # print(accuracy_score(y_test, pipe.predict(X_test)))
    return train_score




def decision_tree_experiments(X_train, y_train, data='wine'):
    pipe = Pipeline([('std', StandardScaler()), ('cfr', DecisionTreeClassifier(random_state=42))])
    vanilla_fit(X_train, y_train, pipe)
    
    min_samples_split_range = np.arange(2, 300, 30)
    min_samples_leaf_range = np.arange(1, 160, 20)
    ccp_alpha_range = np.linspace(0, 0.1, 5)
    max_depth_range = np.arange(2, 25, 5)

	#===============================================================================================
    #min_samples_split_range = np.arange(2, 300, 150)
    #min_samples_leaf_range = np.arange(2, 160, 80)
    #ccp_alpha_range = np.linspace(0, 0.1, 2)
    #max_depth_range = np.arange(2, 10, 5)  
    #===============================================================================================
      
    generate_validation_curve(pipe, X_train, y_train, model='Decision Tree',
                              param_name="min_samples_leaf",
                              search_range=min_samples_leaf_range, data=data)
    generate_validation_curve(pipe, X_train, y_train, model='Decision Tree',
                              param_name="min_samples_split",
                              search_range=min_samples_split_range , data=data)    
    generate_validation_curve(pipe, X_train, y_train, model='Decision Tree',
                          param_name="ccp_alpha",
                          search_range=ccp_alpha_range , data=data)    
    generate_validation_curve(pipe, X_train, y_train, model='Decision Tree',
                          param_name="max_depth",
                          search_range=max_depth_range , data=data)    
    
    tuned_model = tune_hyperparameter({'cfr__min_samples_split': min_samples_split_range,
                                          "cfr__min_samples_leaf": min_samples_leaf_range,
                                          "cfr__ccp_alpha": ccp_alpha_range,
                                          "cfr__max_depth": np.arange(2, 12, 2)
                                          },
                                       pipe, X_train, y_train)
    return tuned_model    


def knn_experiments(X_train, y_train, data='wine'):
    pipe = Pipeline([('std', StandardScaler()), ('cfr', KNeighborsClassifier())])
    vanilla_fit(X_train, y_train, pipe)
    
    n_neighbors_range = np.arange(1, 50, 10)
    p_range = np.linspace(1, 3, 5)
    weights = ['uniform', 'distance']
    #===============================================================================================
    #n_neighbors_range = np.arange(1, 50, 25)
    #p_range = np.linspace(1, 3, 2)
    #weights = ['uniform', 'distance']
    #===============================================================================================
    generate_validation_curve(pipe, X_train, y_train, model='kNN',
                              param_name="n_neighbors",
                              search_range=n_neighbors_range, data=data)
    generate_validation_curve(pipe, X_train, y_train, model='kNN',
                              param_name="p",
                              search_range=p_range , data=data)    
    generate_validation_curve(pipe, X_train, y_train, model='kNN',
                          param_name="weights",
                          search_range=weights , data=data)    

    tuned_model = tune_hyperparameter({'cfr__n_neighbors': n_neighbors_range,
                                      'cfr__weights': weights,
                                      'cfr__p': p_range
                                      },
                                       pipe, X_train, y_train)
    return tuned_model 


def neural_network_experiments(X_train, y_train, data='wine'):
    pipe = Pipeline([('std', StandardScaler()), ('cfr', MLPClassifier(random_state=42, max_iter=100))])
    vanilla_fit(X_train, y_train, pipe)
    
    activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha = np.logspace(-2, 4, 6)
    hidden_layer_sizes = [(6, 4, 2), (10,5,2), (12,6,4,2)]
    #===============================================================================================
    #activation = ['relu']
    #alpha = np.logspace(-4, 4, 2)
    #===============================================================================================
    
    generate_validation_curve(pipe, X_train, y_train, model='Neural Network',
                              param_name="activation",
                              search_range=activation, data=data)
    
    generate_loss_learning_curve(X_train, y_train, model='Neural Network',
                              param_name="activation",
                              search_range=activation, data=data)
    
    
    generate_validation_curve(pipe, X_train, y_train, model='Neural Network',
                              param_name="alpha",
                              search_range=alpha, data=data, log=True)
    
    generate_loss_learning_curve(X_train, y_train, model='Neural Network',
                              param_name="alpha",
                              search_range=alpha, data=data)
    
    generate_validation_curve(pipe, X_train, y_train, model='Neural Network',
                              param_name="hidden_layer_sizes",
                              search_range=hidden_layer_sizes, data=data)
    
    generate_loss_learning_curve(X_train, y_train, model='Neural Network',
                              param_name="hidden_layer_sizes",
                              search_range=hidden_layer_sizes, data=data)
    
    tuned_model = tune_hyperparameter({'cfr__alpha': alpha,
                                       'cfr__hidden_layer_sizes' : hidden_layer_sizes
                                       },
                                       pipe, X_train, y_train)
    
    
    return tuned_model    


def support_vector_machine_experiments(X_train, y_train, data='wine'):
    pipe = Pipeline([('std', StandardScaler()), ('cfr', SVC(random_state=42))])
    vanilla_fit(X_train, y_train, pipe)
    
    C_range = np.logspace(-2, 3, 5)
    gamma_range = np.logspace(-6, -1, 4)
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    
    
    #===============================================================================================
    #
    #C_range = np.logspace(-2, 3, 2)
    #gamma_range = np.logspace(-6, -1, 2)
    #kernel_options = ['linear']
    # 
    #===============================================================================================
    
    
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine',
                              param_name="C",
                              search_range=C_range, data=data, log=True)
    
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine',
                                  param_name="gamma",
                                  search_range=gamma_range, data=data, log=True)  
      
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine',
                                  param_name="kernel",
                                  search_range=kernel_options, data=data)    
    
    tuned_model = tune_hyperparameter({'cfr__C': C_range,
                                      'cfr__kernel': kernel_options,
                                      'cfr__gamma': gamma_range},
                                       pipe, X_train, y_train)
    return tuned_model    


def boosted_tree_experiments(base, X_train, y_train, data='wine'):
    pipe = Pipeline([('std', StandardScaler()),
                 ('cfr', AdaBoostClassifier(base_estimator=base, random_state=42))])
    vanilla_fit(X_train, y_train, pipe)
    
    n_estimators_range = np.arange(1, 1000, 100)
    learning_rate_range = np.linspace(0.01, 1, 5)
    
    #===============================================================================================
    #
    #n_estimators_range = np.arange(1,3, 1)
    #learning_rate_range = np.linspace(0.01, 1, 1)
    # 
    #===============================================================================================
    
    
    generate_validation_curve(pipe, X_train, y_train, model='Boosted Tree',
                              param_name="n_estimators",
                              search_range=n_estimators_range, data=data)
    generate_validation_curve(pipe, X_train, y_train, model='Boosted Tree',
                              param_name="learning_rate",
                              search_range=learning_rate_range, data=data)
    
    tuned_model = tune_hyperparameter({'cfr__learning_rate': learning_rate_range,
              'cfr__n_estimators':     np.arange(1, 500, 100)
              },
                                       pipe, X_train, y_train)
    return tuned_model    




