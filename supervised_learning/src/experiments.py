import warnings

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
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
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')











def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    '''
    Code taken from (Reference): Scikit-Learn example 
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    '''
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



def vanilla_fit(X_train, y_train, pipe):
    pipe.fit(X_train, y_train)
    train_score = pipe.score(X_train, y_train)
    print(train_score)
    #print(accuracy_score(y_test, pipe.predict(X_test)))
    return train_score


def generate_validation_curve(pipe, X_train, y_train, model, param_name, search_range,   data='wine'):
    train_scores, test_scores = validation_curve(estimator=pipe, X=X_train, y=y_train, 
                                                 param_name="cfr__"+param_name,
                                                 param_range=search_range, cv=5,
                                                 n_jobs=-1)
    plt.figure(figsize=(5,5))
    plt.plot(search_range, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(search_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for{}'.format(model))
    plt.xlabel(param_name)
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()

    plt.savefig(r'..\plot\{}_val_curve_{}_{}.png'.format(data, model, param_name), figsize=(5, 5))
    plt.show()
    
def tune_hyperparameter(param_grid, pipe, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, random_state=42)
    tuned_model = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1)
    tuned_model.fit(X_train, y_train)
    print("Tuned params: {}".format(tuned_model.best_params_))
    
    ypred = tuned_model.predict(X_train)
    print(classification_report(y_train, ypred))
    return tuned_model

def decision_tree_experiments(X_train, y_train, data='wine'):
    pipe = Pipeline([('std', StandardScaler()), ('cfr', DecisionTreeClassifier())])
    vanilla_fit(X_train, y_train, pipe)
    
    
    min_samples_split_range = np.arange(1, 300, 10)
    min_samples_leaf_range = np.arange(1, 160, 10)
    ccp_alpha_range = np.linspace(0, 0.1, 10)
    max_depth_range = np.arange(2, 25, 1)
    
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
                                          "cfr__max_depth": max_depth_range
                                          }, 
                                       pipe, X_train, y_train)
    return tuned_model    

def knn_experiments(X_train, y_train, data='wine'):
    pipe =  Pipeline([('std', StandardScaler()), ('cfr', KNeighborsClassifier())])
    vanilla_fit(X_train, y_train, pipe)
    
    
    n_neighbors_range = np.arange(1, 50, 5)
    p_range = np.linspace(1, 3, 8)
    weights = ['uniform', 'distance']
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
    pipe =  Pipeline([('std', StandardScaler()), ('cfr', MLPClassifier())])
    vanilla_fit(X_train, y_train, pipe)
    
    
    activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha = np.logspace(-4, 4, 9)
    
    generate_validation_curve(pipe, X_train, y_train, model='Neural Network', 
                              param_name="activation", 
                              search_range=activation, data=data)
    generate_validation_curve(pipe, X_train, y_train, model='Neural Network', 
                              param_name="alpha", 
                              search_range=alpha, data=data)
    tuned_model = tune_hyperparameter({'cfr__alpha': alpha,
                                       }, 
                                       pipe, X_train, y_train)
    return tuned_model    

def support_vector_machine_experiments(X_train, y_train, data='wine'):
    pipe =  Pipeline([('std', StandardScaler()), ('cfr', SVC())])
    vanilla_fit(X_train, y_train, pipe)
    
    C_range = np.logspace(-2, 3, 6)
    gamma_range = np.logspace(-6, -1, 5)
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine', 
                              param_name="C", 
                              search_range=C_range, data=data)
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine', 
                                  param_name="gamma", 
                                  search_range=gamma_range, data=data)    
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine', 
                                  param_name="kernel", 
                                  search_range=kernel_options, data=data)    
    
    
    tuned_model = tune_hyperparameter({'cfr__C': C_range,
                                      'cfr__kernel': kernel_options,
                                      'cfr__gamma': gamma_range}, 
                                       pipe, X_train, y_train)
    return tuned_model    

def boosted_tr_experiments(base, X_train, y_train, data='wine'):
    pipe =  Pipeline([('std', StandardScaler()), 
                 ('cfr', AdaBoostClassifier(base_estimator=base))])
    vanilla_fit(X_train, y_train, pipe)
    
    C_range = np.logspace(-2, 3, 6)
    gamma_range = np.logspace(-6, -1, 5)
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine', 
                              param_name="C", 
                              search_range=C_range, data=data)
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine', 
                                  param_name="gamma", 
                                  search_range=gamma_range, data=data)    
    generate_validation_curve(pipe, X_train, y_train, model='Support Vector Machine', 
                                  param_name="kernel", 
                                  search_range=kernel_options, data=data)    
    
    
    tuned_model = tune_hyperparameter({'cfr__C': C_range,
                                      'cfr__kernel': kernel_options,
                                      'cfr__gamma': gamma_range}, 
                                       pipe, X_train, y_train)
    return tuned_model    



def get_dataset1(verbose=True):
    df = pd.read_csv(r'..\data\winequality-white.csv', sep=';')
    y1 = df['quality'] >= 7
    X1 = df.drop(columns='quality')
    
    if verbose==True:
        print(df.shape)
        print(df.info())
        print(df.shape)
        print(df.info())
        print(df.describe())
        df.describe()
        plt.rcParams["figure.figsize"] = 20, 20
        # Basic correlogram
        sns.pairplot(df)
        plt.show()
        plt.savefig(r'..\plot\wine_scatterplot.png', figsize=(20, 20))
        
        print(X1.shape)
        print(y1.shape)
        print(X1.head())
        print(y1)

    return X1, y1

def get_dataset2():
    X2 = None
    y2 = None
    return X2, y2



def main(verbose):
    
    X1, y1 = get_dataset1(verbose)
    X2, y2 = get_dataset2()
    
    
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)
    
    '''
    decision_tree:
        vanilla fit
        validation curves
        gridsearch
        learning curves
    '''
    tuned_decision_tree_model_wine = decision_tree_experiments(X_train1, y_train1, data='wine')
    tuned_decision_tree_model_dataset2 = decision_tree_experiments(X_train1, y_train1, data='dataset2')
    generate_learning_curves(tuned_decision_tree_model_wine, 
                             tuned_decision_tree_model_dataset2,
                             X_train1, y_train1, X_train2, y_train2)
    
    tuned_knn_model_wine = knn_experiments(X_train1, y_train1, data='wine')
    tuned_knn_model_dataset2 = knn_experiments(X_train1, y_train1, data='dataset2')
    generate_learning_curves(tuned_knn_model_wine, 
                             tuned_knn_model_dataset2,
                             X_train1, y_train1, X_train2, y_train2)
    
    tuned_neural_network_model_wine = neural_network_experiments(X_train1, y_train1, data='wine')
    tuned_neural_network_model_dataset2 = neural_network_experiments(X_train1, y_train1, data='dataset2')
    generate_learning_curves(tuned_neural_network_model_wine, 
                             tuned_neural_network_model_dataset2,
                             X_train1, y_train1, X_train2, y_train2)
    
    tuned_support_vector_machine_model_wine = support_vector_machine_experiments(X_train1, y_train1, data='wine')
    tuned_support_vector_machine_model_dataset2 = support_vector_machine_experiments(X_train1, y_train1, data='dataset2')
    generate_learning_curves(tuned_support_vector_machine_model_wine, 
                             tuned_support_vector_machine_model_dataset2,
                             X_train1, y_train1, X_train2, y_train2)
    
    
def generate_learning_curves(tuned_model_1, tuned_model_2, X_train1, y_train1, X_train2, y_train2):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    plot_learning_curve(tuned_model_1.best_estimator_
                        , 'Wine Data', 
                        X_train1, y_train1, axes=axes[:, 0], ylim=(0.7, 1.01),
                        cv=StratifiedKFold(n_splits=5, random_state=42), n_jobs=-1)
    plot_learning_curve(tuned_model_2.best_estimator_
                        , 'Dataset2', 
                        X_train2, y_train2, axes=axes[:, 1], ylim=(0.7, 1.01),
                        cv=StratifiedKFold(n_splits=5, random_state=42), n_jobs=-1)
    plt.show()
    
    
    

    
    



if __name__=='__main__':
    main()
