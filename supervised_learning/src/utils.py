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
import os
import datetime

this_dir =  os.path.dirname(__file__)

    
def tune_hyperparameter(param_grid, pipe, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, random_state=42)
    tuned_model = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1)
    tuned_model.fit(X_train, y_train)
    print("Tuned params: {}".format(tuned_model.best_params_))
    
    ypred = tuned_model.predict(X_train)
    print(classification_report(y_train, ypred))
    return tuned_model


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
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

def generate_learning_curves(tuned_model_1, tuned_model_2, X_train1, y_train1, X_train2, y_train2, model):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    plot_learning_curve(tuned_model_1.best_estimator_
                        , 'Wine Data',
                        X_train1, y_train1, axes=axes[:, 0], ylim=(0.7, 1.01),
                        cv=StratifiedKFold(n_splits=5, random_state=42), n_jobs=-1)
    plot_learning_curve(tuned_model_2.best_estimator_
                        , 'Dataset2',
                        X_train2, y_train2, axes=axes[:, 1], ylim=(0.7, 1.01),
                        cv=StratifiedKFold(n_splits=5, random_state=42), n_jobs=-1)
    plt.savefig(os.path.join(this_dir,os.pardir, "plot", '{}_learning_curve.png'.format(model)))
    #plt.show()



def generate_validation_curve(pipe, X_train, y_train, model, param_name, search_range, data='wine'):
    train_scores, test_scores = validation_curve(estimator=pipe, X=X_train, y=y_train,
                                                 param_name="cfr__" + param_name,
                                                 param_range=search_range, cv=5,
                                                 n_jobs=-1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(search_range, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(search_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for {}'.format(model))
    plt.xlabel(param_name)
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    


    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig(os.path.join(this_dir,this_dir, os.pardir, "plot", 
                        '{}_val_curve_{}_{}.png'.format(data, model, param_name)), figsize=(5, 5))
    #plt.show()
