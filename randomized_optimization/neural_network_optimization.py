

from sklearn.model_selection import train_test_split
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


this_dir =  os.path.dirname(__file__)
os.makedirs(os.path.join(this_dir, 'nn_opt'))

def get_dataset(verbose=True):
    df = pd.read_csv(os.path.join(this_dir,  "data", "diabetes.csv"))
    y = df['Outcome']
    X = df.drop(columns='Outcome')        
    return X, y


def tune_hyperparameter(param_grid, pipe, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, random_state=42)
    tuned_model = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring='f1', return_train_score=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
        tuned_model.fit(X_train, y_train)
    print("Tuned params: {}".format(tuned_model.best_params_))
    
    ypred = tuned_model.predict(X_train)
    print('refit train metrics=\n', classification_report(y_train, ypred))
    return tuned_model


X2, y2 = get_dataset(verbose=False)

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

#transform data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
hidden_layers = [12,6,4,2]



algorithms = ['RHC', 'SA', 'GA', 'GD']
algos = {
        'SA': { 'algorithm': ['simulated_annealing'],
                'schedule': [mlrose.GeomDecay(init_temp=10, decay=0.99),
                          mlrose.GeomDecay(init_temp=100, decay=0.99),
                          mlrose.GeomDecay(init_temp=1000, decay=0.99),
                          mlrose.ExpDecay(init_temp=10, exp_const=0.05), 
                          mlrose.ExpDecay(init_temp=100, exp_const=0.05), 
                          mlrose.ExpDecay(init_temp=1000, exp_const=0.05), 
                          ],
                    'max_iters' : [2**14],
                  'learning_rate': [0.01]
              },
            'GA': {'algorithm':[ 'genetic_alg'],
                    'pop_size' : [100, 200, 300],
                     'mutation_prob': [0.2, 0.3, 0.5],
                    'max_iters' : [2**13],
                  'learning_rate': [0.1]
                  },
            'RHC': {'algorithm': ['random_hill_climb'],
                    'restarts': np.arange(1,21, 5),
                    'max_iters' : [2**14],
                  'learning_rate': [0.01]
                   },
            'GD': {'algorithm': ['gradient_descent'],
                    'max_iters' : [2**13],
                  'learning_rate':[0.01]
                  }
        }
models = {}
for algo in algorithms:
    classifier =  mlrose.NeuralNetwork(hidden_nodes=hidden_layers, activation='tanh', 
              bias=True, is_classifier=True, early_stopping=True, 
              clip_max=1000000.0, max_attempts=10, random_state=42, curve=True)

    
    models[algo] = tune_hyperparameter(algos[algo],
                                       classifier, X_train, y_train)
    
    print("train f1 score", models[algo].score(X_train, y_train))
    print("test f1 score", models[algo].score(X_test, y_test))

    
for model in models:
    print("MODEL NAME: ", model, "\n==========")
    print("Final loss value: ", models[model].best_estimator_.loss)
    print("Refit Time: ", models[model].refit_time_)
    loss = models[model].best_estimator_.fitness_curve
    if model == 'GD':
        loss = loss * -1
    print("Length of Iteration", len(loss))
    df = pd.DataFrame(models[model].cv_results_)
    #print(df.info())
    #print(df)
    plt.plot(loss - loss[0], label=model+'_fitness_curve')
    plt.xscale('log')
    
plt.legend()
plt.show()
    