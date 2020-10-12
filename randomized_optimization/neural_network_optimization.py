

from sklearn.model_selection import train_test_split
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import datetime

this_dir =  os.path.dirname(__file__)

nn_dir = os.path.join(this_dir, 'nn_opt')
if not os.path.exists(nn_dir):
    os.makedirs(nn_dir, False)

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
models = {}

algos = {
        'SA': { 'algorithm': ['simulated_annealing'],
                'schedule': [mlrose.GeomDecay(init_temp=1, decay=0.99),
                          mlrose.GeomDecay(init_temp=100, decay=0.99),
                          mlrose.GeomDecay(init_temp=1000, decay=0.99),
                          mlrose.ExpDecay(init_temp=1, exp_const=0.05), 
                          mlrose.ExpDecay(init_temp=100, exp_const=0.05), 
                          mlrose.ExpDecay(init_temp=1000, exp_const=0.05), 
                          ],
                    'max_iters' : [3000],
                  'learning_rate': [0.5],
                  'activation': ['relu', 'sigmoid', 'tanh'], 
                   'clip_max': [10.0]
              },
            'GA': {'algorithm':[ 'genetic_alg'],
                    'pop_size' : [100, 500],
                     'mutation_prob': [0.3, 0.5],
                    'max_iters' : [3000],
                  'learning_rate': [0.5],
                  'activation': ['relu', 'sigmoid', 'tanh'], 
                   'clip_max': [10.0]
                  },
            'RHC': {'algorithm': ['random_hill_climb'],
                    'restarts': [10],
                    'max_iters' : [3000],
                  'learning_rate': [0.5],
                  'activation': ['relu', 'sigmoid', 'tanh'], 
                   'clip_max': [10.0]
                   },
            'GD': {'algorithm': ['gradient_descent'],
                    'max_iters' : [2000],
                  'learning_rate':[0.05],
                  'activation': ['relu', 'sigmoid', 'tanh'], 
                   'clip_max': [10000000.0]
                  }
        }

algorithms = ['RHC', 'SA', 'GA', 'GD']


for algo in algorithms:
    classifier =  mlrose.NeuralNetwork(hidden_nodes=hidden_layers,
              bias=True, is_classifier=True, early_stopping=True, 
              clip_max=10.0, max_attempts=100, random_state=42, curve=True)

    
    models[algo] = tune_hyperparameter(algos[algo],
                                       classifier, X_train, y_train)


print(models)
for algo in models:
    with open(os.path.join(this_dir, 'nn_opt', algo+'_nn.pkl'), 'wb') as f: 
        pickle.dump(models[algo], f)


fig, ax = plt.subplots(1,4, figsize=(20,3))

for i in range(len(algorithms)):    
    model = algorithms[i] 
    loss = models[model].best_estimator_.fitness_curve
    #if model == 'GD':
    #    loss = loss * -1
    #print("Length of Iteration", len(loss))
   
    #df = pd.DataFrame(models[model].cv_results_)
    #print(df.info())
    #print(df)
    ax[i].plot(loss, label='Best '+model)
    ax[i].set_xscale('log')
    ax[i].set_ylabel(' Loss Function')
    ax[i].set_xlabel('Iterations')
    ax[i].grid()
    ax[i].legend()
    
plt.suptitle('Fitness Over Iterations')

plt.legend()

plt.savefig(os.path.join(this_dir, 'nn_opt', 'nn_loss.png'))
plt.show()
plt.close()

results = []


for model in models:
    #print("MODEL NAME: ", model, "\n==========")
    
    #print("Train Time: ",  models[model].refit_time_)

    train_time.append(models[model].refit_time_)
    train_f1 = models[model].score(X_train, y_train)
    #print("Train F1 Score: ", train_f1)

    start = datetime.datetime.now()
    y_test_pred = models[model].predict(X_test)
    pred_time = (datetime.datetime.now() - start).microseconds/1000
    #print("Test Prediction Time: ", pred_time)

    test_f1 = models[model].score(X_test, y_test)
    #print("Test F1 Score: ", test_f1)
    
    iter_reqd = len(models[model].best_estimator_.fitness_curve)
    #print("Iter Reqd: ", iter_reqd)
    
    #print("Final F1 Score (loss): ", models[model].best_estimator_.loss)
    results.append([model, models[model].refit_time_, iter_reqd, models[model].best_estimator_.loss, train_f1, test_f1])

results_df = pd.DataFrame(results, columns=['Algorithm', 'Train Time(s)', '#Iterations Reqd', 'Final Loss Value', 'Train F1 Score', 'Test F1 Score'])
results_df.set_index('Algorithm', inplace=True)
print(results_df)

fig, ax = plt.subplots(1,4, figsize=(17,3))
results_df[['Train F1 Score', 'Test F1 Score']].plot.bar(ax=ax[3])
ax[3].set_ylabel('F1 Score')
ax[3].set_title('F1 Scores')
ax[3].grid()
#plt.savefig(os.path.join(this_dir, 'nn_opt', 'train_time.png'))
#plt.show()

#plt.close()

results_df[['Train Time(s)']].plot.bar(ax=ax[0])
ax[0].set_ylabel('Time (sec)')
ax[0].set_title('Training Time using Best Parameters')
ax[0].grid()


#fig, ax = plt.subplots(2,2, figsize=(10,10))
results_df['#Iterations Reqd'].plot.bar(ax=ax[1])
ax[1].set_ylabel('#Iterations Reqd')
ax[1].set_title('#Iterations Reqd')
ax[1].grid()


results_df['Final Loss Value'].plot.bar(ax=ax[2])
ax[2].set_ylabel('Loss Function')
ax[2].set_title('Final Loss Value')
ax[2].grid()


plt.savefig(os.path.join(this_dir, 'nn_opt', 'nn_final.png'))


plt.show()
plt.close()





    
    

