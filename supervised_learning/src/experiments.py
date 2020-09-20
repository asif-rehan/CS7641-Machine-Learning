import sys
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
warnings.filterwarnings('ignore')
    
from warnings import simplefilter
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

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
import pickle

from get_data import get_dataset1, get_dataset2
from run_model_config import *
from utils import *

this_dir =  os.path.dirname(__file__)



def test_score(result_df, X_train, y_train, X_test, y_test, data):
    result_df['Training Time (sec)'] = ""
    result_df['Training F1 Score'] = ""
    result_df['Testing Time (sec)'] = ""
    result_df['Testing F1 Score'] = ""
    
    for index, row in result_df.iterrows():
        tuned_model = row['Tuned Model']
        start = datetime.datetime.now()
        tuned_model.fit(X_train, y_train)
        end = datetime.datetime.now()
        training_time = (end - start).microseconds / (1000*1000)          #1
        result_df.at[index, 'Training Time (sec)'] = training_time
        training_score = tuned_model.score(X_train, y_train)
        #print(training_score)
        result_df.at[index, 'Training F1 Score'] = training_score        #2 
        
        start = datetime.datetime.now()
        test_score = tuned_model.score(X_test, y_test)
        #print(test_score)
        result_df.at[index, 'Testing F1 Score'] = test_score            #3
        end = datetime.datetime.now()
        testing_time = (end - start).microseconds / (1000*1000)               #4
        result_df.at[index, 'Testing Time (sec)'] = testing_time
    
    print(result_df)
    
    #now make some plots
    result_df.set_index('Model', inplace=True)
    result_df[['Training F1 Score', 'Testing F1 Score']].plot(kind='bar')
    plt.grid()
    plt.ylabel('Score')
    plt.title('Classification Score-{} Data'.format(data))
    plt.tight_layout()
    
    plt.savefig(os.path.join(this_dir,this_dir, os.pardir, "plot", '{}_score.png'.format(data)))


    result_df[['Training Time (sec)', 'Testing Time (sec)']].plot(kind='bar')
    plt.grid()
    plt.ylabel('Time (sec)')
    plt.title('Training and Testing Time-{} Data'.format(data))
    plt.tight_layout()
    plt.savefig(os.path.join(this_dir,os.pardir, "plot", '{}_time.png'.format(data)))
    
    return result_df

@ignore_warnings(category=ConvergenceWarning)
def main(verbose=True, warm_start=False, models=['dt', 'knn', 'svm', 'boost', 'ann']):
    X1, y1 = get_dataset1(verbose)
    X2, y2 = get_dataset2(verbose)
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)
    
    result_data_wine = []
    result_data_pima = []
    if not warm_start:
        
        if 'dt' in models:
            print('''
            #===============================================================================================
            # # Decision Tree
            #===============================================================================================
            ''')
            if warm_start:
                tuned_decision_tree_model_wine = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_decision_tree_model_wine.pkl"), 'rb'))
                tuned_decision_tree_model_dataset2 = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_decision_tree_model_dataset2.pkl"), 'rb'))
            else:
                print("\twine\n----")
                tuned_decision_tree_model_wine = decision_tree_experiments(X_train1, y_train1, data='wine')
                print("\tpima\n----")
                tuned_decision_tree_model_dataset2 = decision_tree_experiments(X_train2, y_train2, data='pima')
                generate_learning_curves(tuned_decision_tree_model_wine,
                                         tuned_decision_tree_model_dataset2,
                                         X_train1, y_train1, X_train2, y_train2,
                                         'Decision Tree')
                pickle.dump(tuned_decision_tree_model_wine, open(os.path.join(this_dir,os.pardir, "models", "tuned_decision_tree_model_wine.pkl"), 'wb'))
                pickle.dump(tuned_decision_tree_model_dataset2, open(os.path.join(this_dir,os.pardir, "models", "tuned_decision_tree_model_dataset2.pkl"), 'wb'))
            result_data_wine.append(['Decision Tree', 'wine', tuned_decision_tree_model_wine])
            result_data_pima.append(['Decision Tree', 'pima', tuned_decision_tree_model_dataset2])
            
        if 'boost' in models:
            print('''
            #===============================================================================================
            # Boosted Tree    
            #===============================================================================================
            ''')
            if warm_start:
                tuned_boosted_tree_model_wine = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_boosted_tree_model_wine.pkl"), 'rb'))
                tuned_boosted_tree_model_dataset2 = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_boosted_tree_model_dataset2.pkl"), 'rb'))
            else:
                print("\twine\n----")
                tuned_boosted_tree_model_wine = boosted_tree_experiments(tuned_decision_tree_model_wine.best_estimator_.get_params()['cfr'],
                                                                         X_train1, y_train1, data='wine')
                print("\tpima\n----")
                tuned_boosted_tree_model_dataset2 = boosted_tree_experiments(tuned_decision_tree_model_dataset2.best_estimator_.get_params()['cfr'], X_train2, y_train2, data='pima')
                generate_learning_curves(tuned_boosted_tree_model_wine,
                                         tuned_boosted_tree_model_dataset2,
                                         X_train1, y_train1, X_train2, y_train2, 'Boosted Tree')
                pickle.dump(tuned_boosted_tree_model_wine, open(os.path.join(this_dir,os.pardir, "models", "tuned_boosted_tree_model_wine.pkl"), 'wb'))
                pickle.dump(tuned_boosted_tree_model_dataset2, open(os.path.join(this_dir,os.pardir, "models", "tuned_boosted_tree_model_dataset2.pkl"), 'wb'))
            result_data_wine.append(['Boosted Tree', 'wine', tuned_boosted_tree_model_wine])
            result_data_pima.append(['Boosted Tree', 'pima', tuned_boosted_tree_model_dataset2])
        
        if 'knn' in models:
            print('''
            #===============================================================================================
            # kNN
            #===============================================================================================
            ''')
            if warm_start:
                tuned_knn_model_wine = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_knn_model_wine.pkl"), 'rb'))
                tuned_knn_model_dataset2 = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_knn_model_dataset2.pkl"), 'rb'))
            else:    
                print("\twine\n----")
                tuned_knn_model_wine = knn_experiments(X_train1, y_train1, data='wine')
                print("\tpima\n----")
                tuned_knn_model_dataset2 = knn_experiments(X_train2, y_train2, data='pima')
                generate_learning_curves(tuned_knn_model_wine,
                                         tuned_knn_model_dataset2,
                                         X_train1, y_train1, X_train2, y_train2, 'kNN')
                pickle.dump(tuned_knn_model_wine, open(os.path.join(this_dir,os.pardir, "models", "tuned_knn_model_wine.pkl"), 'wb'))
                pickle.dump(tuned_knn_model_dataset2, open(os.path.join(this_dir,os.pardir, "models", "tuned_knn_model_dataset2.pkl"), 'wb'))
            result_data_wine.append(['kNN', 'wine', tuned_knn_model_wine])
            result_data_pima.append(['kNN', 'pima', tuned_knn_model_dataset2])
            
        if 'ann' in models:
            print('''
            #===============================================================================================
            # Neural Network
            #===============================================================================================
            ''')
            if warm_start:
                tuned_neural_network_model_wine = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_neural_network_model_wine.pkl"), 'rb'))
                tuned_neural_network_model_dataset2 = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_neural_network_model_dataset2.pkl"), 'rb'))
                
            else:
                
                print("\twine\n----")
                tuned_neural_network_model_wine = neural_network_experiments(X_train1, y_train1, data='wine')
                print("\tpima\n----")
                tuned_neural_network_model_dataset2 = neural_network_experiments(X_train2, y_train2, data='pima')
                generate_learning_curves(tuned_neural_network_model_wine,
                                         tuned_neural_network_model_dataset2,
                                         X_train1, y_train1, X_train2, y_train2, 'Neural Network')
                pickle.dump(tuned_neural_network_model_wine, open(os.path.join(this_dir,os.pardir, "models", "tuned_neural_network_model_wine.pkl"), 'wb'))
                pickle.dump(tuned_neural_network_model_dataset2, open(os.path.join(this_dir,os.pardir, "models", "tuned_neural_network_model_dataset2.pkl"), 'wb'))
            result_data_wine.append(['Neural Network', 'wine', tuned_neural_network_model_wine])
            result_data_pima.append(['Neural Network', 'pima', tuned_neural_network_model_dataset2])
            
        if 'svm' in models:
            print('''
            #===============================================================================================
            # Support Vector Machine
            #===============================================================================================    
            ''')
            if warm_start:
                
                tuned_support_vector_machine_model_wine = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_support_vector_machine_model_wine.pkl"), 'rb'))
                tuned_support_vector_machine_model_dataset2 = pickle.load(open(os.path.join(this_dir,os.pardir, "models", "tuned_support_vector_machine_model_dataset2.pkl"), 'rb'))
            else:
                print("\twine\n----")
                tuned_support_vector_machine_model_wine = support_vector_machine_experiments(X_train1, y_train1, data='wine')
                print("\tpima\n----")
                tuned_support_vector_machine_model_dataset2 = support_vector_machine_experiments(X_train2, y_train2, data='pima')
                generate_learning_curves(tuned_support_vector_machine_model_wine,
                                         tuned_support_vector_machine_model_dataset2,
                                         X_train1, y_train1, X_train2, y_train2, "Support Vector Machine")
                pickle.dump(tuned_support_vector_machine_model_wine, open(os.path.join(this_dir,os.pardir, "models", "tuned_support_vector_machine_model_wine.pkl"), 'wb'))
                pickle.dump(tuned_support_vector_machine_model_dataset2, open(os.path.join(this_dir,os.pardir, "models", "tuned_support_vector_machine_model_dataset2.pkl"), 'wb'))
            result_data_wine.append(['SVM', 'wine', tuned_support_vector_machine_model_wine])
            result_data_pima.append(['SVM', 'pima', tuned_support_vector_machine_model_dataset2])
    print('''
    #===============================================================================================
    # Final model comparison
    #===============================================================================================
    ''') 
    
    
   
    result_df_wine = pd.DataFrame(result_data_wine, columns=['Model', 'Dataset', 'Tuned Model'])
    result_df_pima = pd.DataFrame(result_data_pima, columns=['Model', 'Dataset', 'Tuned Model'])
    test_score(result_df_wine, X_train1, y_train1, X_test1, y_test1, 'Wine Quality')
    test_score(result_df_pima, X_train2, y_train2, X_test2, y_test2, 'Pima Diabetes')
    return result_df_wine, result_df_pima
    

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(False, False, ['ann'])
