import sys
import os
import warnings
import traceback
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
warnings.filterwarnings('ignore')
    
from warnings import simplefilter
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import time
import datetime
import pickle

this_dir =  os.path.dirname(__file__)

def test_score(result_df,  data, fldr='step4'):
    result_df['Training Time (sec)'] = ""
    result_df['Training F1 Score'] = ""
    result_df['Testing Time (sec)'] = ""
    result_df['Testing F1 Score'] = ""
    
    
    
    
    for index, row in result_df.iterrows():
        X2_train_nn_reduced = row['Reduced X_train']
        y_train = row['y_train']
        X_test = row['Reduced X_test']
        y_test = row['y_test']
        tuned_model = row['Tuned Model']
        start = datetime.datetime.now()
        tuned_model.fit(X2_train_nn_reduced, y_train)
        end = datetime.datetime.now()
        training_time = (end - start).microseconds / (1000*1000)          #1
        result_df.at[index, 'Training Time (sec)'] = training_time
        training_score = tuned_model.score(X2_train_nn_reduced, y_train)
        #print(training_score)
        result_df.at[index, 'Training F1 Score'] = training_score        #2 
        
        start = datetime.datetime.now()
        try:
            test_score = tuned_model.score(X_test, y_test)
        except:
            traceback.print_exc()
        #print(test_score)
        result_df.at[index, 'Testing F1 Score'] = test_score            #3
        end = datetime.datetime.now()
        testing_time = (end - start).microseconds / (1000*1000)               #4
        result_df.at[index, 'Testing Time (sec)'] = testing_time
    
    print(result_df[['Model', 'Training Time (sec)', 'Testing Time (sec)', 
                     'Training F1 Score', 'Testing F1 Score']])
    
    #now make some plots
    result_df.set_index('Model', inplace=True)
    result_df[['Training F1 Score', 'Testing F1 Score']].plot(kind='bar')
    plt.grid()
    plt.ylabel('Score')
    plt.title('Classification Score-{} Data'.format(data))
    plt.tight_layout()
    
    folder = os.path.join(this_dir, os.pardir, "plot", fldr)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, '{}_score.png'.format(data)))


    result_df[['Training Time (sec)', 'Testing Time (sec)']].plot(kind='bar')
    plt.grid()
    plt.ylabel('Time (sec)')
    plt.title('Training and Testing Time-{} Data'.format(data))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, '{}_time.png'.format(data)))
    
    return result_df
