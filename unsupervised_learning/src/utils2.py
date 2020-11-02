import sys
import os
import warnings
import itertools
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


def plot_loss_curves(result_df, folder):
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(111)
    for index, row in result_df.iterrows():
        ax.plot(row['Loss'], label=index)
    
    ax.legend()
    ax.grid()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss function')
    plt.savefig(os.path.join(folder, 'loss_curves.png'))

def test_score_fun(result_df,  data, fldr='step4'):
    
    print(result_df[['Model', 'Dimension',  
                     'Training F1 Score', 'Testing F1 Score', 'Training Time(sec)']])
    
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
    plt.close()
    

    result_df['Training Time(sec)'].plot(kind='bar')
    plt.grid()
    plt.ylabel('Time (sec)')
    plt.title('Training Time-{} Data'.format(data))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, '{}_time.png'.format(data)))
    
    plt.close()
    
    plot_loss_curves(result_df, folder)
    
    
    return result_df
