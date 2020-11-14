 
from collections import OrderedDict 
import os
import sys

import gym
from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import mdptoolbox, mdptoolbox.example
from numpy.random import choice

import numpy as np
import pandas as pd
import seaborn as sns

# import hiive_mdptoolbox.example
# import hiive_mdptoolbox
np.random.seed(44)

config = {'discount':np.linspace(0.8, 0.99, 10),
          'problems': [
                        
                        {'problem': mdptoolbox.example.forest,
                         'name': 'random',
                        'ranges': {'S':range(2, 5), 'A': [2]},
                        'params': {'S':2}
                        },
                        {'problem': mdptoolbox.example.rand,
                         'name':    'forest',
                         'ranges': {'S':range(2, 5), 'A': range(2, 5)},
                         'params': {'S':2, 'A':2}
                        }
             ]
          }


def run_PI_experiments(problem):
    data = []
    ranges = problem['ranges']
    
    
    for gamma in config['discount']:
        for param in problem['params']:
            for value in problem['ranges'][param]:
                problem['params'][param] = value
                
            #p = problem['problem'](**problem['params'])
    
    None

    
def run_VI_experiments(problem):
    None

    
def run_QL_experiments(problem):
    
    None
    

def run_all_experiments(problem):
    pi_data = run_PI_experiments(problem)
    vi_data = run_VI_experiments(problem)
    ql_data = run_QL_experiments(problem)
    # combine datasets
    data = None
    return data


def run_experiments_both_problems():
    p1_data = run_all_experiments(config['problems'][0])
    #p2_data = run_all_experiments(config['problems'][1])


def main():
    run_experiments_both_problems()

    
for S in range(2, 8):
    P, R = mdptoolbox.example.forest(S=3, r1=4, r2=2, p=0.1, is_sparse=False)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    vi.verbose
    
    vi.run()
    vi.V
    vi.policy
    vi.iter
    vi.time
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.96)
    pi.setVerbose()
    pi.verbose
    
    pi.run()
    pi.V
    pi.policy
    pi.iter
    pi.time
    
    ql = mdptoolbox.mdp.QLearning(P, R, 0.96) 
    ql.run()
    ql.V
    ql.policy
    ql.time
    ql.max_iter
    ql.mean_discrepancy
