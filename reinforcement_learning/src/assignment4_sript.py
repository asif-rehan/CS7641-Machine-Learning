 
from collections import OrderedDict 
import os
import sys
import random
import gym
import hiive.mdptoolbox.mdp as MDP 
import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.util as mdp_util
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
import a4_plot_util
from config import config

# import hiive_mdptoolbox.example
# import hiive_mdptoolbox
random.seed(42)
out = os.path.join(os.path.dirname(__file__), os.path.pardir, 'out')
os.makedirs(out, exist_ok=True)

class Experiments:

    def __init__(self, p, name, params):
        self.p = p
        self.name = name
        self.params = params
        
    #def get_action_size(self):
    #    return self.params[0].get('A', 2)


def create_experiment_configs():
    config['problems']['random'] = Experiments(example.rand, 'random',
                                        [{'S':s, 'A':a} 
                                            for s in config['s_ranges']['random'] 
                                            for a in range(8, 9)])
    config['problems']['forest'] = Experiments(example.forest, 'forest',
                                        [{'S':s} 
                                            for s in config['s_ranges']['forest']])

def run_PI_experiments(problem, name, load=True):
    data = []
    
    if load:
        with open(os.path.join(out, 'PI_results_'+ name +'.pkl'), 'rb') as f:
            df = pickle.load(f)
        return df
    
    print("===========\nPolicy Iteration\n==========")
    for gamma in config['discount']:
        for param in problem.params:
            P, R = problem.p(**param)
            mdp_util.check(P, R)
            for eval_type in ['matrix']:
                pi = MDP.PolicyIteration(P, R, gamma, max_iter=1000, eval_type=eval_type, skip_check=False)
                run_stats = pi.run()
                
                data.append([
                                problem.name, pi.S, 
                                pi.A, pi.gamma, 
                                [i['Time'] for i in run_stats], 
                                pi.iter, pi.max_iter, pi.eval_type, 
                                [i['Mean V'] for i in run_stats], 
                                np.std(pi.V), 
                                [i['Max V'] for i in run_stats], 
                                [i['Reward'] for i in run_stats], 
                                [i['Error'] for i in run_stats],
                                pi.policy]
                            )
                
                print(problem.name, pi.S, pi.A, gamma, eval_type)
    
    df = pd.DataFrame(data, columns=['name', '#states', '#actions', 'discount', 'time', 'iter', 'max_iter',
                                'eval_type', 'mean_V', 'max_V', 'std_V', 'reward', 'error_mean', 'policy'])
    
    with open(os.path.join(out, 'PI_results_'+ name +'.pkl'), 'wb') as f:
        pickle.dump(df, f)
    
    return df

    
def run_VI_experiments(problem, name, load=True):
    data = []
    if load:
        with open(os.path.join(out, 'VI_results_'+ name +'.pkl'), 'rb') as f:
            df = pickle.load(f)
        return df
    
    print("===========\nValue Iteration\n==========")
    for gamma in config['discount']:
        for param in problem.params:
            P, R = problem.p(**param)
            mdp_util.check(P, R)
            vi = MDP.ValueIteration(P, R, gamma, epsilon=0.01, max_iter=1000, skip_check=False)
            run_stats = vi.run()
            data.append([
                                problem.name, vi.S, 
                                vi.A, vi.gamma, 
                                [i['Time'] for i in run_stats], 
                                vi.iter, vi.max_iter,  
                                [i['Mean V'] for i in run_stats], 
                                np.std(vi.V), 
                                [i['Max V'] for i in run_stats], 
                                [i['Reward'] for i in run_stats], 
                                [i['Error'] for i in run_stats],
                                vi.policy])
            print(problem.name, vi.S, vi.A, gamma)
            
    df = pd.DataFrame(data, columns=['name', '#states', '#actions', 'discount', 'time', 'iter', 'max_iter',
                                'mean_V', 'max_V', 'std_V', 'reward', 'error_mean', 'policy'])
    
    with open(os.path.join(out, 'VI_results_'+ name +'.pkl'), 'wb') as f:
        pickle.dump(df, f)
    
    return df
    
def run_QL_experiments(problem, name, load=True):
    data = []
    if load:
        with open(os.path.join(out, 'QL_results_'+ name +'.pkl'), 'rb') as f:
            df = pickle.load(f)
        return df
    
    print("===========\nQ-Learning\n==========")
    for gamma in config['ql_discount']:
        for alpha in config['ql_alpha']:
            for eps in config['ql_epsilon']:
                for param in [config['ql_params'][name]]:
                    P, R = problem.p(**param)
                    mdp_util.check(P, R)
                    for n in config['ql_iters']:
                        ql = MDP.QLearning(P, R, gamma, alpha=alpha, epsilon=eps, n_iter=n)
                        run_stats = ql.run()
                        data.append([
                                        problem.name, ql.S, 
                                        ql.A, ql.gamma, alpha, eps,
                                        [i['Time'] for i in run_stats], 
                                        ql.max_iter,  
                                        [i['Mean V'] for i in run_stats], 
                                        np.std(ql.V), 
                                        [i['Max V'] for i in run_stats], 
                                        [i['Reward'] for i in run_stats], 
                                        [i['Error'] for i in run_stats],
                                        ql.policy
                            ])
                        print(problem.name, ql.S, ql.A, gamma, n, alpha, eps)
    df = pd.DataFrame(data, columns=['name', '#states', '#actions', 'discount', 'alpha', 'epsilon', 'time', 'iter',
                                'mean_V', 'max_V', 'std_V', 'reward', 'error_mean', 'policy'])
    
    with open(os.path.join(out, 'QL_results_'+ name +'.pkl'), 'wb') as f:
        pickle.dump(df, f)
    return df


def run_all_experiments(problem, name, load=True):
    
    pi_data = run_PI_experiments(problem, name, load)    
    vi_data = run_VI_experiments(problem, name, load)
    ql_data = run_QL_experiments(problem, name, load)
    
    a4_plot_util.iter_time_vs_state(pi_data, vi_data, out, name)

    a4_plot_util.iter_time_vs_discount(pi_data, vi_data, out, name)

    a4_plot_util.error_vs_iter(pi_data, vi_data, out, config['s_ranges'][name], name)
    a4_plot_util.discount_vs_reward_v(pi_data, vi_data, config, out, name)
    df = a4_plot_util.check_pi_vi_policies_same(pi_data, vi_data, config, out, name)
    a4_plot_util.ql_rewards(ql_data, out, name)
    a4_plot_util.ql_errors(ql_data, out, name)

    a4_plot_util.print_results(pi_data, vi_data, ql_data, df, config, name)

def run_experiments_both_problems():
    create_experiment_configs()
    for i in config['problems'].keys():
        p = config['problems'][i]
        print(p.name.upper()+'\n==========')
        run_all_experiments(p, p.name, load=False)
        print('\n\n')

def main():
    run_experiments_both_problems()


if __name__=="__main__":
    main()