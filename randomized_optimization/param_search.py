import warnings
warnings.simplefilter("ignore")

import numpy as np
import mlrose_hiive as mlrose
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os



def plot_ga_param_search(runs_ga, problem_name, outdir, limit_iter1=3000, limit_iter2=5000):
    
    pop = [100, 200, 300]
    mut = [0.1, 0.2, 0.3]
    
    combinations = [(p, m) for p in pop for m in mut]
    color = {100: 'r', 200:'b', 300:'g'}
    marker = {0.1: 'o', 0.2:'+', 0.3:'x'}
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    #fig.figure(figsize=(10,10))
    for p,m in combinations:
        runs = runs_ga[(runs_ga['Population Size'] == p) & (runs_ga['Mutation Rate'] == m) & (runs_ga['Iteration'] <= limit_iter1)]
        
        ax[0].plot(runs['Iteration'], runs['Fitness'], c=color[p], marker=marker[m], ls='--',label='Population Size='+str(p)+' Mutation Rate='+str(m))
        #ax[0].legend()
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Fitness')
        ax[0].set_title('GA-'+problem_name+'-Fitness')
        ax[0].grid()
        ax[0].legend()
    
    for p,m in combinations:
        runs = runs_ga[(runs_ga['Population Size'] == p) & (runs_ga['Mutation Rate'] == m)  & (runs_ga['Iteration'] <= limit_iter2)]
        
        ax[1].plot(runs['Iteration'], runs['Time'],  c=color[p], marker=marker[m], ls='--', label='Population Size='+str(p)+' Mutation Rate='+str(m))
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Time')
        ax[1].set_title('GA-'+problem_name+'-Time')
        ax[1].grid()
        #ax[1].legend()
    #plt.show()
    plt.savefig(os.path.join(outdir, problem_name+'_GA_param_search.png'))
    plt.close()

def plot_sa_param_search(runs, problem_name, outdir, limit_iter1=10000, limit_iter2=5000):

    temp = [100, 250, 500]
    decay = ["exponential", "geometric"]
    
    combinations = [(t, d) for t in temp for d in decay]
    color = {100: 'r', 250:'b', 500:'g'}
    marker = {"exponential": 'o', "geometric":'+'}
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    #fig.figure(figsize=(10,10))
    for t,d in combinations:
        run = runs[(runs['Temperature'] == t) & (runs['schedule_type'] == d) & (runs['Iteration'] <= limit_iter1)]
        #print(run[['Temperature', 'schedule_type']].head())
        ax[0].plot(run['Iteration'], run['Fitness'], c=color[t], marker=marker[d], ls='--', 
                   label='Temperature='+str(t)+' Decay='+str(d))
        #ax[0].legend()
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Fitness')
        ax[0].set_title('SA-'+problem_name+'-Fitness')
        ax[0].grid()
        ax[0].legend()
    
    for t,d in combinations:
        run = runs[(runs['Temperature'] == t) & (runs['schedule_type'] == d)  & (runs['Iteration'] <= limit_iter2)]
    
        ax[1].plot(run['Iteration'], run['Time'],  c=color[t], marker=marker[d], ls='--', 
                   label='Temperature='+str(t)+' Decay='+str(d))
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Time')
        ax[1].set_title('SA-'+problem_name+'-Time')
        ax[1].grid()
        #ax[1].legend()
    ##plt.show()
    plt.savefig(os.path.join(outdir, problem_name+'_SA_param_search.png'))
    plt.close()

def plot_rhc_param_search(rhc_curves_runs, rhc_runs_stats, problem_name, outdir, limit_iter1=np.inf, limit_iter2=np.inf):
    restarts = np.arange(1,11)
    
    #color = {0: 'r', 10:'b', 50:'g'}
    markers = {0:'o', 1:'^', 2:'s', 3:'+', 4:'*', 5:'x', 6:'p', 7:'h', 8:'<', 9:'>', 10:''}
    #markers = {i:'$'+str(i)+'$' for i in restarts}
    #print(markers)
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    for t in restarts:
        run = rhc_curves_runs[(rhc_curves_runs['current_restart'] == t) & (rhc_curves_runs['Iteration'] <=limit_iter1)]
        run = run.iloc[::300, :]
        ax[0].plot(run['Iteration']-run.iloc[0]['Iteration'], run['Fitness'], ls='--', marker=markers[t],
                   label='Restart='+str(t))
        #ax[0].plot(run['Iteration'], run['Fitness'], ls='--',  marker=markers[t],
        #           label='Restart='+str(t))
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Fitness')
        ax[0].set_title('RHC-'+problem_name+'-Fitness')
        ax[0].grid()
        ax[0].legend()
        
    for t in restarts:
        run = rhc_runs_stats[(rhc_runs_stats['current_restart'] == t) & (rhc_runs_stats['Iteration'] <= limit_iter2)]
    
        #ax[1].plot(run['Iteration']-run.iloc[0]['Iteration'], run['Time']-run.iloc[0]['Time'],  ls='--', 
        #           label='Restart='+str(t))
        ax[1].plot(run['Iteration'], run['Time']-run.iloc[0]['Time'],  ls='--',  marker=markers[t],
               label='Restart='+str(t))
    
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Time')
        ax[1].set_title('RHC-'+problem_name+'-Time')
        ax[1].grid()
        ax[1].legend()
    #plt.show()
    plt.savefig(os.path.join(outdir, problem_name+'_RHC_param_search.png'))
    plt.close()

def plot_mimic_param_search(runs, problem_name, outdir, limit_iter1=1000, limit_iter2=2000):
    keep_pct = [0.1, 0.2, 0.3]
    population_sizes=[100, 200, 300]
    combinations = [(t, p) for t in keep_pct for p in population_sizes]
    
    color = {0.1: 'r', 0.2:'b', 0.3:'g'}
    marker = {100: '^', 200:'x', 300:'+'}
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    #fig.figure(figsize=(10,10))
    for t, p in combinations:
        run = runs[(runs['Keep Percent'] == t) & (runs['Population Size'] == p) &  (runs['Iteration'] <= limit_iter1)]
        ax[0].plot(run['Iteration'], run['Fitness'], 
                   c=color[t], marker=marker[p],
                   ls='--', 
                   label='Keep Pct='+str(t)+' Population Size='+str(p))
        #ax[0].legend()
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Fitness')
        ax[0].set_title('MIMIC-'+problem_name+'-Fitness')
        ax[0].grid()
        ax[0].legend()
    
    for t, p in combinations:
        run = runs[(runs['Keep Percent'] == t) & (runs['Population Size'] == p) &  (runs['Iteration'] <= limit_iter2)]
    
        ax[1].plot(run['Iteration'], run['Time'],  
                   c=color[t], marker=marker[p],
                   ls='--', 
                   label='Keep Pct='+str(t)+' Population Size='+str(p))
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Time')
        ax[1].set_title('MIMIC-'+problem_name+'-Time')
        ax[1].grid()
        ax[1].legend()
    #plt.show()
    plt.savefig(os.path.join(outdir, problem_name+'_MIMIC_param_search.png'))
    plt.close()

def plot_best_models(ga_best, sa_best, rhc_best, mimic_best, ax, problem_name):
    ax[0].plot(ga_best['Iteration'], ga_best['Fitness'], 'ro--', label='GA best')
    ax[0].plot(sa_best['Iteration'], sa_best['Fitness'], 'b^--', label='SA best')
    ax[0].plot(rhc_best['Iteration'], rhc_best['Fitness'], 'yd--', label='RHC best')
    ax[0].plot(mimic_best['Iteration'], mimic_best['Fitness'], 'cs--', label='MIMIC best')
    ax[0].legend()
    ax[0].grid() #ax[0].set_xscale('log')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Fitesss')
    ax[0].set_title(problem_name+' Fitness Comparison')
    
    ax[1].plot(ga_best['Iteration'], ga_best['Time'], 'ro--', label='GA best')
    ax[1].plot(sa_best['Iteration'], sa_best['Time'], 'b^--', label='SA best')
    ax[1].plot(rhc_best['Iteration'], rhc_best['Time'] - rhc_best.iloc[0]['Time'], 'yd--', label='RHC best')
    ax[1].plot(mimic_best['Iteration'], mimic_best['Time'], 'cs--', label='MIMIC best')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Time')
    ax[1].set_title(problem_name+' Time Comparison')
    

def run_param_search(problem, output_dir):
    rhc = mlrose.RHCRunner(problem=problem, experiment_name="RHC", 
        output_directory=output_dir, 
        seed=42, 
        iteration_list=2 ** np.arange(15), 
        max_attempts=1000, 
        restart_list=[10])
    rhc_run_stats, rhc_run_curves = rhc.run()
    sa = mlrose.SARunner(problem=problem, 
        experiment_name="SA", 
        output_directory=output_dir, 
        seed=42, 
        iteration_list=2 ** np.arange(20), 
        max_attempts=1000, 
        temperature_list=[100, 250, 500], 
        decay_list=[mlrose.ExpDecay, mlrose.GeomDecay])
    sa_run_stats, sa_run_curves = sa.run()
    ga = mlrose.GARunner(problem=problem, 
        experiment_name="GA", 
        output_directory=output_dir, 
        seed=42, 
        iteration_list=2 ** np.arange(13), 
        max_attempts=1000, 
        population_sizes=[100, 200, 300], 
        mutation_rates=[0.1, 0.2, 0.3])
    ga_run_stats, ga_run_curves = ga.run()
    mimic = mlrose.MIMICRunner(problem=problem, 
        experiment_name="MIMIC", 
        output_directory=output_dir, 
        seed=42, 
        iteration_list=2 ** np.arange(13), 
        population_sizes=[100, 200, 300], 
        max_attempts=500, 
        keep_percent_list=[0.1, 0.2, 0.3], 
        use_fast_mimic=True)
    mimic_run_stats, mimic_run_curves = mimic.run()
