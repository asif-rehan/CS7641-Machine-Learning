import numpy as np
import pandas as pd
import param_search
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import os

this_dir =  os.path.dirname(__file__)
#===================================================================================================
# STEP1: DEFINE A FITNESS FUNCTION, PROBLEM, AND OUTPUT DIRECTORY
#===================================================================================================
fitness = mlrose.FlipFlop()
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
problem_name = 'flipflop'
output_dir = os.path.join(this_dir, problem_name)   #r"C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks"

#===================================================================================================
# STEP2: RUN OPTIMIZATIONS FOR A RANGE OF HYPERPARAMETERS FOR RHC, SA, GA, AND MIMIC
#===================================================================================================
param_search.run_param_search(problem, output_dir)

#===================================================================================================
# STEP3: READ IN THE DATA REGARDING THE RAN OPTIMIZATION  
#===================================================================================================
rhc_run_curves = pd.read_csv(os.path.join(output_dir, 'RHC', 'rhc__RHC__curves_df.csv'))#r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\RHC\rhc__RHC__curves_df.csv')
rhc_run_stats = pd.read_csv(os.path.join(output_dir, 'RHC', 'rhc__RHC__run_stats_df.csv'))#r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\RHC\rhc__RHC__run_stats_df.csv')

sa_run_curves = pd.read_csv(os.path.join(output_dir, 'SA', 'sa__SA__curves_df.csv'))#r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\SA\sa__SA__curves_df.csv')
sa_run_stats = pd.read_csv(os.path.join(output_dir, 'SA', 'sa__SA__run_stats_df.csv'))#r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\SA\sa__SA__run_stats_df.csv')

ga_run_curves = pd.read_csv(os.path.join(output_dir, 'GA', 'ga__GA__curves_df.csv'))#r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\GA\ga__GA__curves_df.csv')
ga_run_stats = pd.read_csv(os.path.join(output_dir, 'GA', 'ga__GA__run_stats_df.csv'))#r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\GA\ga__GA__run_stats_df.csv')

mimic_run_curves = pd.read_csv(os.path.join(output_dir, 'MIMIC', 'mimic__MIMIC__curves_df.csv'))#pd.read_csv(r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\MIMIC\mimic__MIMIC__curves_df.csv')
mimic_run_stats = pd.read_csv(os.path.join(output_dir, 'MIMIC', 'mimic__MIMIC__run_stats_df.csv'))#r'C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks\MIMIC\mimic__MIMIC__run_stats_df.csv')


param_search.plot_ga_param_search(ga_run_stats, problem_name,  outdir=output_dir, limit_iter1=4096, limit_iter2=8192)
param_search.plot_sa_param_search(sa_run_stats, problem_name, output_dir, limit_iter1=8192*2, limit_iter2=8192*2)
param_search.plot_rhc_param_search(rhc_run_curves, rhc_run_stats, problem_name, output_dir, limit_iter2=4096)
param_search.plot_mimic_param_search(mimic_run_stats, problem_name,  output_dir, limit_iter1=256)

#===================================================================================================
# STEP4: CHOOSE THE BEST PARAMETER CONFIGURATIONS FROM EACH OF GA, SA, RHC, AND MIMIC
#         AND PLOT THEIR PERFORMANCE
#===================================================================================================
ga_best = ga_run_stats[(ga_run_stats['Population Size'] == 100) & (ga_run_stats['Mutation Rate'] == 0.1) & (ga_run_stats['Iteration'] <= 8192)]
sa_best = sa_run_stats[(sa_run_stats['Temperature'] == 500) & (sa_run_stats['schedule_type'] == 'geometric') & (sa_run_stats['Iteration'] <= 8192) ]
rhc_best = rhc_run_stats[(rhc_run_stats['current_restart'] == 7) & (rhc_run_stats['Iteration'] <= 8192) ]
mimic_best = mimic_run_stats[(mimic_run_stats['Keep Percent'] == 0.3) & (mimic_run_stats['Population Size'] == 300) & (mimic_run_stats['Iteration'] <= 8192)]

#===================================================================================================
# STEP5: RUN OPTIMIZATIONS USING THE BEST PARAMETER CONFIGURATIONS FOR EACH OF GA, SA, RHC, AND 
#    MIMIC FOR A RANGE OF PROBLEM LENGTH
#===================================================================================================

lengths_experiment = [2**x for x in range(10)]

for length in lengths_experiment:
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
    output_dir2 = os.path.join(this_dir, problem_name+'_SA_length', str(length)) #r"C:\Users\AREHAN2\Documents\omscs\CS7641\randomized_optimization\4peaks_GA_length\\" + str(length)
    sa = mlrose.SARunner(problem=problem, 
        experiment_name="SA", 
        output_directory=output_dir2, 
        seed=42, 
        iteration_list=2 ** np.arange(14), 
        max_attempts=1000, 
        temperature_list=[500], 
        decay_list=[mlrose.GeomDecay])
    sa_run_stats, sa_run_curves = sa.run()

#===================================================================================================
# STEP6: PLOT PERFORMANCE FROM STEP#4 AND STEP#5 
#===================================================================================================
fig, ax =  plt.subplots(1,3, figsize=(17,4)) 
param_search.plot_best_models(ga_best, sa_best, rhc_best, mimic_best, ax, problem_name)
#plt.show()
 
max_iteartions_needed = []
for length in [2**x for x in range(10)]:
    output_dir2 = os.path.join(this_dir, problem_name+'_SA_length', str(length))
    ga_run_stats = pd.read_csv(os.path.join(output_dir2, 'SA', 'sa__SA__curves_df.csv'))
    max_iteartions_needed.append(ga_run_stats.iloc[ga_run_stats['Fitness'].idxmax()]['Iteration'])
     
ax[2].plot([2**x for x in range(10)], max_iteartions_needed, 'b^--', label='SA best')
 
ax[2].grid()
ax[2].legend()
ax[2].set_xlabel('Problem Length')
ax[2].set_ylabel('Number of Iterations Required')
ax[2].set_title(problem_name+'SA Model Complexity')
plt.savefig(os.path.join(output_dir, problem_name+'final_comparison.png'))
plt.show()