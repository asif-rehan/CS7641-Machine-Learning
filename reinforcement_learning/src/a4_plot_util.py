import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from config import config
import itertools

def iter_time_vs_state(pi_data, vi_data, out, name):
    fig, ax1 = plt.subplots()
    

    pi_slice = pi_data[(pi_data['eval_type']=='matrix')
                & (pi_data['discount']==0.99)]

    vi_slice = vi_data[vi_data['discount']==0.99]

    #ql_slice = ql_data[(ql_data['#actions']==10)  
    #             & (ql_data['discount']==0.99(pi_data['#actions']==a) & ) & (ql_data['iter']==10000)]

    ax1.plot(pi_slice['#states'], pi_slice['iter'], 'ro-', label='PI Iterations')
    ax1.plot(vi_slice['#states'], vi_slice['iter'], 'rs-', label='VI Iterations')
    #ax1.plot(ql_slice['#states'], ql_slice['iter'], 'c^-', label='QL Iterations')

    ax1.grid()
    ax1.legend(loc='center left')
    ax1.set_ylabel('Iterations')
    ax1.set_xlabel('#states')

    ax2 = ax1.twinx()
    ax2.plot(pi_slice['#states'], [pi_slice.loc[t]['time'][-1] for t in pi_slice.index], 'b^--', label='PI Time')

    ax2.plot(vi_slice['#states'], [vi_slice.loc[t]['time'][-1] for t in vi_slice.index], 'b+--', label='VI Time')
    #ax2.plot(ql_slice['#states'], [ql_slice.loc[t]['time'][-1] for t in ql_slice.index], 'c^--', label='QL Time')

    ax2.set_ylabel('time')
    ax2.legend(loc='center right')
    plt.title('Iterations and Time vs #States-'+name.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'iter_time_vs_state_'+name+'.png'))
    plt.close()
    
def discount_vs_reward_v(pi_data, vi_data, config, out, name):
    s = config['ql_params'][name]['S']

    fig, ax1 = plt.subplots()
    colormap = plt.cm.gist_ncar

    num_plots = len(config['s_ranges'][name])*2
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

    for s in config['s_ranges'][name]:
        pi_slice = pi_data[(pi_data['eval_type']=='matrix')
                    & (pi_data['#states']==s)]

        vi_slice = vi_data[vi_data['#states']==s]

        #ql_slice = ql_data[(ql_data['#actions']==10)  
        #             & (ql_data['discount']==0.99(pi_data['#actions']==a) & ) & (ql_data['iter']==10000)]
        ax1.plot(pi_slice['discount'], [i[-1] for i in pi_slice['reward']], linestyle='-', marker='o', label='PI state='+str(s))
        ax1.plot(vi_slice['discount'], [i[-1] for i in vi_slice['reward']], linestyle='--', marker='s',  label='VI state='+str(s))
        #ax1.plot(ql_slice['#states'], ql_slice['iter'], 'c^-', label='QL Iterations')

    ax1.grid()
    ax1.legend(loc='center left')
    ax1.set_ylabel('Reward')
    ax1.set_xlabel('Discount')

    #ax2 = ax1.twinx()
    #ax2.plot(pi_slice['discount'], [pi_slice.loc[t]['time'][-1] for t in pi_slice.index], 'b^--', label='PI Time')

    #ax2.plot(vi_slice['discount'], [vi_slice.loc[t]['time'][-1] for t in vi_slice.index], 'b+--', label='VI Time')
    #ax2.plot(ql_slice['#states'], [ql_slice.loc[t]['time'][-1] for t in ql_slice.index], 'c^--', label='QL Time')

    #ax2.set_ylabel('time')
    #ax2.legend(loc='center right')
    plt.title('Reward vs Discounts-'+name.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'reward_vs_discount_'+name+'.png'))
    plt.close()


def check_pi_vi_policies_same(pi_data, vi_data, config, out, name):
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    diff_pi_vi = []
    for s in config['s_ranges'][name]:
        pi = pi_data[(pi_data['discount']==0.99) & (pi_data['#states']==s)]
        vi = vi_data[(vi_data['discount']==0.99) & (vi_data['#states']==s)]
        
        pi_policy, vi_policy = pi['policy'], vi['policy']
        
        #cos = cosine_similarity([pi_policy.iloc[i]], [vi_policy.iloc[i]])
        diff = np.array(pi_policy.iloc[0])-np.array(vi_policy.iloc[0])
        match = (diff==0).sum()/diff.shape[0]*100
        diff_pi_vi.append([s, match])
    df = pd.DataFrame(diff_pi_vi, columns=['#states', 'match'])
    #df.set_index('#states')['cosine'].plot(kind='bar', ax=ax1)
    df.set_index('#states')['match'].plot(kind='bar', ax=ax1)
    ax1.set_xlabel('#states')
    ax1.set_ylabel('%match')
    ax1.set_title('Policy Comparison-'+name.upper())
    plt.grid()
    plt.tight_layout()
    fig.savefig(os.path.join(out, '%match_pi_vi-'+name+'.png'))
    plt.close()
    return df


def iter_time_vs_discount(pi_data, vi_data, out, name):

    s = config['ql_params'][name]['S']
    
    fig, ax1 = plt.subplots()

    pi_slice = pi_data[ (pi_data['eval_type']=='matrix')
                & (pi_data['#states']==s)]

    vi_slice = vi_data[vi_data['#states']==s]

    #ql_slice = ql_data[(ql_data['#actions']==10)  
    #             & (ql_data['discount']==0.99(pi_data['#actions']==a) & ) & (ql_data['iter']==10000)]

    ax1.plot(pi_slice['discount'], pi_slice['iter'], 'ro-', label='PI Iterations')
    ax1.plot(vi_slice['discount'], vi_slice['iter'], 'rs-', label='VI Iterations')
    #ax1.plot(ql_slice['#states'], ql_slice['iter'], 'c^-', label='QL Iterations')

    ax1.grid()
    ax1.legend(loc='center left')
    ax1.set_ylabel('Iterations')
    ax1.set_xlabel('Discount')

    ax2 = ax1.twinx()
    ax2.plot(pi_slice['discount'], [pi_slice.loc[t]['time'][-1] for t in pi_slice.index], 'b^--', label='PI Time')

    ax2.plot(vi_slice['discount'], [vi_slice.loc[t]['time'][-1] for t in vi_slice.index], 'b+--', label='VI Time')
    #ax2.plot(ql_slice['#states'], [ql_slice.loc[t]['time'][-1] for t in ql_slice.index], 'c^--', label='QL Time')

    ax2.set_ylabel('time')
    ax2.legend(loc='center right')
    plt.title('Iterations and Time vs Discounts-'+name.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'iter_time_vs_discount_'+name+'.png'))
    plt.close()

def ql_rewards(ql_data, out, name):
    fig, ax = plt.subplots(figsize=(10,5))
    s = config['ql_params'][name]['S']
    colormap = plt.cm.gist_ncar
    
    markers = itertools.cycle((',', '+', 'x', 'o', '*', 's', 'd')) 
    num_plots = len(config['ql_alpha'])*len(config['ql_epsilon'])
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots)))) 
    
    for eps, alpha in list(ql_data[['epsilon', 'alpha']].drop_duplicates().to_records(index=False)):
            #print(alpha, eps)
            x = ql_data[(ql_data['epsilon']==eps) & (ql_data['alpha']==alpha)][['error_mean', 'mean_V', 'reward', 'policy']]
            
            #plt.plot(x['error_mean'].values[0], label='Error alpha='+str(alpha)+' eps='+str(eps), alpha=0.5)
            plt.plot(x['mean_V'].values[0], marker=next(markers), markersize='5', markevery=100, label='alpha='+str(round(alpha,3))+' eps='+str(round(eps,3)))
            #plt.plot(x['reward'].values[0], label='Reward V alpha='+str(alpha)+' eps='+str(eps), alpha=0.25)
    #plt.xscale('log')
    plt.grid(which='both')
    plt.xlabel('iterations/'+str(config['ql_iters'][0]//10000))
    plt.ylabel('mean_V')
    plt.legend(prop={'size': 7})
    plt.title('ql_param_search-'+name.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'ql_param_search-'+name.upper()+'.png'))
    plt.close()

def ql_errors(ql_data, out, name):
    fig, ax = plt.subplots(figsize=(10,5))
    s = config['ql_params'][name]['S']
    colormap = plt.cm.gist_ncar
    
    markers = itertools.cycle((',', '+', 'x', 'o', '*', 's', 'd')) 
    num_plots = len(config['ql_alpha'])*len(config['ql_epsilon'])+2
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 3, num_plots)))) 
    
    for eps, alpha in list(ql_data[['epsilon', 'alpha']].drop_duplicates().to_records(index=False)):
            #print(alpha, eps)
            x = ql_data[(ql_data['epsilon']==eps) & (ql_data['alpha']==alpha)][['error_mean', 'mean_V', 'reward', 'policy']]
            

            #plt.plot(x['error_mean'].values[0], label='Error alpha='+str(alpha)+' eps='+str(eps), alpha=0.5)
            plt.plot(x['error_mean'].values[0], marker=next(markers), markersize='5', markevery=100, alpha=0.5, label='alpha='+str(round(alpha,3))+' eps='+str(round(eps,3)))
            #plt.plot(x['reward'].values[0], label='Reward V alpha='+str(alpha)+' eps='+str(eps), alpha=0.25)
    #plt.xscale('log')
    plt.grid(which='both')
    plt.xlabel('iterations/'+str(config['ql_iters'][0]//10000))
    plt.ylabel('error')
    plt.legend(prop={'size': 7})
    plt.title('QL Error-'+name.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'ql_error-'+name.upper()+'.png'))
    plt.close()


def error_vs_iter(pi_data, vi_data, out, s_range, name): 
    colormap = plt.cm.gist_ncar
    
    num_plots = len(s_range)*2
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots)))) 
    for s in s_range:

        x = pi_data[(pi_data['#states']==s) & (pi_data['discount']==0.99)  & (pi_data['eval_type']=='matrix')]['error_mean']
        plt.plot(x.iloc[0], linestyle='-', label='PI #states='+str(s), alpha=0.7)
        x = vi_data[(vi_data['#states']==s) & (vi_data['discount']==0.99) ]['error_mean']
        plt.plot(x.iloc[0], linestyle='--', label='VI #states='+str(s), alpha=0.7)
    plt.legend()
    plt.grid()
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.title('Error vs Iterations-'+name.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'error_iter_'+name+'.png'))
    plt.close()
    


def print_results(pi_data, vi_data, ql_data, df, config, name):
    s = config['ql_params'][name]['S']
    pi = pi_data[(pi_data['discount']==0.99) & (pi_data['#states']==s)]
    alpha = config['ql_hyper_params'][name]['alpha']
    eps = config['ql_hyper_params'][name]['epsilon']
    a = pi.iloc[0]['#actions']
    vi = vi_data[(vi_data['discount']==0.99) & (vi_data['#states']==s)]
    ql = ql_data[(ql_data['discount']==0.99) & (ql_data['#states']==s) & (ql_data['alpha']==alpha) & (ql_data['epsilon']==eps)]
    pi_policy, vi_policy, ql_policy = pi['policy'], vi['policy'], ql['policy']

    #cos = cosine_similarity([pi_policy.iloc[i]], [vi_policy.iloc[i]])
    diff = np.array(pi_policy.iloc[0])-np.array(ql_policy.iloc[0])
    match_pq = (diff==0).sum() / diff.shape[0] * 100

    diff = np.array(vi_policy.iloc[0])-np.array(ql_policy.iloc[0])
    match_qv = (diff==0).sum() / diff.shape[0] * 100

    df = df.rename(columns={'match':'match_PI_VI'})
    match_pv = df[df['#states']==s]['match_PI_VI'].values[0]
    result = [['x', match_pv, match_pq], [match_pv, 'x', match_qv], [match_pq, match_qv, 'x']]
    result = pd.DataFrame(result, columns=['PI', 'VI', 'QL'], index=['PI', 'VI', 'QL'])

    result2 = [
        [pi['time'].values[0][-1], 
    pi.iloc[0]['iter'], 
    pi.iloc[0]['mean_V'][-1],
    pi.iloc[0]['reward'][-1], 
    'PI'],

    [vi['time'].values[0][-1], 
    vi.iloc[0]['iter'], 
    vi.iloc[0]['mean_V'][-1], 
    vi.iloc[0]['reward'][-1],
    'VI'], 

    [ql['time'].values[0][-1], 
    ql.iloc[0]['iter'], 
    ql.iloc[0]['mean_V'][-1],
    ql.iloc[0]['reward'][-1],
    'QL']  
    ]
    df2 = pd.DataFrame(result2, columns=['Time', 'Iters', 'Mean V', 'Total Reward', 'Algorithm'])
    df2 = df2.set_index('Algorithm').round(3)
    print(df2)
    print("\n\n%Match of Policies as a Matrix\n------------------------\n")
    print(result)
