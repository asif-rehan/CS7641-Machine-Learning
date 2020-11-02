#%%
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection._split import train_test_split
import traceback
from unsupervised_learning.src import plot_clusters
import copy
warnings.filterwarnings("ignore")
import os
os.chdir(os.path.dirname(__file__))

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure
from sklearn.random_projection import SparseRandomProjection
from get_data import *
from utils import *
from utils2 import *
from plot_clusters import *


#%%

def get_best_clusters(x1, y1, x2, y2):
    clusters = {}
    for d, (x, y) in {'wine':(x1, y1), 'pima':(x2, y2)}.items():
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(x)
        kmeans_pred = kmeans.predict(x)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(x)
        gmm_pred = gmm.predict(x)
        clusters[d] = {'kmeans':{'obj':kmeans, 'clusters':kmeans_pred}, 
            'gmm':{'obj':gmm, 'clusters':gmm_pred}}
    
    return clusters

def run_cluster(x1, y1, x2, y2, plot=True, title=""):
    data = []
    cluster_algo = 'KMeans'
    for d, (x,y) in {'wine': (x1,y1), 'pima': (x2,y2)}.items():
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            t1 = datetime.datetime.now()
            kmeans.fit(x)
            tdelta = datetime.datetime.now() - t1
            pred_cluster = kmeans.predict(x)
            #print(pred_cluster)
            h, c, v = homogeneity_completeness_v_measure(y, pred_cluster)
            data.append([cluster_algo, kmeans, d, k, kmeans.score(x), kmeans.inertia_, 
                         silhouette_score(x, pred_cluster), kmeans.n_iter_, 
                         tdelta.total_seconds(), h, c, v])
    
    df_kmeans = pd.DataFrame(data, columns=["Algo", "kmeans", "Data", "k", "score", "Inertia", 
                                            "silhouette score", "#Iterations", "Time", "Homogeniety Score", "Completeness Score", "V-Measure"])
    df_kmeans=df_kmeans.set_index('k')
    
    df_kmeans[df_kmeans['Data']=='wine']
    
    
    data = []
    cluster_algo = 'GMM'
    for d, (x,y) in {'wine': (x1,y1), 'pima': (x2,y2)}.items():
        for k in range(2, 11):
            gmm = GaussianMixture(n_components=k, random_state=42)
            t1 = datetime.datetime.now()
            gmm.fit(x)
            tdelta = datetime.datetime.now() - t1
            pred_cluster = gmm.predict(x)
            #print(pred_cluster)
            h, c, v = homogeneity_completeness_v_measure(y, pred_cluster)
            data.append([cluster_algo, gmm, d, k, gmm.score(x), gmm.lower_bound_, gmm.aic(x), gmm.bic(x), 
                         silhouette_score(x, gmm.predict(x)), gmm.n_iter_, tdelta.total_seconds(),
                          h, c, v])
    
    df_gmm = pd.DataFrame(data, columns=["Algo", "gmm", "Data", "k", "score", "Lower Bound", "AIC", 
                                         "BIC", "silhouette score", "#Iterations", "Time", 
                                        "Homogeniety Score", "Completeness Score", "V-Measure"])
    df_gmm = df_gmm.set_index('k')

        
    #print(df_kmeans)
    if plot: 
        cluster_chart(df_kmeans, df_gmm, title)
        
        
        plt.close('all')
        save_loc = 'step1' if title=="" else 'step3'
        
        for index, row in df_kmeans.iterrows():
            ttl = 'KMeans-' + row['Data']+'-'+title
            x = x1 if row['Data']=='wine' else x2
            plot_clusters.plot_kmeans(row['kmeans'], x, index, rseed=42, save_loc=save_loc, title=ttl)
        
        for index, row in df_gmm.iterrows():
            ttl = 'GMM-'+row['Data']+'-'+title
            x = x1 if row['Data']=='wine' else x2
            plot_clusters.plot_gmm(row['gmm'], x, index, save_loc=save_loc, title=ttl)
    return df_kmeans, df_gmm


def cluster_chart(df_kmeans, df_gmm, title=""):
    
    #%%
    fig_row_col = 420
    fig = plt.figure(figsize=(12,12))
    
    ax1 = plt.subplot(fig_row_col+1)
    ax1_rev = ax1.twinx()
    ax1.set_title('KMeans Clustering')
    
    color="tab:blue"
    df_kmeans[df_kmeans['Data']=='wine']['score'].plot(linestyle='dashed', marker='o', ax=ax1, color=color, label='Score-Wine')
    df_kmeans[df_kmeans['Data']=='pima']['score'].plot(linestyle='solid', marker='o', ax=ax1, color=color, label='Score-Pima')
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_ylabel('Score', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()
    ax1.set_xlabel('k')
    ax1.legend()
    
    color="tab:red"
    df_kmeans[df_kmeans['Data']=='wine']['Inertia'].plot(linestyle='dashed', marker='o', ax=ax1_rev, color=color, label='Inertia-wine')
    df_kmeans[df_kmeans['Data']=='pima']['Inertia'].plot(linestyle='solid', marker='o', ax=ax1_rev, color=color, label='Inertia-Pima')
    ax1_rev.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xlabel('k')
    ax1_rev.set_ylabel('Inertia', color=color)  # we already handled the x-label with ax1
    ax1_rev.tick_params(axis='y', labelcolor=color)
    ax1_rev.grid()
    ax1_rev.legend()
    
    
    ax322 = plt.subplot(fig_row_col+2)
    ax322_rev = ax322.twinx()
    ax322.set_title('Expectation Maximization(GMM) Clustering')
    
    color="tab:blue"
    df_gmm[df_gmm['Data']=='wine']['score'].plot(linestyle='dashed', marker='o', ax=ax322, color=color, label='Score-Wine')
    df_gmm[df_kmeans['Data']=='pima']['score'].plot(linestyle='solid', marker='o', ax=ax322, color=color, label='Score-Pima')
    ax322.ticklabel_format(useOffset=False, style='plain')
    ax322.set_ylabel('Score', color=color)  # we already handled the x-label with ax1
    ax322.tick_params(axis='y', labelcolor=color)
    ax322.grid()
    ax322.set_xlabel('k')
    ax322.legend()
    
    
    
    
    ax2 = plt.subplot(fig_row_col+3)
    ax2_rev = ax2.twinx()
    
    color="tab:blue"
    df_kmeans[df_kmeans['Data']=='wine']['#Iterations'].plot(linestyle='dashed', marker='o', ax=ax2, color=color, label='#Iterations-Wine')
    df_kmeans[df_kmeans['Data']=='pima']['#Iterations'].plot(linestyle='solid', marker='o', ax=ax2, color=color, label='#Iterations-Pima')
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax2.set_xlabel('k')
    ax2.set_ylabel('#Iterations', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid()
    ax2.set_xlabel('k')
    ax2.legend(loc='center right')
    
    
    color="tab:red"
    df_kmeans[df_kmeans['Data']=='wine']['Time'].plot(linestyle='dashed', marker='o', ax=ax2_rev, color=color, label='Time-Wine')
    df_kmeans[df_kmeans['Data']=='pima']['Time'].plot(linestyle='solid', marker='o', ax=ax2_rev, color=color, label='Time-Wine')
    ax2_rev.ticklabel_format(useOffset=False, style='plain')
    ax2_rev.set_xlabel('k')
    ax2_rev.set_ylabel('Time', color=color)  # we already handled the x-label with ax1
    ax2_rev.tick_params(axis='y', labelcolor=color)
    ax2_rev.grid()
    ax2_rev.legend()
    
    
    
    ax324 = plt.subplot(fig_row_col+4)
    ax324_rev = ax324.twinx()
    
    color="tab:blue"
    df_gmm[df_gmm['Data']=='wine']['#Iterations'].plot(linestyle='dashed', marker='o', ax=ax324, color=color, label='#Iterations-Wine')
    df_gmm[df_gmm['Data']=='pima']['#Iterations'].plot(linestyle='solid', marker='o', ax=ax324, color=color, label='#Iterations-Pima')
    ax324.ticklabel_format(useOffset=False, style='plain')
    ax324.set_xlabel('k')
    ax324.set_ylabel('#Iterations', color=color)  # we already handled the x-label with ax1
    ax324.tick_params(axis='y', labelcolor=color)
    ax324.grid()
    ax324.set_xlabel('k')
    ax324.legend()
    
    
    color="tab:red"
    df_gmm[df_gmm['Data']=='wine']['Time'].plot(linestyle='dashed', marker='o', ax=ax324_rev, color=color, label='Time-Wine')
    df_gmm[df_gmm['Data']=='pima']['Time'].plot(linestyle='solid', marker='o', ax=ax324_rev, color=color, label='Time-Wine')
    ax324_rev.ticklabel_format(useOffset=False, style='plain')
    ax324_rev.set_xlabel('k')
    ax324_rev.set_ylabel('Time', color=color)  # we already handled the x-label with ax1
    ax324_rev.tick_params(axis='y', labelcolor=color)
    ax324_rev.grid()
    ax324_rev.legend()
    
    
    ax3 = plt.subplot(fig_row_col+5)
    color="tab:green"
    df_kmeans[df_kmeans['Data']=='wine']['silhouette score'].plot(linestyle='dashed', marker='s', ax=ax3, color=color, label='Silhouette-Wine')
    df_kmeans[df_kmeans['Data']=='pima']['silhouette score'].plot(linestyle='solid', marker='s', ax=ax3, color=color, label='Silhouette-Pima')
    ax3.ticklabel_format(useOffset=False, style='plain')
    ax3.set_ylabel('Silhouette score', color=color)  # we already handled the x-label with ax1
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid()
    ax3.set_xlabel('k')
    ax3.legend()
    
    
    ax326 = plt.subplot(fig_row_col+6)
    ax326_rev = ax326.twinx()
    color="tab:green"
    df_gmm[df_gmm['Data']=='wine']['silhouette score'].plot(linestyle='dashed', marker='s', ax=ax326, color=color, label='Silhouette-Wine')
    df_gmm[df_gmm['Data']=='pima']['silhouette score'].plot(linestyle='solid', marker='s', ax=ax326, color=color, label='Silhouette-Pima')
    ax326.ticklabel_format(useOffset=False, style='plain')
    ax326.set_xlabel('k')
    ax326.set_ylabel('Silhouette score', color=color)  # we already handled the x-label with ax1
    ax326.tick_params(axis='y', labelcolor=color)
    ax326.grid()
    ax326.legend()
    
    color="tab:red"
    df_gmm[df_gmm['Data']=='wine']['AIC'].plot(linestyle='dashed', marker='s', ax=ax326_rev, color=color, label='AIC-Wine')
    df_gmm[df_gmm['Data']=='pima']['AIC'].plot(linestyle='solid', marker='s', ax=ax326_rev, color=color, label='AIC-Pima')
    df_gmm[df_gmm['Data']=='wine']['BIC'].plot(linestyle='dashed', marker='^', ax=ax326_rev, color=color, label='BIC-Wine')
    df_gmm[df_gmm['Data']=='pima']['BIC'].plot(linestyle='solid', marker='^', ax=ax326_rev, color=color, label='BIC-Pima')
    ax326_rev.ticklabel_format(useOffset=False, style='plain')
    ax326_rev.set_ylabel('AIC/BIC', color=color)  # we already handled the x-label with ax1
    ax326_rev.tick_params(axis='y', labelcolor=color)
    ax326_rev.grid()
    ax326_rev.legend()
    
    ax4 = plt.subplot(fig_row_col+7)
    ax4_rev = ax4.twinx()
    
    color="tab:green"
    df_kmeans[df_kmeans['Data']=='wine']['V-Measure'].plot(linestyle='dashed', marker='s', ax=ax4, color=color, label='V-Measure')
    df_kmeans[df_kmeans['Data']=='pima']['V-Measure'].plot(linestyle='solid', marker='s', ax=ax4, color=color, label='V-Measure')
    df_kmeans[df_kmeans['Data']=='wine']["Homogeniety Score"].plot(linestyle='dashed', marker='o', ax=ax4_rev, color='b', label='Homogeniety Score')
    df_kmeans[df_kmeans['Data']=='pima']["Homogeniety Score"].plot(linestyle='solid', marker='^', ax=ax4_rev, color='b', label='Homogeniety Score')
    df_kmeans[df_kmeans['Data']=='wine']["Completeness Score"].plot(linestyle='dashed', marker='o', ax=ax4_rev, color='r', label='Completeness Score')
    df_kmeans[df_kmeans['Data']=='pima']["Completeness Score"].plot(linestyle='solid', marker='^', ax=ax4_rev, color='r', label='Completeness Score')
    ax4.ticklabel_format(useOffset=False, style='plain')
    ax4.set_ylabel('V-Measure', color=color)  # we already handled the x-label with ax1
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.grid()
    ax4.legend()
    ax4_rev.ticklabel_format(useOffset=False, style='plain')
    ax4_rev.set_ylabel("Homogeniety/Completeness Score", color=color)  # we already handled the x-label with ax1
    ax4_rev.tick_params(axis='y', labelcolor=color)
    ax4_rev.grid()
    ax4_rev.legend()
    
    ax4 = plt.subplot(fig_row_col+8)
    ax4_rev = ax4.twinx()

    color="tab:green"
    df_gmm[df_gmm['Data']=='wine']['V-Measure'].plot(linestyle='dashed', marker='s', ax=ax4, color=color, label='V-Measure')
    df_gmm[df_gmm['Data']=='pima']['V-Measure'].plot(linestyle='solid', marker='s', ax=ax4, color=color, label='V-Measure')
    df_gmm[df_gmm['Data']=='wine']["Homogeniety Score"].plot(linestyle='dashed', marker='o', ax=ax4_rev, color='b', label='Homogeniety Score')
    df_gmm[df_gmm['Data']=='pima']["Homogeniety Score"].plot(linestyle='solid', marker='^', ax=ax4_rev, color='b', label='Homogeniety Score')
    df_gmm[df_gmm['Data']=='wine']["Completeness Score"].plot(linestyle='dashed', marker='o', ax=ax4_rev, color='r', label='Completeness Score')
    df_gmm[df_gmm['Data']=='pima']["Completeness Score"].plot(linestyle='solid', marker='^', ax=ax4_rev, color='r', label='Completeness Score')

    ax4.ticklabel_format(useOffset=False, style='plain')
    ax4.set_ylabel('V-Measure', color=color)  # we already handled the x-label with ax1
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.grid()
    ax4.legend()
    ax4_rev.ticklabel_format(useOffset=False, style='plain')
    ax4_rev.set_ylabel("Homogeniety/Completeness Score", color=color)  # we already handled the x-label with ax1
    ax4_rev.tick_params(axis='y', labelcolor=color)
    ax4_rev.grid()
    ax4_rev.legend()
    
    plt.suptitle('Clustering-'+title)
    
    plt.tight_layout(pad=2.0)
    #plt.suptitle('Clustering Wine and Pima Data', size=15)
    
    if title=="":
        plot_dir = os.path.join(os.pardir, r'plot', 'step1')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir,'step1_cluster_metrics_plot.png'))
    else:
        plot_dir = os.path.join(os.pardir, r'plot', 'step3')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir,'step3_cluster_metrics_'+title+'.png'))
        
        
        


def dimensionality_reduction(x1, y1, x2, y2):
    run_PCA(x1, y1, x2, y2)
    run_ICA(x1, y1, x2, y2)
    run_RP(x1, y1, x2, y2)
    best_features = run_RFC(x1, y1, x2, y2)
    return best_features

def run_PCA(x1, y1, x2, y2):
    fig = plt.figure(figsize=(12,5))
    
    data = []
    
    for i, (d, x) in enumerate({'Wine': x1, 'Pima': x2}.items()):
        ax = plt.subplot(121+i)
        pca = PCA(whiten=True, random_state=42)
        pca.fit(x)
        ax.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), 'bs--', label='run_PCA-'+d+' Expl.Var')
        
        ax.tick_params(axis='y', labelcolor='b')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_ylabel("Cumu. Expl. Variance Ratio", color='b')
        
        
        ax_rev = ax.twinx()
        ax_rev.plot(np.arange(1, len(pca.singular_values_)+1), pca.singular_values_, 'ro-', label='run_PCA-'+d+' Singl. Vals')
        #print('run_PCA-'+d+'Avg Log Likelihood=', pca.score(x))
        ax_rev.tick_params(axis='y', labelcolor='r')
        ax_rev.ticklabel_format(useOffset=False, style='plain')
        ax_rev.set_ylabel("Singular Values", color='r')
        ax.grid()
        ax_rev.grid()
        
        ax_rev.legend(loc='center right')
        ax.legend()
        ax.set_xlabel('#Principal Components')
        
    plt.suptitle("PCA Screeplot with Singular Values")
    plt.tight_layout()
    plot_dir = os.path.join(os.pardir, r'plot', 'step2','PCA')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir,'step2_PCA_scree_plot.png'))
    

    
    

        
        
def run_ICA(x1, y1, x2, y2):
    fig = plt.figure(figsize=(10,10))
    
    data = []
    
    for i, (d, x) in enumerate({'Wine': x1, 'Pima': x2}.items()):
        #ax = plt.subplot(221+i)
        kurt_values = []
        
        #print(x.shape)
        
        for n in range(2, x.shape[1]):
            ica = FastICA(n_components=n, whiten=True, random_state=42)
            ica.fit(x)
            x_ica = ica.transform(x, )
            kurt_mean = np.mean(np.abs((kurtosis(x_ica, axis=0, fisher=True))))
            kurt_std = np.std(np.abs((kurtosis(x_ica, axis=0, fisher=True))))
        
            #print('n=', n, 'kurt=', np.mean(kurtosis(x_ica, axis=0, fisher=True),axis=0))
            
            kurt_values.append([kurt_mean, kurt_std])
        
        kurt_values = np.array(kurt_values)
        #print(kurt_values)
        
        kurt_before_transform = kurtosis(x, axis=0, fisher=True)
        kurt_values = np.absolute(np.array(kurt_values))
        kurt_values_mean = np.mean(kurt_values, axis=0)
        kurt_values_std = np.std(kurt_values, axis=0)
        
        ax = plt.subplot(221+i)
        ax.plot(range(1,x_ica.shape[1]), kurt_values[:,0], 'bs--', label='run_ICA-'+d+' Kurt-Transformed')
        ax.fill_between(range(1,x_ica.shape[1]), kurt_values[:,0]+kurt_values[:,1], kurt_values[:,0], alpha=0.3)
        #ax.plot(range(1,x.shape[1]+1), kurt_before_transform, 'ro--', label='run_ICA-'+d+' Kurt-Original')
        #print('run_ICA-'+d+'=', pca.score(x))
        ax.legend()
        ax.set_xlabel('#Independent Components')
        ax.set_ylabel('Avg Kurtosis')
        
        ax = plt.subplot(223+i)
        ax_rev = ax.twinx()
        
        for i in range(x_ica.shape[1]):
            sns.kdeplot(x_ica[:, i], color='b',  linestyle='dashed', 
                        ax=ax, 
                       alpha=0.2)
            xi = x[:,i] if type(x)==np.ndarray else x[x.columns[i]]
            sns.kdeplot(xi, color='r', linestyle='dashed', 
                        ax=ax_rev, 
                        alpha=0.2)
        
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_ylabel('Density of recovered signals', color='b')  # we already handled the x-label with ax1
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid()
        ax.set_xlabel('Value')
        #ax.legend()
    
        ax_rev.ticklabel_format(useOffset=False, style='plain')
        ax_rev.set_ylabel('Density for raw data', color='r')  # we already handled the x-label with ax1
        ax_rev.tick_params(axis='y', labelcolor='r')
        ax_rev.grid()
        #ax_rev.legend()
        #print(kurt_values)
        #data.append([pca.score(x)])
        ax_rev.set_xlabel('Value')
    plt.tight_layout()
        
    plot_dir = os.path.join(os.pardir, r'plot', 'step2','ICA')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir,'step2_ICA_kurtosis.png'))
    
    
def get_best_dimensionality_reductions(x1, x2, best_features):
    dim_reds = {}
    for d, x in {'wine':x1, 'pima':x2}.items():
        pca = PCA(n_components=0.95, whiten=True, random_state=42)
        pca.fit(x)
        
        
        
        k = dim_reds.setdefault('pca', {})
        k[d] = pca
        
        
               
        k = dim_reds.setdefault('rfc', {})
        k[d] = best_features[d]
    
    k = dim_reds.setdefault('ica', {})
    ica = FastICA(n_components=8, whiten=True, random_state=42)
    ica.fit(x1)
    k['wine'] = ica
    ica = FastICA(n_components=6, whiten=True, random_state=42)
    ica.fit(x2)
    k['pima'] = ica
        
    k = dim_reds.setdefault('rp', {})
    rp = SparseRandomProjection(random_state=42, n_components=8)
    rp.fit(x1)
    k['wine'] = rp
    
    rp = SparseRandomProjection(random_state=42, n_components=6)
    rp.fit(x2)
    k['pima'] = rp

        
    return dim_reds

    
        
    



def run_RP(x1, y1, x2, y2):
    
    data = []
    for i, (d, x) in enumerate({'Wine': x1, 'Pima': x2}.items()):   
        
        for s in [42, 1, 360, 2020, 1000]:
            for n in range(1, x1.shape[1]+10):
                random_projection = SparseRandomProjection(random_state=s, n_components=n)
                # data has this shape:  row, col = 4898, 11 
    
                random_projection.fit(x)
                components =  random_projection.components_.toarray() # shape=(5, 11) 
                p_inverse = np.linalg.pinv(components.T) # shape=(5, 11) 
    
                #now get the transformed data using the projection components
                reduced_data = random_projection.transform(x)
                reconstructed= reduced_data.dot(p_inverse) 
    
                #print(reduced_data.shape) #(4898, 5)
                #print(reconstructed.shape) #(4898, 11), back in original shape
    
                assert  x.shape ==  reconstructed.shape
                error = mean_squared_error(x, reconstructed)
                
                data.append([d, s, n, error])
                
    data = pd.DataFrame(data, columns=['data', 'seed', '#Projections', 'Reconstruction Error'])
    #print(data)
    recons_error_mean = data.groupby(by=['data', '#Projections'])['Reconstruction Error'].mean().reset_index()
    #print(recons_error_mean)
    recons_error_std = data.groupby(by=['data', '#Projections']).std()['Reconstruction Error'].reset_index()
    #print(recons_error_std)
    figure = fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot(111)
    d = recons_error_mean[recons_error_mean['data']=='Wine']
    ax1.plot(d["#Projections"], d['Reconstruction Error'], 'bo--', label='Wine')
    ax1.fill_between(d["#Projections"], d['Reconstruction Error']-recons_error_std[recons_error_std['data']=='Wine']['Reconstruction Error'],
                    d['Reconstruction Error']+recons_error_std[recons_error_std['data']=='Wine']['Reconstruction Error'], alpha=0.1)
    
    
    d = recons_error_mean[recons_error_mean['data']=='Pima']
    ax1.plot(d["#Projections"], d['Reconstruction Error'], 'rs-', label='Pima')
    ax1.fill_between(d["#Projections"], d['Reconstruction Error']-recons_error_std[recons_error_std['data']=='Pima']['Reconstruction Error'],
                    d['Reconstruction Error']+recons_error_std[recons_error_std['data']=='Pima']['Reconstruction Error'], alpha=0.1)
    ax1.grid()
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_xlabel('#Projections')
    ax1.set_title('Reconstruction Error for Random Projection')
    ax1.legend()
    plt.tight_layout(pad=2.0)
    
    plot_dir = os.path.join(os.pardir, r'plot', 'step2','RP')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir,'step2_random_proj_recons_error.png'))
    


def run_RFC(x1, y1, x2, y2):
    
    best_features = {}
    figure = fig = plt.figure(figsize=(5,4))
    
    for i, (d, (x, y,c )) in enumerate({'wine': (x1,y1, 'ro--'), 'pima': (x2,y2, 'bs-')}.items()):  
        rfc = RandomForestClassifier(n_estimators=100, min_samples_leaf=round(0.1*x.shape[0]), n_jobs=-1, 
                                     random_state=42)
        feat_imp = rfc.fit(x,y).feature_importances_ 
        feat_imp = pd.DataFrame(feat_imp,columns=['Feature Importance'], index=range(x.shape[1]))
        
        feat_imp = feat_imp.sort_values(by=['Feature Importance'], ascending=False)
        feat_imp['Cum Sum Importance'] = feat_imp['Feature Importance'].cumsum()
        ax = plt.subplot(121+i)
        ax.plot([str(i) for i in feat_imp.index.tolist()], feat_imp['Cum Sum Importance'], c, label=d)
        feat_imp = feat_imp[feat_imp['Cum Sum Importance'] <= 0.95]
        top_cols = feat_imp.index.tolist()
        #print(top_cols)
        #plt.xticks(rotation=90)
        ax.set_ylabel('Cum. Importance')
        ax.set_xlabel('Column Number')
        ax.legend()
        ax.grid()
        
        best_features[d] = top_cols
        
    plt.suptitle('Random Forest Classifier Feature Importance')
    plot_dir = os.path.join(os.pardir, 'plot', 'step2','RFC')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir,'step2_random_proj_recons_error.png'))
    
    return best_features


def tune_nn(activation, alpha, hidden_layer_sizes, x2_train_nn_reduced, y2_train_nn, pipe):
    
    tuned_model = tune_hyperparameter({'cfr__alpha':alpha, 
                                        'cfr__hidden_layer_sizes':hidden_layer_sizes, 
                                        'cfr__activation':activation}, 
                                       pipe, x2_train_nn_reduced, y2_train_nn)
    return tuned_model

def main():
    '''
    kmeans, gmm = cluster()
    reduced_data1, reduced_data2 = dimensionality_reduction()    #input 2 data, output 8 dataset
    cluster11, cluster22 = cluster_on_reduced_data(reduced_data)        #output 16 dataset
    compare_clusters(clusters1, clusters2, cluster11, cluster22)
    nn_dim_red = neural_net(reduced_data1)
    nn_cluster = neural_net(clusters1, clusters2)
    compare_neural_nets()
    '''
    x1, y1 = get_dataset1(False)
    x2, y2 = get_dataset2(False)
    X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.33, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.33, random_state=42)
    
    X1_train_clustr, X1_train_nn, y1_train_clustr, y1_train_nn = train_test_split(
                                        X1_train, y1_train, test_size=0.5, random_state=42)
    X2_train_clustr, X2_train_nn, y2_train_clustr, y2_train_nn = train_test_split(
                                        X2_train, y2_train, test_size=0.5, random_state=42)
    

    X1_scaler = MinMaxScaler()
    X2_scaler = MinMaxScaler()
    
    X1_train_clustr = X1_scaler.fit_transform(X1_train_clustr)
    X2_train_clustr = X2_scaler.fit_transform(X2_train_clustr)
    
    X1_train_nn = X1_scaler.transform(X1_train_nn)
    X2_train_nn = X2_scaler.transform(X2_train_nn)
    
    X1_test = X1_scaler.transform(X1_test)
    X2_test = X2_scaler.transform(X2_test)
    
    #===============================================================================================
    # # STEP1: CLUSTER 
    #===============================================================================================
    print("\n=========\n","STEP#1", "\n=========")
    run_cluster(X1_train_clustr,y1_train_clustr,X2_train_clustr,y2_train_clustr, True)
    clusters = get_best_clusters(X1_train_clustr,y1_train_clustr,X2_train_clustr,y2_train_clustr)
    #===============================================================================================
    # # STEP2: DIMENSIONALITY REDUCTION AND FEATURE SELECTION 
    #    FIND BEST PROJECTIONS/FEATURE FOR EACH DIMENSIONALITY REDUCTION/FEATURE SELECTION
    #===============================================================================================
    print("\n=========\n","STEP#2", "\n=========")
    best_features = dimensionality_reduction(X1_train_clustr,y1_train_clustr,X2_train_clustr,y2_train_clustr)
    #best_features = run_RFC(X1_train_clustr,y1_train_clustr,X2_train_clustr,y2_train_clustr)
    print("best RFC features", best_features)
    #best_features = {'wine': [10, 7, 1, 4, 6, 2], 'pima': [1, 5, 7, 6, 3, 2]}
    best_reducers = get_best_dimensionality_reductions(X1_train_clustr, X2_train_clustr, best_features)
    
    #===============================================================================================
    # # STEP3: CLUSTER FOR KMEANS AND GMM FOR 4 DIM. RED. ALGORITHMS AND 2 DATASET. TOTAL 16 COMBO
    #===============================================================================================
    print("\n=========\n","STEP#3", "\n=========")

    for d in best_reducers:
        if d == 'rfc':
            x1_train_clustr_reduced = X1_train_clustr[:, best_reducers[d]['wine']]
            x2_train_clustr_reduced = X2_train_clustr[:, best_reducers[d]['pima']]
        else:
            reducer_wine = best_reducers[d]['wine']
            reducer_pima = best_reducers[d]['pima']
            x1_train_clustr_reduced = reducer_wine.transform(X1_train_clustr)
            x2_train_clustr_reduced = reducer_pima.transform(X2_train_clustr)
                
        run_cluster(x1_train_clustr_reduced, y1_train_clustr, 
                    x2_train_clustr_reduced, y2_train_clustr, plot=True, title=d+'-'+d.upper())
                    
          

    
    #===============================================================================================
    # STEP4: BUILD 4 NEURAL NET MODELS FOR REDUCED PIMA DATASET BY USING THE DIM. RED. ALGORITHMS 
    # Train NN
    #===============================================================================================

    activation = ['logistic', 'tanh']
    alpha = np.logspace(-2, 4, 15)
    hidden_layer_sizes = [(12,6,4,2)]
    result_data = []
    models = {}
    for m in ['pca', 'ica', 'rp', 'rfc', 'benchmark', 'kmeans', 'gmm']:
        k = models.setdefault(m, {})
        k['model'] = pipe = Pipeline([('cfr', MLPClassifier((6, 4, 2), random_state=42, activation='logistic',
                    max_iter=100, tol=0.001, n_iter_no_change=80, learning_rate='adaptive'))])
    
    

    print("\n=========\n","STEP#4", "\n=========")
    #X2_train_nn = X2_scaler.transform(X2_train_nn)
    #X2_test = X2_scaler.transform(X2_test)
    for d in best_reducers:
        print(d.upper(), "\n=========")
        
        if d == 'rfc':
            # REDUCE THE TRAINING SET SAVED FOR NEURAL NET USING THE BEST DIM. RED. ALGOS FROM STEP#2  
            x2_train_nn_reduced = X2_train_nn[:, best_reducers[d]['pima']]
            x2_test_reduced = X2_test[:, best_reducers[d]['pima']]
        else:
            reducer_pima = best_reducers[d]['pima']
            print('reducer=', reducer_pima)
            x2_train_nn_reduced = reducer_pima.transform(X2_train_nn)
            x2_test_reduced = reducer_pima.transform(X2_test)

        #pipe = Pipeline([('cfr', MLPClassifier((6, 4, 2), random_state=42, activation='logistic',
        #            max_iter=100, tol=0.001))])
        #models[d]['model'] = models[d]['model'].fit(x2_train_nn_reduced, y2_train_nn)
        models[d]['model'] = tune_nn(activation, alpha, hidden_layer_sizes, x2_train_nn_reduced, y2_train_nn, models[d]['model'])        
        #tuned_model = pipe
        train_score = models[d]['model'].score(x2_train_nn_reduced, y2_train_nn)
        test_score = models[d]['model'].score(x2_test_reduced, y2_test)
        
        
        result_data.append([d.upper(), x2_train_nn_reduced.shape[1],
                            train_score, test_score,
                            models[d]['model'].best_estimator_['cfr'].loss_curve_, models[d]['model'].refit_time_])

    print('BENCHMARK', "\n=========")

    models['benchmark']['model'].fit(X2_train_nn, y2_train_nn)    
    models['benchmark']['model'] = tune_nn(activation, alpha, hidden_layer_sizes, X2_train_nn, y2_train_nn, models['benchmark']['model'])
    #tuned_model = pipe
    train_score = models['benchmark']['model'].score(X2_train_nn, y2_train_nn)
    test_score = models['benchmark']['model'].score(X2_test, y2_test)
    result_data.append(['BENCHMARK', X2_train_nn.shape[1], 
                        train_score, test_score, models['benchmark']['model'].best_estimator_['cfr'].loss_curve_, models['benchmark']['model'].refit_time_])
    
    
    
    
    #===============================================================================================
    # # STEP 5: ADD CLUSTERS FROM STEP1 AS FEATURES TO THE DATA AND RERUN NEURAL NETWORK
    #===============================================================================================
    print("\n=========\n","STEP#5", "\n=========")
    
    d = 'pima'
        
    c = 'kmeans'
    cluster_algo = clusters[d][c]['obj']
    #print(cluster_algo)
    X2_train_nn1 = X2_train_nn
    #print('X2_train_nn', X2_train_nn)
    
    cluster_train_pred = cluster_algo.predict(X2_train_nn)
    #print("cluster_train_pred", cluster_train_pred)
    cluster_test_pred = cluster_algo.predict(X2_test)
         
    #add the clusters as a new feature
    enhanced_X2_train_nn1 = np.append(X2_train_nn, cluster_train_pred.reshape(-1,1), axis=1)
    enhanced_X2_test1 = np.append(X2_test, cluster_test_pred.reshape(-1,1), axis=1)
                     
    #pipe = Pipeline([('cfr', MLPClassifier((6, 4, 2), random_state=42, activation='logistic', max_iter=100, tol=0.001))])
    models[c]['model'] = models[c]['model'].fit(enhanced_X2_train_nn1, y2_train_nn)
    #tuned_model = tune_nn(activation, alpha, hidden_layer_sizes, enhanced_X2_train_nn, y2_train_nn, pipe)
    models[c]['model'] = tune_nn(activation, alpha, hidden_layer_sizes, enhanced_X2_train_nn1, y2_train_nn, models[c]['model'])        
    train_score = models[c]['model'].score(enhanced_X2_train_nn1, y2_train_nn)
    print(train_score)
    test_score = models[c]['model'].score(enhanced_X2_test1, y2_test)
    #print("models[c]['model']", models[c]['model'])
    #print([c.upper(), enhanced_X2_train_nn1.shape[1],
    #                        train_score, test_score, models[c]['model'].best_estimator_['cfr'].loss_curve_,
                            
    #                        ])  
    result_data.append([c.upper(), enhanced_X2_train_nn1.shape[1],
                            train_score, test_score, models[c]['model'].best_estimator_['cfr'].loss_curve_, 
                            models[c]['model'].refit_time_
                            
                            ])
    #print(result_data)
    

    c = 'gmm'
    cluster_algo = clusters[d][c]['obj']
    #print(cluster_algo)
    #print('X2_train_nn', X2_train_nn)
    cluster_train_pred = cluster_algo.predict(X2_train_nn)
    #print("cluster_train_pred", cluster_train_pred)
    cluster_test_pred = cluster_algo.predict(X2_test)
         
    #add the clusters as a new feature
    enhanced_X2_train_nn = np.append(X2_train_nn, cluster_train_pred.reshape(-1,1), axis=1)
    enhanced_X2_test = np.append(X2_test, cluster_test_pred.reshape(-1,1), axis=1)
                     
    #pipe = Pipeline([('cfr', MLPClassifier((6, 4, 2), random_state=42, activation='logistic', max_iter=100, tol=0.001))])
    #models[c]['model'] = models[c]['model'].fit(enhanced_X2_train_nn, y2_train_nn)
    models[c]['model'] = tune_nn(activation, alpha, hidden_layer_sizes, enhanced_X2_train_nn, y2_train_nn, models[c]['model'])        
    train_score = models[c]['model'].score(enhanced_X2_train_nn, y2_train_nn)
    #print(train_score)
    test_score = models[c]['model'].score(enhanced_X2_test, y2_test)    
    result_data.append([c.upper(), enhanced_X2_train_nn.shape[1],
                            train_score, test_score, models[c]['model'].best_estimator_['cfr'].loss_curve_,
                            models[c]['model'].refit_time_
                            
                            ])
                 
                 
                 
                 
    #print(result_data)
                 
                 
    result_df = pd.DataFrame(result_data, columns=['Model', 'Dimension',  
                     'Training F1 Score', 'Testing F1 Score', 'Loss', 'Training Time(sec)'])
    
     
    test_score_fun(result_df, 'pima-reduced dimension', 'step4_and_5')
         
        
        
    
    
    
    
    
    
    #clusters1, clusters2 = cluster()
    
if __name__=="__main__":
    main()
    