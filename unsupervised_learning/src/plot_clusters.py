''' 
CODE COPIED FROM: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

Python Data Science Handbook
by Jake VanderPlas
Released November 2016
Publisher(s): O'Reilly Media, Inc.
ISBN: 9781491912058

'''
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None, save_loc='', title=''):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    #===============================================================================================
    # centers = kmeans.cluster_centers_
    # radii = [cdist(X[labels == i], [center]).max()
    #          for i, center in enumerate(centers)]
    # for c, r in zip(centers, radii):
    #     ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
    # 
    #===============================================================================================
    head = 'cluster_vis_kmeans_k{}_'.format(n_clusters) + title
    ax.set_title(head)
    
    plot_dir = os.path.join(os.pardir, r'plot', save_loc)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir,head+'.png'))
    plt.close()
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, n_clusters, label=True, ax=None, save_loc='', title=''):
    #===============================================================================================
    # ax = ax or plt.gca()
    # labels = gmm.fit(X).predict(X)
    # if label:
    #     ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # else:
    #     ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    # ax.axis('equal')
    # 
    # w_factor = 0.2 / gmm.weights_.max()
    # for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    #     draw_ellipse(pos, covar, alpha=w * w_factor)
    #===============================================================================================
    labels = gmm.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    #ax.scatter(gmm.cluster_centers_, s=40, color='r', marker='X',)
    head = 'cluster_vis_gmm_k{}_'.format(n_clusters) + title
    ax.set_title(head)
    
    plot_dir = os.path.join(os.pardir, r'plot', save_loc)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir,head+'.png'))
    plt.close()