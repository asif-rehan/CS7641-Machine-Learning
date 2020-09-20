import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

this_dir =  os.path.dirname(__file__)

def get_dataset1(verbose=True):
    df = pd.read_csv(os.path.join(this_dir, os.pardir, "data", "winequality-white.csv"), sep=';')
    y1 = df['quality'] >= 6
    print(y1.describe())
    X1 = df.drop(columns='quality')
    
    if verbose:
        
        print("Value Counts", y1.value_counts())
        print(df.info())
        print(df.shape)
        print(df.describe())
        #plt.rcParams["figure.figsize"] = 20, 20
        # Basic correlogram
        sns.pairplot(df)
        #plt.show()
        plt.savefig(os.path.join(this_dir, os.pardir, "plot", 'wine_scatterplot.png'))
        plt.close()
        print(X1.shape)
        print(y1.shape)
        print(X1.head())
        print(y1.head())

    return X1, y1


def get_dataset2(verbose=True):
    df = pd.read_csv(os.path.join(this_dir, os.pardir, "data", "diabetes.csv"))
    y2 = df['Outcome']
    X2 = df.drop(columns='Outcome')
    
    if verbose:
        print("Value Counts", y2.value_counts())
        print(df.shape)
        print(df.info())
        print(df.describe())
        #plt.rcParams["figure.figsize"] = 20, 20
        # Basic correlogram
        sns.pairplot(df)
        #plt.show()
        plt.savefig(os.path.join(this_dir, os.pardir, "plot", 'pima_scatterplot.png'))
        plt.close()
        print(X2.shape)
        print(y2.shape)
        print(X2.head())
        print(y2.head())
        
    return X2, y2
