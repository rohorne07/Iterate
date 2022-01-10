import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import scipy.stats as ss
from sklearn.ensemble import RandomForestRegressor

from predict import run_algorithm_residuals


random.seed(1)
np.random.seed(1)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


algorithm = RandomForestRegressor(bootstrap= False,
                                 max_depth = 100,
                                 max_features = 'log2',
                                 min_samples_leaf = 2,
                                 min_samples_split = 2,
                                 n_estimators = 950,
                                 random_state=1)
kernel = C(1., 'fixed') * Matern(length_scale=1.0, length_scale_bounds='fixed', nu=1.5)

ROOT = '/Users/rohorne07/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine Learning/Docking/TRIALSET/'
og_path = ROOT + 'Docking_Thalves_ITERATION_3_hits.csv_MPO.csv'
og_path2 = ROOT + 'Docker_Tanimoto_ZINC_PCID.csv'
output_path = ROOT + 'ITERATION/Summary/'
output_fig_path = ROOT + 'ITERATION/Figures/'


df_top100_gp = []
df_test_gp = []


df_top100_rfr = []
df_test_rfr = []
df_coeff = pd.DataFrame()

for i in [1,10,100,1000,10000,100000,1000000, 10000000, 100000000, 42]:

    df_top100rfr, df_testrfr, feat_coeff_rfr, X_obs, y_obs, X_test = run_algorithm_residuals(og_path,
    og_path2,
    algorithm,
    kernel,
    scoring='Norm_T_half',
    feature = 'Feature',
    Feat_length1=0,
    Feat_length2=56,
    uncertainty=0.1,
    T69=False,
    T52=False,
    random_state=i)
    
    df_top100_rfr.append(df_top100rfr)
    df_test_rfr.append(df_testrfr)

df_label = pd.read_csv(ROOT + 'Docking_Thalves_ITERATION_3.csv')

df_parent = pd.read_csv(ROOT + 'Docker_Tanimoto_ZINC_PCID.csv')

df_parent = df_parent[~df_parent['PCID'].isin(df_label['PCID'])]


dock = np.array(df_label['Docking'])
loose = np.array(df_label['Loose'])
close = np.array(df_label['Close'])
iter1 = np.array(df_label['Iter1'])
iter2 = np.array(df_label['Iter2'])
iter3 = np.array(df_label['Iter3'])


o48 = np.array(df_label['48'])
o52 = np.array(df_label['52'])
o68 = np.array(df_label['68'])
o69 = np.array(df_label['69'])

p48 = np.array(df_parent['48'])
p52 = np.array(df_parent['52'])
p68 = np.array(df_parent['68'])
p69 = np.array(df_parent['69'])


labels_dock = np.concatenate([
    dock, np.zeros(len(X_test))
])
labels_close = np.concatenate([
    close, np.zeros(len(X_test))
])
labels_loose = np.concatenate([
    loose, np.zeros(len(X_test))
])
labels_iter1 = np.concatenate([
    iter1, np.zeros(len(X_test))
])
labels_iter2 = np.concatenate([
    iter2, np.zeros(len(X_test))
])
labels_iter3 = np.concatenate([
    iter3, np.zeros(len(X_test))
])


labels_48 = np.concatenate([
    o48, p48
])
labels_52= np.concatenate([
    o52, p52
])
labels_68= np.concatenate([
    o68, p68
])
labels_69= np.concatenate([
    o69, p69
])



from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1).fit(X_obs)
dist = np.ravel(nbrs.kneighbors(X_test)[0])

X_obs_minus_dock = np.zeros(len(X_obs))+dock

X = np.vstack([ X_obs, X_test ])
labels_minus_dock = np.concatenate([
    X_obs_minus_dock, np.ones(len(X_test))
])
labels = np.concatenate([
    np.zeros(len(X_obs)), np.ones(len(X_test))
])


from fbpca import pca
U, s, Vt = pca(X, k=3)
X_pca = U * s


from umap import UMAP
um = UMAP(
    n_neighbors=15,
    min_dist=0.5,
    n_components=2,
    metric='euclidean',
    random_state=70000
)
X_umap = um.fit_transform(X)

from MulticoreTSNE import MulticoreTSNE as TSNE

tsne = TSNE(
        n_components=2,
        n_jobs=20,
        random_state=1
    )

X_tsne = tsne.fit_transform(X)


def graph_parent(file, title, stage):
    for name, coords in zip(
            [ 'pca', 'umap', 'tsne' ],
            [ X_pca, X_umap, X_tsne ],
    ):
        plt.figure()
        sns.scatterplot(x=coords[labels_48 == 1, 0], y=coords[labels_48 == 1, 1],
                        color='xkcd:green', alpha=0.1,label = '48')
        sns.scatterplot(x=coords[labels_52 == 1, 0], y=coords[labels_52 == 1, 1],
                        color='xkcd:teal', alpha=0.1,label = '52')
        sns.scatterplot(x=coords[labels_68 == 1, 0], y=coords[labels_68 == 1, 1],
                        color='xkcd:purple', alpha=0.1,label = '68')
        sns.scatterplot(x=coords[labels_69 == 1, 0], y=coords[labels_69 == 1, 1],
                        color='xkcd:blue', alpha=0.1, label = '69')


        plt.scatter(x=coords[stage==1, 0], y=coords[stage==1, 1],
                    color='orange', alpha=1.0, marker='x', linewidths=10, label = 'Training Subset')
        
        plt.legend()
        plt.scatter(x=coords[0, 0], y=coords[0, 1],
                    color='tab:red', alpha=1.0, marker='o', linewidths=5,)
        plt.scatter(x=coords[1, 0], y=coords[1, 1],
                    color='tab:red', alpha=1.0, marker='o', linewidths=5,)
        plt.scatter(x=coords[2, 0], y=coords[2, 1],
                    color='tab:red', alpha=1.0, marker='o', linewidths=5,)
        plt.scatter(x=coords[3, 0], y=coords[3, 1],
                    color='tab:red', alpha=1.0, marker='o', linewidths=5,)
        
        plt.savefig(ROOT+'higher dimension/{}/{}/latent_scatter_{}_ypred_{}{}.png'
                    .format(file,title, name, 'RFR+GP', 'asyn1'), dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(x=coords[labels == 1, 0], y=coords[labels == 1, 1],
                    c=ss.rankdata(df_test_rfr[0]['sigma']), alpha=0.1, cmap='coolwarm')

        plt.savefig(ROOT+'higher dimension/{}/{}/latent_scatter_{}_var_{}{}.png'
                    .format(file,title, name, 'RFR+GP', 'asyn1'), dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(x=coords[labels == 1, 0], y=coords[labels == 1, 1],
                    c=df_test_rfr[0]['acquire'], alpha=0.1, cmap='YlGnBu')

        plt.savefig(ROOT+'higher dimension/{}/{}/latent_scatter_{}_acq_{}{}.png'
                    .format(file,title, name, 'RFR+GP', 'asyn1'), dpi=300)
        plt.close()
 


