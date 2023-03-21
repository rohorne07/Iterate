import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import os
cwd = os.getcwd()
print(cwd)

import pandas as pd
np.random.seed(1)
random.seed(1)

from sklearn.preprocessing import StandardScaler



def str_to_list(string):
    string = string.strip('][').split(', ')
    string2 = []
    for entry in string:
        string2.append(float(entry))
    return string2

def random_sample(path, n=100, scoring='Score'):
    df = pd.read_csv(path)
    df = df[df[scoring] < 0]
    df_pred = df.sample(n, random_state = 1)
    df_test = df[~df['PCID'].isin(df_pred['PCID'])]
    return df_pred, df_test

def train_test(df_pred, df_test, feature):
    df_pred['Feat_list'] = df_pred[feature].apply(str_to_list)
    X_obs = list(df_pred['Feat_list'])
    X_obs = np.array(X_obs)
    y_obs = np.array(list(df_pred['VINA']))
    scaler = StandardScaler()
    scaler = scaler.fit(X_obs)
    X_obs = scaler.fit_transform(X_obs)
    
    df_test['Feat_list'] = df_test[feature].apply(str_to_list)
    X_test = list(df_test['Feat_list'])
    X_test = np.array(X_test)
    y_test = np.array(list(df_test['VINA']))
    X_test = scaler.transform(X_test)
    
    return X_obs, y_obs, X_test, y_test


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    """
    Generate 3 plots for 2 models: the training and cross validation R2 score,
    the fit times with increasing training sample, and the fit times vs R2
    score.
    
    Args:
    estimator instance, chart title string, X array, y array, axes, ylim int,
    cross validation int, n_jobs to run in parallel int, train sizes array
        
    Returns:
    plot of R2 score vs increasing training set size for training and cross 
    validation sets
    plot of fit times with increasing training set size
    plot of R2 score with increasing fit times
        
    """
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                        scoring = 'r2')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class estimator_pipeline(BaseEstimator):
    def __init__(self, algorithm, gp2):
        self.algorithm = algorithm
        self.gp2 = gp2
    def fit(self, X, y):
        self.algo_ = self.algorithm.fit(X, y)
        self.y_pred_ = self.algo_.predict(X)
        self.gp_ = self.gp2.fit(X,y-self.y_pred_)
        return self
    def predict(self, X):
        check_is_fitted(self)
        residuals, sigma = self.gp_.predict(X, return_std = True)
        y_pred = self.algo_.predict(X)
        self.y_pred2_ = y_pred+residuals
        # print(self.y_pred2_)
        return self.y_pred2_
    def score(self, y, y2):
        return r2_score(y, y2)

# path = '../data/zinc.csv'
path = '/Users/rohorne07/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/Paper_1/Iterate-main/data/zinc.csv'
df_pred, df_test = random_sample(path, 4000, 'VINA')
X_obs, y_obs, X_test, y_test = train_test(df_pred, df_test, 'Feature_half')

kernel2 = C(1.0, 'fixed') * Matern(length_scale=1.0, length_scale_bounds='fixed', nu=1.5)
gp2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=9, normalize_y=True)

algorithm = RandomForestRegressor(max_depth = 50,
                                 max_features = 'log2',
                                 min_samples_leaf = 2,
                                 min_samples_split = 2,
                                 n_estimators = 950,
                                 random_state=1)
                                

est = estimator_pipeline(algorithm, gp2)


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X, y = X_obs, y_obs


title = r"Learning Curves (GP)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = gp2
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0, 1.01),
                cv=cv, n_jobs=4)


title = r"Learning Curves (GP x RFR)"
estimator = est
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0, 1.01),
                    cv=cv, n_jobs=4)
plt.tight_layout()
plt.show()



