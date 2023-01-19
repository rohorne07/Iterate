import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import rankdata
import os

def str_to_list(string, 
                Feat_length1, 
                Feat_length2):
    '''
    
    Args: 
    Feature string, start and stop points for feature selection
    
    Returns:
    list of feature subset
    
    '''
    string = string.strip('][').split(', ')
    string2 = []
    for entry in string[Feat_length1:Feat_length2+1]:
        string2.append(float(entry))
    return string2


def new_train(df_path,
              df_path2,
              scoring='Norm_T_half',
              feature = 'Feature',
              Feat_length1=0,
              Feat_length2=56,
              T69=False,
              T52=False):
    
    '''
    
    Args: 
    path string (training dataframe), path string (testing dataframe), string 
    identifying target y value column, string identifying feature column, 
    start and stop ints of the feature set to be used, bools to choose
    whether to only train on a single substructure set.
    
    Returns:
    X_obs array, y_obs array, X_test array, test dataframe with training 
    instances removed, training dataframe
    
    '''
    
    df_pred = pd.read_csv(df_path)
    
    if T69==True:
        df_pred = df_pred[df_pred['Tanimoto_69'] > 0.25]
    if T52==True:
        df_pred = df_pred[df_pred['Tanimoto_52'] > 0.25]
    
    df_pred['Feat_list'] = df_pred[feature].apply(str_to_list, args=(Feat_length1,Feat_length2,))
    
    X_obs = list(df_pred['Feat_list'])
    X_obs = np.array(X_obs)
    y_obs = np.array(list(df_pred[scoring]))
    scaler = StandardScaler()
    X_obs = scaler.fit_transform(X_obs)
    
    df_test = pd.read_csv(df_path2)
    if T69==True:
        df_test = df_test[df_test['Tanimoto_69'] > 0.25]
    if T52==True:
        df_test = df_test[df_test['Tanimoto_52'] > 0.25]
        
    df_test = df_test[~df_test['PCID'].isin(df_pred['PCID'])]
    
    df_test['Feat_list'] = df_test[feature].apply(str_to_list, args=(Feat_length1,Feat_length2,))
    X_test = list(df_test['Feat_list'])
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    
    return X_obs, y_obs, X_test, df_test, df_pred



def acquisition_rank(y_pred,
                     var_pred,
                     beta):
    
    '''
    
    Args: 
    Predicted y array, predicted var array, weight int of uncertainty
    
    Returns:
    Ranking of molecules minimising uncertainty and maximising y

    '''
    
    return rankdata(y_pred) + (beta * rankdata(-var_pred))


def run_algorithm_residuals(df_path,
        df_path2,
        algorithm,
        kernel,
        scoring='Norm_T_half',
        feature = 'Feature',
        Feat_length1=0,
        Feat_length2=56,
        uncertainty=0.1,
        T69=False,
        T52=False,
        random_state=1):
    
    
    '''
    
    Args: 
    Path string to training dataframe, path string to testing dataframe,
    sklearn ML model, GP kernel, y string, feature string, start and stop ints
    of the feature set to be used, uncertainty weighting int, bools to choose
    whether to only train on a single substructure set, random state int 
    
    Returns:
    Dataframe of top 100 molecules predicted, dataframe of whole test set with
    predicted values, feature importance dataframe, X_obs feature array, y_obs
    target array, X_test feature array
    
    '''
    
    X_obs, y_obs, X_test, df_test, df_pred = new_train(df_path,
                                            df_path2,
                                            scoring=scoring,
                                            feature=feature,
                                            Feat_length1=Feat_length1,
                                            Feat_length2=Feat_length2,
                                            T69=T69,
                                            T52=T52)
    
    algorithm.fit(X_obs,y_obs)
    y_pred = algorithm.predict(X_obs)
    
    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=9,
                                  normalize_y=True,
                                  random_state=random_state)
    gp.fit(X_obs,y_obs-y_pred)
    
    #predict
    residuals, sigma = gp.predict(X_test, return_std = True)
    
    y_pred = algorithm.predict(X_test)
    y_pred = y_pred+residuals
    
    df_test['y_pred'] = y_pred
    df_test['sigma'] = sigma
    temp_array = algorithm.feature_importances_
    
    if uncertainty > 0:
        df_test['acquire'] = acquisition_rank(y_pred, sigma, 0.1)
        df_top100 = df_test.nlargest(100, 'acquire')
    #analyse predicted subset
    else:       
        df_top100 = df_test.nlargest(100, 'y_pred')
    temp_array = algorithm.feature_importances_
    
    
    return df_top100, df_test, temp_array, X_obs, y_obs, X_test

