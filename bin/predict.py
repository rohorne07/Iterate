import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rankdata
import os

random.seed(1)
np.random.seed(1)

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


    
def RUN(filename1,):
    
    '''
    
    Args: 
    String to append to output filenames
    
    Returns:
    After running 10 trainings from 10 different random starting states the top
    100 molecules from each random state are appended into a dataframe labelled
    Top_1000. All of the test values are grouped on molecule identifier,
    averaging predictions in a dataframe labelled TEST. The molecules in the
    Top_1000 dataframe are similarly grouped to produce a dataframe labelled
    Top_100_Aggregated which also contains the number of times each molecule 
    appeared in the Top_1000 dataframe under the column 'replicates'
        
    
    '''
    
    df_top100_rfr = []
    df_test_rfr = []
    
    
    for i in [1,10,100,1000,10000,100000,1000000, 10000000, 100000000, 42]:
        df_top100rfr, df_testrfr, feat_importance, X_obs, y_obs, X_test = run_algorithm_residuals(og_path,
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
        
    dfTOP_RFR56 = pd.concat(df_top100_rfr)
    dfTEST_RFR56 = pd.concat(df_test_rfr)
    
    dfTOP_RFR56.to_csv(output_path+f'Top_1000_{filename1}.csv')
    dfTEST_RFR56_grouped = dfTEST_RFR56.groupby('PCID').agg('mean')
    dfTEST_RFR56_grouped.to_csv(output_path+f'TEST_{filename1}.csv')
    
    molecule_count_rfr56 = dfTOP_RFR56['PCID'].value_counts()
    molecule_count_rfr56_grouped = dfTOP_RFR56.groupby('PCID').agg('mean')
    molecule_count_rfr56 = pd.DataFrame(molecule_count_rfr56).reset_index()
    molecule_count_rfr56.rename(columns = {'index':'PCID', 'PCID':'Replicates'}, inplace = True)
    
    df_final = molecule_count_rfr56_grouped.merge(molecule_count_rfr56, how = 'left', on = 'PCID')
    df_final.to_csv(output_path+f'Top_100_Aggregated_{filename1}.csv')

algorithm = RandomForestRegressor(
                                 max_depth = 50,
                                 max_features = 'log2',
                                 min_samples_leaf = 2,
                                 min_samples_split = 2,
                                 n_estimators = 950,
                                 random_state=1)
kernel = C(1., 'fixed') * Matern(length_scale=1.0, length_scale_bounds='fixed', nu=1.5)


ROOT = '../data/'

og_path = ROOT + 'docker.csv'
og_path2 = ROOT + 'zinc.csv'

os.makedirs(ROOT + 'Prediction')
output_path = ROOT + 'Prediction/'

    
RUN('ITERATION')    
    
