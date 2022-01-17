import seaborn as sns
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from scipy.stats import rankdata
from scipy.stats import shapiro

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)
random.seed(1)

def str_to_list(string,
                Feat_length1,
                Feat_length2):
    
    '''
    
    Args: 
    feature string, start and stop ints for feature selection
    
    Returns:
    list containing feature subset

    '''
    
    string = string.strip('][').split(', ')
    string2 = []
    for entry in string[Feat_length1:Feat_length2+1]:
        string2.append(float(entry))
    return string2

def random_sample(path,
                  n=100,
                  random_state=1,
                  scoring='Score'):
    
    '''
    
    Args: 
    path string, int for sample size, int for random state, y string
    
    Returns:
    dataframe to be used for training, dataframe to be used for testing

    '''
    
    df = pd.read_csv(path)
    df = df[df[scoring] < 0]
    df_pred = df.sample(n, random_state = random_state)
    df_test = df[~df['PCID'].isin(df_pred['PCID'])]
    return df_pred, df_test

def new_train(df_path,
              df_pred,
              scoring='Score'):
    
    '''
    
    Args: 
    path string, training dataframe, y string
    
    Returns:
    dataframe to be used for testing

    '''
    
    df = pd.read_csv(df_path)
    df = df[df[scoring] < 0]
    df_test = df[~df['PCID'].isin(df_pred['PCID'])]
    return df_test

def train_test(df_pred,
               df_test,
               feature,
               scoring,
               Feat_length1,
               Feat_length2):
    
    '''
    
    Args: 
    training dataframe, testing dataframe, feature string, y string,
    int for start point, int for endpoint
    
    Returns:
    X_obs array, y_obs array, X_test array, y_test array

    '''
    
    df_pred['Feat_list'] = df_pred[feature].apply(str_to_list, args=(Feat_length1,Feat_length2,))
    X_obs = list(df_pred['Feat_list'])
    X_obs = np.array(X_obs)
    y_obs = np.array(list(df_pred[scoring]))
    
    scaler = StandardScaler()
    X_obs = scaler.fit_transform(X_obs)
    
    df_test['Feat_list'] = df_test[feature].apply(str_to_list, args=(Feat_length1,Feat_length2,))
    X_test = list(df_test['Feat_list'])
    X_test = np.array(X_test)
    y_test = np.array(list(df_test[scoring]))
    
    X_test = scaler.transform(X_test)
    return X_obs, y_obs, X_test, y_test


def Violin_plot(dataframe,
                output_fig,
                number,
                scoring):
    
    '''
    
    Args: 
    dataframe, path string, int, score string
    
    Returns:
    None (saves plots to output_fig path)

    '''
    
    ax2 = sns.violinplot(x="Iteration", y=scoring, data=dataframe)
    fig2 = ax2.get_figure()
    fig2.savefig(output_fig + f'{number} Score.png', dpi=300)
    
    plt.clf()
    ax3 = sns.violinplot(x="Iteration", y="y_pred", data=dataframe)
    fig3 = ax3.get_figure()
    fig3.savefig(output_fig + f'{number} Pred_Score.png', dpi=300)
    return None


def Box_plot(dataframe,
             output_fig,
             number,
             j,
             scoring):
    
    '''
    
    Args: 
    dataframe, path string, int, int, score string
    
    Returns:
    None (saves plots to output_fig path)

    '''
    
    dataframe = dataframe[dataframe[scoring] < 0]
    ax2 = sns.boxplot(x="Iteration", y=scoring, data=dataframe, palette='viridis')
    
    if scoring == 'FRED':
        ax2.axhline(-8.681415, ls='--')  #-8.681415
    else:
        ax2.axhline(-8., ls='--')
        
    fig2 = ax2.get_figure()
    fig2.savefig(output_fig + f'{number} Score{j}.png', dpi=300)

    plt.clf()
    ax3 = sns.boxplot(x="Iteration", y="y_pred", data=dataframe, palette='viridis')
    
    if scoring == 'FRED':
        ax3.axhline(-8.681415, ls='--')  #-8.681415
    else:
        ax3.axhline(-8., ls='--')
        
    fig3 = ax3.get_figure()
    fig3.savefig(output_fig + f'{number} Pred_Score{j}.png', dpi=300)
    return None


def acquisition_rank(y_pred, var_pred, beta):
    
    '''
    
    Args: 
    y_pred array, var_pred array, int
    
    Returns:
    ranked array

    '''
    
    return rankdata(-y_pred) + (beta * rankdata(-var_pred))

    
def iterate_residuals(kernel,
                      og_path,
                      n,
                      m,
                      output_path,
                      output_fig_path,
                      algorithm=None,
                      uncertainty=0.1,
                      scoring='Score',
                      Feat_length1=0,
                      Feat_length2=56,):
    
    '''
    
    Args: 
    GP kernel, path string, iteration int, iteration int, path string,
    path string, sklearn ML model, uncertainty weighting int, y string, 
    start feature int, end feature int
    
    Returns:
    dataframe, all predicted values plus metrics at each iteration,
    saves dataframes containing all molecule data plus targetted summaries of
    top ranked molecules

    '''
    
    
    for j in range(1, n+1):
        
        df_pred, df_test = random_sample(og_path, 100, j, scoring)
    
        X_obs, y_obs, X_test, y_test = train_test(df_pred,
                                                  df_test,
                                                  'Feature',
                                                  scoring,
                                                  Feat_length1,
                                                  Feat_length2)
        info_wholeset = []
        info_predicted = []
        
        for i in range(1, m+1):
            
            # define kernel and fit/predict
            if algorithm == None:
                #fit
                gp = GaussianProcessRegressor(kernel=kernel,
                                              n_restarts_optimizer=9,
                                              normalize_y=True)
                gp.fit(X_obs,y_obs)
                score_obs = gp.score(X_obs, y_obs)
                y_pred, sigma = gp.predict(X_test, return_std = True)
                score_test = r2_score(y_test, y_pred)
                
            if algorithm != None:
                #fit
                algorithm.fit(X_obs,y_obs)
                y_pred = algorithm.predict(X_obs)
                
                gp = GaussianProcessRegressor(kernel=kernel,
                                              n_restarts_optimizer=9,
                                              normalize_y=True)
                gp.fit(X_obs,y_obs-y_pred)
                
                #predict
                residuals, sigma = gp.predict(X_test, return_std = True)
                
                y_pred = algorithm.predict(X_test)
                y_pred = y_pred+residuals
                score_obs = algorithm.score(X_obs, y_obs)
                score_test = r2_score(y_test, y_pred)
            
            #analyse test set result
            df_test['y_pred'] = y_pred
            df_test['sigma'] = sigma
            
            y_pred_mean = df_test['y_pred'].mean()
            y_sigma_mean = df_test['sigma'].mean()
            score_mean = df_test[scoring].mean()
            
            #test for difference in grad from 0
            lm = ols(f'y_pred~{scoring}',data=df_test).fit()
            aov = anova_lm(lm,typ=1)
            degrees_freedom, F, p_value = aov.iloc[0, 0], aov.iloc[0, 3], aov.iloc[0, 4]
            
            #correlation and mse
            pcorr = df_test[[scoring, 'y_pred']].corr('pearson').iloc[0,1]
            scorr = df_test[[scoring, 'y_pred']].corr('spearman').iloc[0,1]    
            mse = mean_squared_error(df_test[scoring], df_test['y_pred'])
            residuals = np.abs(df_test['y_pred']) - np.abs(df_test[scoring])
            normality = shapiro(residuals)
            
            info_wholeset.append((i,
                                  y_pred_mean,
                                  y_sigma_mean,
                                  score_mean,
                                  score_obs,
                                  score_test,
                                  mse,
                                  np.sqrt(mse),
                                  pcorr,
                                  scorr,
                                  degrees_freedom,
                                  F,
                                  p_value,
                                  normality))
           
            
            if uncertainty > 0:
                df_test['acquire'] = acquisition_rank(y_pred, sigma, uncertainty)
                df_top100 = df_test.nlargest(100, 'acquire')
                df_top100['Iteration'] = i
            else:       
                df_top100 = df_test.nsmallest(100, 'y_pred')
                df_top100['Iteration'] = i
            
            #analyse predicted subset
            y_pred_mean = df_top100['y_pred'].mean()
            y_sigma_mean = df_top100['sigma'].mean()
            score_mean = df_top100[scoring].mean()
            
            #test for difference in grad from 0
            lm = ols(f'y_pred~{scoring}',data=df_top100).fit()
            aov = anova_lm(lm,typ=1)
            degrees_freedom, F, p_value = aov.iloc[0, 0], aov.iloc[0, 3], aov.iloc[0, 4]
            
            #correlation, mse, Tanimoto
            pcorr = df_top100[[scoring, 'y_pred']].corr('pearson').iloc[0,1]
            scorr = df_top100[[scoring, 'y_pred']].corr('spearman').iloc[0,1]    
            mse = mean_squared_error(df_top100[scoring], df_top100['y_pred'])
            residuals = np.abs(df_top100['y_pred']) - np.abs(df_top100[scoring])
            normality = shapiro(residuals)
            
            Tan_48 = df_top100.Tanimoto_48.mean()
            Tan_52 = df_top100.Tanimoto_52.mean()
            Tan_68 = df_top100.Tanimoto_68.mean()
            Tan_69 = df_top100.Tanimoto_69.mean()
            r2 = r2_score(np.array(df_top100[scoring]), np.array(df_top100['y_pred']))

            if scoring == 'FRED':
                hits = len(df_top100[df_top100[scoring] < -9.395770])
            else:
                hits = len(df_top100[df_top100[scoring] < -8.5])
                
                
            info_predicted.append((i,
                                   y_pred_mean,
                                   y_sigma_mean,
                                   score_mean,
                                   score_obs,
                                   score_test,
                                   mse,
                                   np.sqrt(mse),
                                   pcorr,
                                   scorr,
                                   degrees_freedom,
                                   F,
                                   p_value,
                                   Tan_48,
                                   Tan_52,
                                   Tan_68,
                                   Tan_69,
                                   r2,
                                   normality,
                                   hits))
            
            if i == 1:
                df_summ = df_top100.copy()
            else:
                df_summ = df_summ.append(df_top100)
            
            df_top100.drop(columns = ['y_pred'], inplace = True)
            df_pred = df_pred.append(df_top100)
            
            df_test = new_train(og_path, df_pred)
            
            X_obs, y_obs, X_test, y_test = train_test(df_pred,
                                                      df_test,
                                                      'Feature',
                                                      scoring,
                                                      Feat_length1,
                                                      Feat_length2)

        plt.clf()
        
        Box_plot(df_summ, output_fig_path, n, j, scoring)
        
        df2 = pd.DataFrame(info_wholeset)
        df2.columns = ['Iteration', 
                       'Pred_Score_Mean',
                       'Uncertainty_Mean',
                       'Score_Mean',
                       'Score_obs',
                       'Score_test',
                       'MSE', 
                       'RMSE', 
                       'pcorr', 
                       'scorr', 
                       'degrees_freedom',
                       'F value', 
                       'p_value',
                       'Normality']
        df3 = pd.DataFrame(info_predicted)
        df3.columns = ['Iteration', 
                       'Pred_Score_Mean',
                       'Uncertainty_Mean',
                       'Score_Mean',
                       'Score_obs',
                       'Score_test',
                       'MSE', 
                       'RMSE', 
                       'pcorr', 
                       'scorr', 
                       'degrees_freedom',
                       'F value', 
                       'p_value',
                       'Tanimoto_48',
                       'Tanimoto_52',
                       'Tanimoto_68',
                       'Tanimoto_69',
                       'r2_score',
                       'Normality',
                       'No. of hits']
        
        df_summ.to_csv(output_path + f'{j}predictedfulldata.csv')
        df2.to_csv(output_path + f'{j}summarytestset.csv')
        df3.to_csv(output_path + f'{j}summarytop100.csv')
    return df2


kernel = C(1., 'fixed') * Matern(length_scale=1.0, length_scale_bounds='fixed', nu=1.5)
algorithm = RandomForestRegressor(bootstrap= False,
                                 max_depth = 100,
                                 max_features = 'log2',
                                 min_samples_leaf = 2,
                                 min_samples_split = 2,
                                 n_estimators = 950,
                                 random_state=1)
ROOT = '../data/'
og_path = ROOT+'zinc.csv'
os.mkdir(ROOT+'ITERATION/Summary/')
os.mkdir(ROOT+'ITERATION/Figures/')
output_path = ROOT+'ITERATION/Summary/'
output_fig_path = ROOT+'ITERATION/Figures/'

summary = iterate_residuals(kernel,
                            og_path,
                            10,
                            10,
                            output_path,
                            output_fig_path,
                            algorithm,
                            uncertainty=0.1,
                            scoring='VINA',
                            Feat_length1=0,
                            Feat_length2=56,
                            )





















