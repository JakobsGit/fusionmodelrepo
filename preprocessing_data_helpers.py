# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def replacenans(dataset):
    r"""
    Replaces all NaN in stock data DataFrame with data from day before
    
    Parameters
    ----------
    dataset : DataFrame output from getsp500data()
        Containing the following Columns:
            Open, High, Low, Volume

    Returns
    -------
    None.

    """
    
    dataset[dataset['Open'].isna()==True]
    naindex = dataset.index[dataset['Open'].isna()==True]
    
    
    for i in naindex:
        dataset.loc[i,'Open'] = float(dataset.loc[(i-1),'Open'])
        dataset.loc[i,'High']=float(dataset.loc[i-1,'High'])
        dataset.loc[i,'Low']=float(dataset.loc[i-1,'Low'])
        dataset.loc[i,'Close']=float(dataset.loc[i-1,'Close'])
        dataset.loc[i,'Volume']=float(dataset.loc[i-1,'Volume'])

    return



def createreturncolumn(dataset,n, approach):
    r"""
    Creates a new column containing returns for a DataFrame with stock data
    
    Parameters
    ----------
    dataset : DataFrame containing the Columns Close and Stock
        
    n: Parameter to calculate stock return after n trade days
    approach: triggers different return calculations 

    Returns
    -------
    dataset : DataFrame
        Input DataFrame with additional Return Columns

    """

    dataset['Return']= np.zeros(dataset.shape[0]) 
    dataset['Return'] = dataset.shift(-n)['Close'].values
    
    dataset['1dayReturn']= np.zeros(dataset.shape[0])
    dataset['1dayReturn'] = dataset.shift(-1)['Close'].values
    
    rowstobedeleted = []  
    for stockindex in np.unique(dataset['Stock']):
            for i in range(-n,0):
                rowstobedeleted.append(dataset.index[dataset['Stock']==stockindex][i])   
    rowstobedeleted = np.array(rowstobedeleted)    
    dataset = dataset.drop(rowstobedeleted, axis=0)
    
    dataset.reset_index(level=0, inplace=True)
    dataset = dataset.drop(columns='index')
        
    if (approach == 240) | (approach == 31):
        dataset['Return'] = dataset['Return']/dataset['Close']
        dataset['Return'] = dataset['Return']-1
        
        dataset['1dayReturn'] = dataset['1dayReturn']/dataset['Close']
        dataset['1dayReturn'] = dataset['1dayReturn']-1
    
    if (approach == 63) | (approach == 241):

        dataset['Return'] = dataset['Return']-dataset['Close']     
        dataset['1dayReturn'] = dataset['1dayReturn']-dataset['Close']
    
    return dataset



def createtargetcolumn(dataset, approach): 
    r"""
    approach = 240 or approach = 31:
    Creates Target column indicating if the return of a stock is higher
    than the median return of the set of stocks for each day
    
    approach = 63:  
    creates target column containing the absolute return 
    (price at time point t - price at time point t-n)
    
    Parameters
    ----------
    dataset : DataFrame with stock data containing the following columns:
        Date, Return
    
    approach: triggers different target calculations

    Returns
    -------
    dataset : DataFrame with additional column Target
    
    """
    if (approach == 240) | (approach == 31):
        dataset['Target'] = 0
        dataset.loc[dataset['Return']>0,'Target'] = 1

    if (approach == 63) | (approach == 241):
        dataset['Target'] = dataset['Return']        

    return dataset


def deletedividendentries(dataset):
    r"""
    deletes all entries for a date of a stock data DataFrame that contain 
    only dividend information and no price movements 

    Parameters
    ----------
    dataset : DataFrame with stock data containing the following columns:
        Date, Stock, Close, Dividends

    Returns
    -------
    dataset : Cleaned DataFrame

    """
    
    limitnumber = np.unique(dataset['Stock']).shape[0]*0.2
    for dateindex in np.unique(dataset['Date']):
        if dataset[dataset['Date']==dateindex].shape[0] < limitnumber:
            #print(dateindex)
            #print(dateindex)
            #print(dataset[dataset['Date']==dateindex].shape[0])
            rowindex = dataset.index[dataset['Date']==dateindex]
            for i in rowindex:
              #print(i)
              if float(dataset.loc[i]['Close']) == float(dataset.loc[i-1]['Close']):
                  #print(dataset.loc[rowindex])
                  #print(dataset.loc[rowindex]['Dividends'])
                  #print(dataset.loc[rowindex-1])
                  dataset = dataset.drop(i, axis=0)
                  dataset.reset_index(level=0, inplace=True)
                  dataset = dataset.drop(columns='index')
    return dataset
    


def createseries(dataset, timesteps, n, returnfeature):

    dataset = dataset.drop(columns = ['Dividends', 'Stock Splits' ])          
    dataset = dataset.reset_index(level=0, drop=True)
    used_only_as_input =(timesteps+n)*np.unique(dataset['Stock']).shape[0]
    datapoints = dataset.shape[0]  

    if returnfeature == 1: 
        X = np.zeros((datapoints-used_only_as_input, timesteps, 5))
    else: 
        X = np.zeros((datapoints-used_only_as_input, timesteps, 4))   
    
    y = []
    y_df = dataset.iloc[0:1,:]
    
    endindex = np.unique(dataset['Date']).shape[0]
    slicenum = 0

    for i in range(timesteps+n-1, endindex-1):
        print(100*i/endindex," %")
        perioddata = dataset[dataset.Date >=np.unique(dataset.Date)[i-(timesteps+n-1)]]
        datelimit = np.unique(dataset.Date)[i]

        perioddata = perioddata[perioddata.Date <= datelimit]                                     
        for stockindex in np.unique(perioddata['Stock']):
            one_stock_data = perioddata[perioddata.Stock==stockindex]
            if one_stock_data.shape[0]>=timesteps+n:
                
                if returnfeature == 1:
                    X[slicenum,:,:4] = np.array(one_stock_data.iloc[(-n-timesteps+1):, [1,2,3,4]])
                    X[slicenum,:,4] = np.array(one_stock_data.iloc[(-n-timesteps):-n, 8])
                else:
                    X[slicenum,:,:] = np.array(one_stock_data.iloc[(-n-timesteps):-n, [1,2,3,4]])
                y_vec = np.array(one_stock_data['Target'])
                y.append(float(y_vec[-1]))               
                y_df1 = one_stock_data.iloc[-1:,:]              
                y_df = pd.concat([y_df,y_df1],ignore_index=True)               
                slicenum = slicenum+1

    y = np.array(y)
    y_df = y_df.drop(0, axis=0)      
    y_df = y_df.reset_index(level=0, drop=True)

    return X, y, y_df
 


def standardize_input(y_train_df, dataset, X_train, X_val, X_test, timesteps, n):
    dataset['Date'] = dataset['Date'].astype('datetime64[s]')
    y_train_df.Date = y_train_df.Date.astype('datetime64[s]')
    startdate = np.min(y_train_df.Date)

    alldates = np.unique(dataset.Date).astype('datetime64[s]')
    outputindexstart = int(np.where(alldates == startdate)[0])
    onlyinputdata = dataset[(dataset['Date']< startdate) & (dataset['Date'] >= alldates[outputindexstart-timesteps-n+2])] 
    onlyinputdata = onlyinputdata.drop(columns= ['Dividends', 'Stock Splits'])
    trainingdatapoints = pd.concat([onlyinputdata, y_train_df])

    numoffeatures = 5
    colaverage = np.zeros(numoffeatures) 
    colstd = np.zeros(numoffeatures) 

    counter = 0
    for columnindex in ['Open','High','Low','Close','Return']:
      colaverage[counter] = np.average(trainingdatapoints[str(columnindex)])
      colstd[counter] = np.std(trainingdatapoints[str(columnindex)])
      counter = counter + 1

    for colindex in np.arange(0, numoffeatures):
      X_train[:,colindex::5] = X_train[:,colindex::5] - colaverage[colindex]
      X_train[:,colindex::5] = X_train[:,colindex::5]/colstd[colindex]

      X_val[:,colindex::5] = X_val[:,colindex::5] - colaverage[colindex]
      X_val[:,colindex::5] = X_val[:,colindex::5]/colstd[colindex]

      X_test[:,colindex::5] = X_test[:,colindex::5] - colaverage[colindex]
      X_test[:,colindex::5] = X_test[:,colindex::5]/colstd[colindex]

    return X_train, X_val, X_test



def splitdata(X,X_lin,y,y_df, fold_size, foldindex, last_train_date_index):  

    startindex = fold_size*(foldindex-1)
    startdate = np.unique(y_df.Date)[startindex] 
    
    train_chunk_date_index =  last_train_date_index + startindex
    train_chunk_max_date = np.unique(y_df.Date)[train_chunk_date_index]          
    train_indices = y_df.index[(y_df.Date<train_chunk_max_date) & (y_df.Date>=startdate)]

    X_train = X[train_indices,:]
    y_train = y[train_indices]
    y_train_df = y_df.loc[train_indices,:]
    X_train_lin = X_lin[train_indices,:]

    val_date_end_index = train_chunk_date_index + fold_size
    val_chunk_max_date = np.unique(y_df.Date)[val_date_end_index]         
    val_indices = y_df.index[(y_df.Date<val_chunk_max_date) & (y_df.Date>=train_chunk_max_date)]

    X_val = X[val_indices,:]
    y_val = y[val_indices]
    y_val_df = y_df.loc[val_indices,:]

    X_val_lin = X_lin[val_indices,:]

    test_date_end_index = val_date_end_index+fold_size-1  #+ numberofdays%(number_trade_periods + number_of_folds)-1
    test_chunk_max_date = np.unique(y_df.Date)[test_date_end_index]         
    test_indices = y_df.index[(y_df.Date<test_chunk_max_date) & (y_df.Date>=val_chunk_max_date)]

    X_test = X[test_indices,:]
    y_test = y[test_indices]
    y_test_df = y_df.loc[test_indices,:]

    X_test_lin = X_lin[test_indices,:]

    return X_train, X_val, X_test, X_train_lin, X_val_lin, X_test_lin, y_train_df, y_val_df, y_test_df, y_train, y_val, y_test

def create2dy(y1d):

    y2d = np.zeros(shape=(y1d.shape[0],2))
    y2d[:,0] = y1d 
    y2d[y2d[:,0]==0,1] =1

    return y2d





