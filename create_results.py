# -*- coding: utf-8 -*-


from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report

import pandas as pd
import numpy as np


def calculate_performance(dataset):
  foldauc_list = []
  foldbalacc_list = []
  foldacc_list = []
  for fold in np.unique(dataset.foldindex):
    folddata = dataset[dataset.foldindex == fold]
    foldauc = roc_auc_score(folddata.Target, folddata.Prediction)
    foldauc_list.append(foldauc)
    foldbalacc = balanced_accuracy_score(folddata.Target, np.round(folddata.Prediction))
    foldbalacc_list.append(foldbalacc)
    foldacc = accuracy_score(folddata.Target, np.round(folddata.Prediction))
    foldacc_list.append(foldacc)
  
  auc = np.mean(foldauc_list)
  balacc = np.mean(foldbalacc_list)
  acc = np.mean(foldacc_list)

  return auc, balacc, acc

def createresults(arch, approach, forecastdays, toppercentile):

    probopt = 1
    test_df_dir = arch +'/' + arch+ 'test_df_'  + str(approach)+'_d_'+str(forecastdays) +'.csv'    
    val_df_dir = arch +'/'  + arch+ 'val_df_' + str(approach)+'_d_'+str(forecastdays) +'.csv'

    val_df = pd.read_csv(val_df_dir) 
    test_df = pd.read_csv(test_df_dir)

    upthreshold = []
    for fold in np.unique(val_df.foldindex):
      folddata = val_df[val_df.foldindex == fold]
      foldpredarray = np.asarray(folddata.Prediction)
      upthreshold.append(np.percentile(foldpredarray, 100-toppercentile))
    higherthreshold = np.min(upthreshold)
      
    downthreshold = []
    for fold in np.unique(val_df.foldindex):
      folddata = val_df[val_df.foldindex == fold]
      foldpredarray = np.asarray(folddata.Prediction)
      downval = np.percentile(foldpredarray, toppercentile)
      downthreshold.append(downval)
      
    downthreshold=np.asarray(downthreshold)
    lowerthreshold = np.max(downthreshold[downthreshold<0.5])

    val_df_subset = val_df[(val_df.Prediction>=higherthreshold)|(val_df.Prediction<=lowerthreshold)]
    test_df_subset = test_df[(test_df.Prediction>=higherthreshold)|(test_df.Prediction<=lowerthreshold)]

    zeroarray = np.zeros(1)
    performance_dict = {'model':np.array(['Model']), 'valacc':zeroarray, 'valbalacc':zeroarray,'valAUC':zeroarray, 'testacc':zeroarray, 'testbalacc':zeroarray, 'testAUC':zeroarray}
    performance_df = pd.DataFrame(performance_dict, columns = ['model','valacc','valbalacc','valAUC', 'testacc','testbalacc', 'testAUC'], index = [0])

    valauc, valbalacc, valacc =  calculate_performance(val_df_subset)
    testauc, testbalacc, testacc =  calculate_performance(test_df_subset)

    performance_df['model'] = arch
    performance_df['valacc'] = valacc
    performance_df['valbalacc'] = valbalacc
    performance_df['valAUC'] = valauc
    performance_df['testacc'] = testacc
    performance_df['testbalacc'] = testbalacc
    performance_df['testAUC'] = testauc

    performancedir = arch+ '/' +arch+'performance_' + str(approach) +".csv"
    with open(performancedir, 'w') as csv_file:
        performance_df.to_csv(path_or_buf=csv_file,  index=False)

    zeroarray = np.zeros(1)
    finperformance_dict= {'model':np.array(['Model']), 'returnlong':zeroarray, 'returnshort':zeroarray,'return':zeroarray, 'min':zeroarray, 'max':zeroarray,  'cumulreturnlong':zeroarray, 'cumulreturnshort':zeroarray}
    finperformance_df = pd.DataFrame(finperformance_dict, columns = ['model','returnlong','returnshort','return', 'min', 'max','cumulreturnlong','cumulreturnshort'], index = [0])

    finperformance_df['model'] = arch
    finperformance_df['returnlong'] = np.average(test_df_subset[test_df_subset.Prediction>0.5].Return)
    finperformance_df['returnshort'] = -np.average(test_df_subset[test_df_subset.Prediction<0.5].Return)
    finperformance_df['return'] = finperformance_df['returnshort'] + finperformance_df['returnlong']
    
    daylist = []
    returnlist = []
    cumulreturn = []
    for tradeday in np.unique(test_df_subset.Date):
      dayset = test_df_subset[test_df_subset.Date ==tradeday]
      posdayset = dayset[dayset.Prediction >0.5]
      
      if posdayset.shape[0]>0:
        daylist.append(tradeday)
        returnlist.append(1+np.average(posdayset.Return)-0.001)
        cumulreturn.append(np.product(returnlist))

    finperformance_df['cumulreturnlong'] = np.product(returnlist)

    longmin = np.min(returnlist)
    longmax = np.max(returnlist)

    daylist = []
    returnlist = []
    cumulreturn = []
    for tradeday in np.unique(test_df_subset.Date):
      dayset = test_df_subset[test_df_subset.Date ==tradeday]
      posdayset = dayset[dayset.Prediction <0.5]
      
      if posdayset.shape[0]>0:
        daylist.append(tradeday)
        returnlist.append(1-np.average(posdayset.Return)-0.001)
        cumulreturn.append(np.product(returnlist))

    finperformance_df['cumulreturnshort'] = np.product(returnlist)

    shortmin = np.min(returnlist)
    shortmax = np.max(returnlist)

    finperformance_df['min'] = -(1-np.min([longmin, shortmin]))
    finperformance_df['max'] = np.max([longmax, shortmax])-1

    financialperformancedir = arch +'/' +arch+ 'financial_performance_' + str(approach) +".csv"
    with open(financialperformancedir, 'w') as csv_file:
        finperformance_df.to_csv(path_or_buf=csv_file,  index=False)

    return performance_df, finperformance_df


def createpredictionmatrices(arch, approach, forecastdays):

    allweights_dir = arch +'/' + arch +'allweights_' +str(approach)+'_d_'+str(forecastdays) +'.csv'
    allweightmatrix =  np.loadtxt(allweights_dir, delimiter=",")

    bullishmatrix = np.zeros((20,5))
    bearishmatrix = np.zeros((20,5))

    for i in range(0,5):
      bullishmatrix[:,i] = np.exp(allweightmatrix[:,2*i])/(np.exp(allweightmatrix[:,2*i])+np.exp(allweightmatrix[:,2*i+1]))

    for i in range(0,5):
      bearishmatrix[:,i] = np.exp(-allweightmatrix[:,2*i])/(np.exp(-allweightmatrix[:,2*i])+np.exp(-allweightmatrix[:,2*i+1]))

    bullishmatrix[8,:] = 0

    bearishmatrix[[4, 7, 13, 16, 17],:] = 0

    print(bullishmatrix)
    print('')
    print(bearishmatrix)


    bullish_dir = arch +'/' + arch +'bullish_interpretation' +str(approach)+'_d_'+str(forecastdays) +'.csv'
    np.savetxt(bullish_dir, bullishmatrix)

    bearish_dir = arch +'/' + arch +'bearish_interpretations' +str(approach)+'_d_'+str(forecastdays) +'.csv'
    np.savetxt(bearish_dir, bearishmatrix)

    return bullishmatrix, bearishmatrix
