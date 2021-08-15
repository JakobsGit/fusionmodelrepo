## Master's Thesis Code Repository


**get_data_functions.py**

* contains function that  gathers S&P 500 stock data with highest trade volume from yahoo finance

**preprocessing_data_helpers.py**

contains functions that preprocess the data 
	* remove "N/A"
	* standardize data
	* feature and target generation
	* split data into training, validation, test sets

**create_results.py**

contains functions that produce results by
	* calculating predicitive accuracy (accuracy, balanced accuracy, AUC-score) on validation and test set
	* creating prediction matrices to interpret the predictions of the linear term


**create_keras_models.py**

contains function that create neural network model in keras with tensorflow backend
	* simple LSTM
	* simple GRU
	* fusion LSTM
	* fusion GRU
	* fusion ANN

**X_array_part1.npy**
**X_array_part2.npy**

contains X matrix with series of daily opening, closing, high and low prices as well as daily returns of the 20 days for model training.
Matrix is split into two parts due to the file size

**X_lin.npy**

contains a matrix that indicates the occurrence of a list of 20 candlestick pattern

**dataset_50SP500_stocks.zip**

contains all raw data that is utilized

**df_with_features.zip**

contains the pre-processed raw data with features and targets



**main.ipynb**

contains code blocks for 

    * Installing libraries & loading functions and data from repository
    * Downloading and transforming stock data into input data
    * Loading preprocessed input data and creating candlestick pattern feature vectors
    * Hyper-parameter optimization for neural network models
    * Grid-search for random forest model
    * Grid-search for ridge/LASSO regression

**Hyper-parameter optimization for neural network models**

* contains functions that run "cross validation for time series"
* Bayesian optimization to find the best hyper parameters (learning rate, number of nodes, batch size, dropout rate) for a neural network models (based on validation loss on 5 folds)	
* stores the results of the Bayesian optimization
* stores the best parameters for predicting the target one day ahead
* uses the best parameters to predict targets of the test sets and stores the results in a data frame containing the date, return, and price information

Models: 
features: opening, closing, high, low prices, return 
additional features of fusion model: candlestick pattern occurance
target: positive return -> 1, negative or zero return -> 0

models can be chosen by arch variable 


**Grid-search for random forest model**
	
* contains functions that run "cross validation for time series"
* Bayesian optimization to find the best hyper parameters (number of trees, max depth) for a random forest model (based on average validation loss)
* stores the results of the Bayesian optimization
* stores the best parameters for predicting the target in 20,10,5,3,2,1 trading days 
* uses the best parameters to predict targets of the test sets and stores the results in a data frame containing the date, return, and price information

Model: 
features: open, close, high, low prices, return
target: positive return -> 1, negative or zero return -> 0


**Grid-search for ridge/LASSO regression**

* contains 2 different models: than can be chosen by the arch variable: "lasso" or "ridge"
* contains functions that run "cross validation for time series"
* grid search to find the best hyper parameters (lambda) for a ridge/lasso regression	
* stores the results of the grid search
* stores the best parameters for predicting the target one day ahead
* uses the best parameters to predict targets of the test sets and stores the results in a data frame containing the date, return, and price information
	
Model: 
features: open, close, high, low prices, return
target: positive return -> 1, negative or zero return -> 0


**result folder**

contains the results of the hyper parameter optimization of each model

	* fusion ANN model
	* fusion GRU model
	* fusion LSTM model
	* LASSO regression
	* random forest
	* ridge
	* simple GRU network
	* simple LSTM network



