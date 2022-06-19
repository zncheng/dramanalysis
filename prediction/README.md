# Prediction Source Code
The source code implement our proposed server failure prediction workflow, including the source code to generate features for prediction and the source code of time-series cross-validate testing.

## Prerequisite
Python 3: please install [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [joblib](https://joblib.readthedocs.io/en/latest/), [sklearn](https://scikit-learn.org/stable/), [imblearn](https://imbalanced-learn.org/stable/), and [lightgbm](https://lightgbm.readthedocs.io/en/latest/) 

## Usage
+ Assume the data is stored under `../data/`
+ Feature generation
	+ `python3 feature_generation.py` 
	+ The generated features will be stored in `./features/`
		+ totally 32 files will be generated
		+ file name in format `features_$freq_month_$i.csv`
			+ where $freq indicating the prediction intervals, including 5 minutes, 30 minutes, 1 hour, and 1 day.
			+ where $i indicates the month, from 1 - 8 for eight different months
	+ Note that it can take a long time (few hours) for generating features for each month
	+ By default we set the parallelism as 64 since our server used for feature generation has 64 cores, users can also increase or decrease the number of cores for training based on their own hardwares
		+ See the usage of function `apply_parallel_group` 
+ Training-testing 
	+ `python3 prediction.py` 
+ The results will be stored under `./result/`
	+ including the results for four different experiments 

## Functions for Training-testing experiments
+ `exp1_main`: output the precision, recall, and F1-socre when we use different groups of features for predictions in Finding 11
	+ Results: `exp1.csv` 
+ `exp2_main`: output the precision, recall, and F1-socre when we use different prediction models in Finding 12
	+ Results: `exp2.csv` 
+ `exp3_main`: output the precision, recall, and F1-socre when we use different prediction intervals in Finding 13
	+ Results: `exp3.csv` 
+ `exp4_main`: output the number of predicted server failures that are predicted within 5 minutes and at least 5 minutes before predictions in Finding 14
	+ Results: `exp4.csv` 

## Contact
Please email to Zhinan Cheng (zncheng@cse.cuhk.edu.hk) if you have any questions.
