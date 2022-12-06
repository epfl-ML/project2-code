import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helpers

DATA_FOLDER = 'data/'
DATA_FILE1 = '10101.csv'

file1 = DATA_FOLDER + DATA_FILE1
df = pd.read_csv(file1)

k_fold = 4
lambdas = np.logspace(-5, 0, 30)
seed = 13
degrees = [1, 2, 3]

helpers.run_cross_validation(
	df_raw= df, 
	max_invalid_ratio = 0.5, 
	k_fold=k_fold, 
	lambdas=lambdas, 
	degrees=degrees, 
	expand_trig=True, 
	seed=seed, 
	print_best=True
)

# df = helpers.clean_data(df_raw=df, max_invalid_ratio=0.5, degree=2, expand_trig=True)

# print(df)

# helpers.cross_validation(df, k_fold, lambda_=0.01, seed=13)