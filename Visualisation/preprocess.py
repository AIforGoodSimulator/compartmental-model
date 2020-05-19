import pandas as pd

N=18700

def read_preprocess_file(path):
	df=pd.read_csv(path,index_col=0)
	df.R0=df.R0.apply(lambda x: round(complex(x).real,1))
	df_temp=df.drop(['Time', 'R0'], axis=1)
	df_temp=df_temp*N
	df.update(df_temp)
	return df