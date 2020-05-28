import pandas as pd
import glob
from pathlib import Path

N=18700

def read_preprocess_file(path):
	df=pd.read_csv(path,index_col=0)
	df.R0=df.R0.apply(lambda x: round(complex(x).real,1))
	df_temp=df.drop(['Time','R0','latentRate','removalRate','hospRate','deathRateICU','deathRateNoIcu'], axis=1)
	df_temp=df_temp*N
	df.update(df_temp)
	return df

def load_interventions(folder_path):
	intervention_files=[]
	for file in glob.glob(folder_path+"*.csv"):
		intervention_files.append(file)
	selectedInterventions={}
	for file in intervention_files:
		path=Path(file)
		selectedInterventions[path.stem]=read_preprocess_file(file)
	return selectedInterventions