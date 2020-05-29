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

def load_interventions(folder_path,prefix=None,suffix=None):
	intervention_files=[]
	for file in glob.glob(folder_path+"*.csv"):
		intervention_files.append(file)
	selectedInterventions={}
	for file in intervention_files:
		path=Path(file)
		if prefix is not None:
			if suffix is not None:
				if path.stem.startswith(prefix) and path.stem.endswith(suffix):
					selectedInterventions[path.stem]=read_preprocess_file(file)
			else:
				if path.stem.startswith(prefix):
					selectedInterventions[path.stem]=read_preprocess_file(file)
		elif suffix is not None:
			if path.stem.endswith(suffix):
				selectedInterventions[path.stem]=read_preprocess_file(file)
		else:
			selectedInterventions[path.stem]=read_preprocess_file(file)
	return selectedInterventions