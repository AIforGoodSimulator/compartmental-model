#modularise the sections that produce tables and plots in the report

import numpy as np
import pandas as pd

N=18700

file_path='./model_outcomes'
#read in the baseline file

def read_preprocess_file(path):
	df=pd.read_csv(path,index_col=0)
	df.R0=df.R0.apply(lambda x: round(complex(x).real,1))
	df_temp=df.drop(['Time', 'R0'], axis=1)
	df_temp=df_temp*N
	df.update(df_temp)
	return df

def incidence_all_table(df):
	#calculate Peak Day IQR and Peak Number IQR for each of the 'incident' variables to plot
	table_params=['Infected (symptomatic)','Hospitalised','Critical','Change in Deaths']
	grouped=df.groupby('R0')
	incident_rs={}
	for index, group in grouped:
		#for each RO value find out the peak days for each table params
		group=group.set_index('Time')
		incident={}
		for param in table_params:
			incident[param]=(group.loc[:,param].idxmax(),group.loc[:,param].max())
		incident_rs[index]=incident
	iqr_table={}
	for param in table_params:
		day=[]
		number=[]
		for elem in incident_rs.values():
			day.append(elem[param][0])
			number.append(elem[param][1])
		q75_day, q25_day = np.percentile(day, [75 ,25])
		q75_number, q25_number = np.percentile(number, [75 ,25])
		iqr_table[param]=((int(round(q25_day)), int(round(q75_day))),(int(round(q25_number)), int(round(q75_number))))
	table_columns={'Infected (symptomatic)':'Incidence of Symptomatic Cases','Hospitalised':'All Hospital Demand',
					'Critical':'Critical Care Demand','Change in Deaths':'Incidence of Deaths' }
	outcome=[]
	peak_day=[]
	peak_number=[]
	for param in table_params:
		outcome.append(table_columns[param])
		peak_day.append(f'{iqr_table[param][0][0]}-{iqr_table[param][0][1]}')
		peak_number.append(f'{iqr_table[param][1][0]}-{iqr_table[param][1][1]}')
	data={'Outcome':outcome,'Peak Day IQR':peak_day,'Peak Number IQR':peak_number}
	incidence_table=pd.DataFrame.from_dict(data)
	return incidence_table

def incidence_age_table(df):
	#calculate age specific Peak Day IQR and Peak Number IQR for each of the 'incident' variables to contruct table
	table_params=['Infected (symptomatic)','Hospitalised','Critical']
	grouped=df.groupby('R0')
	incident_age={}
	params_age=[]
	for index, group in grouped:
		#for each RO value find out the peak days for each table params
		group=group.set_index('Time')
		incident={}
		for param in table_params:
			for column in baseline.columns:
				if column.startswith(param):   
					incident[column]=(group.loc[:,column].idxmax(),group.loc[:,column].max())
					params_age.append(column)
		incident_age[index]=incident
	params_age_dedup=list(set(params_age))
	incident_age_bucket={}
	for elem in incident_age.values():
		for key,value in elem.items():
			if key in incident_age_bucket:
				incident_age_bucket[key].append(value)
			else:
				incident_age_bucket[key]=[value]
	iqr_table_age={}
	for key,value in incident_age_bucket.items():
		day=[x[0] for x in value]
		number=[x[1] for x in value]
		q75_day, q25_day = np.percentile(day, [75 ,25])
		q75_number, q25_number = np.percentile(number, [75 ,25])
		iqr_table_age[key]=((int(round(q25_day)), int(round(q75_day))),(int(round(q25_number)), int(round(q75_number))))
	arrays =[np.array(['Incident Cases', 'Incident Cases', 'Incident Cases', 'Incident Cases', 'Incident Cases', 
						'Incident Cases', 'Incident Cases', 'Incident Cases','Incident Cases','Hospital Demand',
						'Hospital Demand','Hospital Demand','Hospital Demand','Hospital Demand','Hospital Demand',
						'Hospital Demand','Hospital Demand','Hospital Demand','Critical Demand','Critical Demand',
						'Critical Demand','Critical Demand','Critical Demand','Critical Demand','Critical Demand',
						'Critical Demand','Critical Demand']),
			np.array(['all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years', 
						'60-69 years','70+ years','all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', 
						'40-49 years', '50-59 years','60-69 years','70+ years','all ages', '<9 years', '10-19 years', 
						'20-29 years', '30-39 years', '40-49 years', '50-59 years','60-69 years','70+ years'])]
	peak_day=np.empty(27,dtype="S10")
	peak_number=np.empty(27,dtype="S10")
	for key,item in iqr_table_age.items():
		if key=='Infected (symptomatic)':
			peak_day[0]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
			peak_number[0]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif key=='Hospitalised':
			peak_day[9]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
			peak_number[9]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif key=='Critical':
			peak_day[18]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
			peak_number[18]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif '0-9' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[1]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[1]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[10]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[10]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[19]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[19]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif 'Oct-19' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[2]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[2]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[11]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[11]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[20]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[20]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif '20-29' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[3]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[3]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[12]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[12]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[21]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[21]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif '30-39' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[4]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[4]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[13]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[13]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[22]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[22]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif '40-49' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[5]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[5]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[14]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[14]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[23]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[23]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif '50-59' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[6]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[6]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[15]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[15]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[24]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[24]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif '60-69' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[7]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[7]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[16]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[16]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[25]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[25]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
		elif '70-79' in key:
			if key.startswith('Infected (symptomatic)'):
				peak_day[8]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[8]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Hospitalised'):
				peak_day[17]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[17]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
			elif key.startswith('Critical'):
				peak_day[26]=f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
				peak_number[26]=f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
	d = {'Peak Day, IQR': peak_day.astype(str), 'Peak Number, IQR': peak_number.astype(str)}
	incidence_table_age = pd.DataFrame(data=d, index=arrays)
	return incidence_table_age

def cumulative_all_table(df):
	#now we try to calculate the total count
	#cases: (N-exposed)*0.5 since the asymptomatic rate is 0.5
	#hopistal days: cumulative count of hospitalisation bucket
	#critical days: cumulative count of critical days
	#deaths: we already have that from the frame
	table_params=['Susceptible','Hospitalised','Critical','Deaths']
	grouped=df.groupby('R0')
	cumulative_all={}
	for index, group in grouped:
		#for each RO value find out the peak days for each table params
		group=group.set_index('Time')
		cumulative={}
		for param in table_params:
			if param=='Susceptible':
				cumulative[param]=(N-(group[param].tail(1).values[0]))*0.5
			elif param=='Deaths':
				cumulative[param]=(group[param].tail(1).values[0])
			elif param=='Hospitalised' or param=='Critical':
				cumulative[param]=(group[param].sum())
		cumulative_all[index]=cumulative
	cumulative_count=[]
	for param in table_params:
		count=[]
		for elem in cumulative_all.values():
			count.append(elem[param])
		q75_count, q25_count = np.percentile(count, [75 ,25])
		cumulative_count.append(f'{int(round(q25_count))}-{int(round(q75_count))}')
	data={'Totals':['Symptomatic Cases','Hospital Person-Days','Critical Person-days','Deaths'],'Counts':cumulative_count}
	cumulative_table=pd.DataFrame.from_dict(data)
	return cumulative_table
	
#read in the baseline file
baseline=read_preprocess_file(file_path+'/baseline.csv')

