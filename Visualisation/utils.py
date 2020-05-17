#modularise the sections that produce tables and plots in the report

import numpy as np
import pandas as pd

N=18700

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
			for column in df.columns:
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

def find_first_month(df):
	return df[df['Time']==30]
def find_third_month(df):
	return df[df['Time']==90]
def find_sixth_month(df):
	return df[df['Time']==180]
def find_first_month_diff(df):
	return df[df['Time']<=30].diff(periods=30).tail(1)
def find_third_month_diff(df):
	return df[df['Time']<=90].diff(periods=90).tail(1)
def find_sixth_month_diff(df):
	return df[df['Time']<=180].diff(periods=180).tail(1)
def find_one_month(df):
	return df[df['Time']<=30].cumsum().tail(1)
def find_three_months(df):
	return df[df['Time']<=90].cumsum().tail(1)
def find_six_months(df):
	return df[df['Time']<=180].cumsum().tail(1)
def Merge(dict1, dict2): 
	res = {**dict1, **dict2} 
	return res 

def cumulative_age_table(df):
	#need to have an age break down for this as well
	#1 month 3 month and 6 month breakdown
	arrays =[np.array(['Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases', 
						'Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases','Symptomatic Cases','Hospital Person-Days',
						'Hospital Person-Days','Hospital Person-Days','Hospital Person-Days','Hospital Person-Days','Hospital Person-Days',
						'Hospital Person-Days','Hospital Person-Days','Hospital Person-Days','Critical Person-days','Critical Person-days',
						'Critical Person-days','Critical Person-days','Critical Person-days','Critical Person-days','Critical Person-days',
						'Critical Person-days','Critical Person-days','Deaths','Deaths','Deaths','Deaths','Deaths','Deaths','Deaths','Deaths',
						'Deaths']),
			 np.array(['all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years', 
						'60-69 years','70+ years','all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', 
						'40-49 years', '50-59 years','60-69 years','70+ years','all ages', '<9 years', '10-19 years', 
						'20-29 years', '30-39 years', '40-49 years', '50-59 years','60-69 years','70+ years','all ages', 
						'<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years','60-69 years',
						'70+ years'])]
	table_params=['Susceptible','Hospitalised','Critical','Deaths']
	params_select=['Susceptible','Deaths']
	params_accu=['Hospitalised','Critical']
	columns_to_select=[]
	columns_to_acc=[]
	for column in df.columns:
		for param in params_select:
			if column.startswith(param):
				columns_to_select.append(column)
		for param in params_accu:
			if column.startswith(param):
				columns_to_acc.append(column)
	first_month_select={}
	three_month_select={}
	six_month_select={}

	for column in columns_to_select:
		if 'Susceptible' in column:
			first_month_select[column]=df.groupby('R0')[[column,'Time']].apply(find_first_month_diff)[column].mul(-0.5).quantile([.25, .75])
			three_month_select[column]=df.groupby('R0')[[column,'Time']].apply(find_third_month_diff)[column].mul(-0.5).quantile([.25, .75])
			six_month_select[column]=df.groupby('R0')[[column,'Time']].apply(find_sixth_month_diff)[column].mul(-0.5).quantile([.25, .75])
		else:
			first_month_select[column]=df.groupby('R0')[[column,'Time']].apply(find_first_month)[column].quantile([.25, .75])
			three_month_select[column]=df.groupby('R0')[[column,'Time']].apply(find_third_month)[column].quantile([.25, .75])
			six_month_select[column]=df.groupby('R0')[[column,'Time']].apply(find_sixth_month)[column].quantile([.25, .75])
	first_month_accu={}
	three_month_accu={}
	six_month_accu={}
	for column in columns_to_acc:
		first_month_accu[column]=df.groupby('R0')[[column,'Time']].apply(find_one_month)[column].quantile([.25, .75])
		three_month_accu[column]=df.groupby('R0')[[column,'Time']].apply(find_three_months)[column].quantile([.25, .75])
		six_month_accu[column]=df.groupby('R0')[[column,'Time']].apply(find_six_months)[column].quantile([.25, .75])
	first_month = Merge(first_month_select, first_month_accu) 
	third_month = Merge(three_month_select, three_month_accu) 
	sixth_month = Merge(six_month_select, six_month_accu) 
	first_month_count=np.empty(36,dtype="S15")
	for key,item in first_month.items():
		if key=='Susceptible':
			first_month_count[0]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif key=='Hospitalised':
			first_month_count[9]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif key=='Critical':
			first_month_count[18]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif key=='Deaths':
			first_month_count[27]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif '0-9' in key:
			if key.startswith('Susceptible'):
				first_month_count[1]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[10]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[19]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[28]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif 'Oct-19' in key:
			if key.startswith('Susceptible'):
				first_month_count[2]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[11]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[20]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[29]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif '20-29' in key:
			if key.startswith('Susceptible'):
				first_month_count[3]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[12]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[21]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[30]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif '30-39' in key:
			if key.startswith('Susceptible'):
				first_month_count[4]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[13]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[22]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[31]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif '40-49' in key:
			if key.startswith('Susceptible'):
				first_month_count[5]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[14]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[23]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[32]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif '50-59' in key:
			if key.startswith('Susceptible'):
				first_month_count[6]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[15]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[24]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[33]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif '60-69' in key:
			if key.startswith('Susceptible'):
				first_month_count[7]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[16]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[25]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[34]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
		elif '70-79' in key:
			if key.startswith('Susceptible'):
				first_month_count[8]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				first_month_count[17]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Critical'):
				first_month_count[26]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				first_month_count[35]=f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
	three_month_count=np.empty(36,dtype="S15")
	for key,item in third_month.items():
		if key=='Susceptible':
			three_month_count[0]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif key=='Hospitalised':
			three_month_count[9]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif key=='Critical':
			three_month_count[18]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif key=='Deaths':
			three_month_count[27]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif '0-9' in key:
			if key.startswith('Susceptible'):
				three_month_count[1]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[10]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[19]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[28]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif 'Oct-19' in key:
			if key.startswith('Susceptible'):
				three_month_count[2]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[11]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[20]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[29]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif '20-29' in key:
			if key.startswith('Susceptible'):
				three_month_count[3]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[12]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[21]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[30]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif '30-39' in key:
			if key.startswith('Susceptible'):
				three_month_count[4]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[13]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[22]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[31]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif '40-49' in key:
			if key.startswith('Susceptible'):
				three_month_count[5]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[14]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[23]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[32]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif '50-59' in key:
			if key.startswith('Susceptible'):
				three_month_count[6]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[15]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[24]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[33]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif '60-69' in key:
			if key.startswith('Susceptible'):
				three_month_count[7]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[16]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[25]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[34]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
		elif '70-79' in key:
			if key.startswith('Susceptible'):
				three_month_count[8]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				three_month_count[17]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Critical'):
				three_month_count[26]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				three_month_count[35]=f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
	six_month_count=np.empty(36,dtype="S10")
	for key,item in sixth_month.items():
		if key=='Susceptible':
			six_month_count[0]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif key=='Hospitalised':
			six_month_count[9]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif key=='Critical':
			six_month_count[18]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif key=='Deaths':
			six_month_count[27]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif '0-9' in key:
			if key.startswith('Susceptible'):
				six_month_count[1]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[10]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[19]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[28]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif 'Oct-19' in key:
			if key.startswith('Susceptible'):
				six_month_count[2]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[11]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[20]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[29]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif '20-29' in key:
			if key.startswith('Susceptible'):
				six_month_count[3]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[12]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[21]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[30]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif '30-39' in key:
			if key.startswith('Susceptible'):
				six_month_count[4]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[13]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[22]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[31]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif '40-49' in key:
			if key.startswith('Susceptible'):
				six_month_count[5]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[14]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[23]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[32]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif '50-59' in key:
			if key.startswith('Susceptible'):
				six_month_count[6]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[15]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[24]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[33]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif '60-69' in key:
			if key.startswith('Susceptible'):
				six_month_count[7]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[16]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[25]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[34]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
		elif '70-79' in key:
			if key.startswith('Susceptible'):
				six_month_count[8]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Hospitalised'):
				six_month_count[17]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Critical'):
				six_month_count[26]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
			elif key.startswith('Deaths'):
				six_month_count[35]=f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
	d = {'First month': first_month_count.astype(str), 'First three months': three_month_count.astype(str),
	'First six months': six_month_count.astype(str)}
	count_table_age = pd.DataFrame(data=d, index=arrays)
	return count_table_age

			
#read in the baseline file
# baseline=read_preprocess_file(file_path+'/baseline.csv')

