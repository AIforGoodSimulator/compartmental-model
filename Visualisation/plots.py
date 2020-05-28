#for plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from ipywidgets import fixed,interactive,Layout
from preprocess import read_preprocess_file,load_interventions
import ipywidgets as widgets

# def plot_by_age(column,df):
# 	fig, ax = plt.subplots(1, 9, sharex='col', sharey='row',figsize=(20,5),constrained_layout=True)
# 	for key in df.columns:
# 		if key==column:
# 			sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[0])
# 			ax[0].set_title('all ages')
# 		elif '0-9' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[1])
# 				ax[1].set_title('<9 years')
# 		elif 'Oct-19' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[2])
# 				ax[2].set_title('10-19 years')
# 		elif '20-29' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[3])
# 				ax[3].set_title('20-29 years')
# 		elif '30-39' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[4])
# 				ax[4].set_title('30-39 years')
# 		elif '40-49' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[5])
# 				ax[5].set_title('40-49 years')
# 		elif '50-59' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[6])
# 				ax[6].set_title('50-59 years')
# 		elif '60-69' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[7])
# 				ax[7].set_title('60-69 years')
# 		elif '70+' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[8])
# 				ax[8].set_title('70+ years')

# def plot_by_age_interactive(plot_by_age,df):
# 	w = interactive(plot_by_age,column=widgets.Dropdown(
# 				options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],
# 				value='Infected (symptomatic)',
# 				description='Category:'
# 				),df=fixed(df))
# 	words = widgets.Label('Plot the do nothing scenario in four different categories split by age groups')
# 	container=widgets.VBox([words,w])
# 	container.layout.width = '100%'
# 	container.layout.border = '2px solid grey'
# 	container.layout.justify_content = 'space-around'
# 	container.layout.align_items = 'center'
# 	return container

def plot_by_age_all(df):
	fig, ax = plt.subplots(4, 9, sharex='col', sharey='row',figsize=(20,20),constrained_layout=True)
	columns_to_plot=['Infected (symptomatic)','Hospitalised','Critical','Deaths']
	i=0
	fontdict={'fontsize':15}
	for column in columns_to_plot:
		for key in df.columns:
			if key==column:
				sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,0])
				ax[i,0].set_title('all ages',fontdict)
			elif '0-9' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,1])
					ax[i,1].set_title('<9 years',fontdict)
			elif 'Oct-19' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,2])
					ax[i,2].set_title('10-19 years',fontdict)
			elif '20-29' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,3])
					ax[i,3].set_title('20-29 years',fontdict)
			elif '30-39' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,4])
					ax[i,4].set_title('30-39 years',fontdict)
			elif '40-49' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,5])
					ax[i,5].set_title('40-49 years',fontdict)
			elif '50-59' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,6])
					ax[i,6].set_title('50-59 years',fontdict)
			elif '60-69' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,7])
					ax[i,7].set_title('60-69 years',fontdict)
			elif '70+' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci="sd",data=df,ax=ax[i,8])
					ax[i,8].set_title('70+ years',fontdict)
		i+=1
	fig.suptitle('Fig.1 Plots of changes in symptomatically infected cases, hopitalisation cases, critical care cases and death incidents over the course of simulation datys',
		fontsize=20)


def plot_one_intervention_horizontal(column,baseline,one_intervention_dict):
	fig, ax = plt.subplots(1, len(one_intervention_dict)+1, sharex='col', sharey='row',figsize=(4*len(one_intervention_dict),5))
	sns.lineplot(x="Time", y=column,ci="sd",data=baseline,ax=ax[0])
	ax[0].set_title('Baseline')
	i=1
	for key,value in one_intervention_dict.items():
		sns.lineplot(x="Time", y=column,ci="sd",data=value,ax=ax[i])
		ax[i].set_title(key)
		i+=1

def plot_one_intervention_horizontal_interactive(plot_one_intervention_horizontal,baseline):
	folder_path='./model_outcomes/one_intervention/'
	one_intervention_dict=load_interventions(folder_path)
	#sort the collection of interventions by their keys
	one_intervention_dict={k: v for k, v in sorted(one_intervention_dict.items(), key=lambda item: item[0])}
	w = interactive(plot_one_intervention_horizontal,
					column=widgets.Select(
					options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],
					value='Infected (symptomatic)',
					description='Category:'
					),
					baseline=fixed(baseline),
					one_intervention_dict=fixed(one_intervention_dict))
	words = widgets.Label('Contrast the peak case count between do nothing and a single intervention strategy in place')
	container=widgets.VBox([words,w])
	container.layout.width = '100%'
	container.layout.border = '2px solid grey'
	container.layout.justify_content = 'space-around'
	container.layout.align_items = 'center'
	return container

def plot_one_intervention_vertical(column,one_intervention_dict):
	peak_values={}
	for key,value in one_intervention_dict.items():
		peak_values[key]=value.groupby('R0')[column].max().mean()
	peak_values_sorted={k: v for k, v in sorted(peak_values.items(), key=lambda item: item[1],reverse=True)}
	fig, ax = plt.subplots(len(one_intervention_dict), 1, sharex='row', sharey=True,figsize=(15,20))
	i=0
	for key in peak_values_sorted.keys():
		sns.lineplot(x="Time", y=column,ci="sd",data=one_intervention_dict[key],ax=ax[i])
		ax[i].text(0.5,0.5,key,verticalalignment='center',horizontalalignment='center',fontsize=15,color='green',transform=ax[i].transAxes)
		ax[i].set_ylabel('')    
		i+=1

def plot_one_intervention_vertical_interactive(plot_one_intervention_vertical):
	folder_path='./model_outcomes/one_intervention/'
	one_intervention_dict=load_interventions(folder_path)
	w = interactive(plot_one_intervention_vertical,
					column=widgets.Select(
					options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],
					value='Infected (symptomatic)',
					description='Category:'
					),
					one_intervention_dict=fixed(one_intervention_dict))
	words = widgets.Label('Plot the case counts when one of the intervention is in place and the intervention plots are places by ascending order of effectiveness in reducing peak case counts')
	container=widgets.VBox([words,w])
	container.layout.width = '100%'
	container.layout.border = '2px solid grey'
	container.layout.justify_content = 'space-around'
	container.layout.align_items = 'center'
	return w

def plot_intervention_comparison(scenarioDict,firstIntervention,secondIntervention,selectedCategory):
	fig, ax = plt.subplots(1, 2, sharex='col', sharey='row',figsize=(25,5))
	sns.lineplot(x="Time", y=selectedCategory,ci="sd",data=scenarioDict[firstIntervention],ax=ax[0])
	ax[0].set_title(firstIntervention)
	sns.lineplot(x="Time", y=selectedCategory,ci="sd",data=scenarioDict[secondIntervention],ax=ax[1])
	ax[1].set_title(secondIntervention)

def plot_intervention_comparison_interactive(plot_intervention_comparison,baseline):
	folder_path='./model_outcomes/one_intervention/'
	selectedInterventions=load_interventions(folder_path)
	selectedInterventions['do nothing']=baseline
	first = widgets.Dropdown(options=selectedInterventions.keys(),value='do nothing',description='Compare:',disabled=False)
	second = widgets.Dropdown(options=selectedInterventions.keys(),value='shielding',description='With:',disabled=False)
	category = widgets.Dropdown(options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],description='Category:',disabled=False)
	w = interactive(plot_intervention_comparison,scenarioDict=fixed(selectedInterventions),firstIntervention=first,secondIntervention=second,
				selectedCategory=category)
	controls = widgets.HBox(w.children[:-1], layout = Layout(flex_flow='row wrap'))
	output = w.children[-1]
	words = widgets.Label('Compare two different intervention strategies or compare a particular intervention strategy with do nothing scenario')
	container=widgets.VBox([words,controls,output])
	container.layout.width = '100%'
	container.layout.border = '2px solid grey'
	container.layout.justify_content = 'space-around'
	container.layout.align_items = 'center'
	return container


