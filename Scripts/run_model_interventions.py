from initialise_parameters import preparePopulationFrame
from run_model import run_simulation


#Baseline dict for modifications later
# camp
def intialise_control_dict(hygiene_value=0.7,hygiene_timing=[0,0],icu_capacity=6,isolate_value=10
	,isolate_timing=[0,0],shielding=False,highrisk_value=20,highrisk_timing=[0,0],highrisk_cat=2):
	camp = 'Moria'
	population_frame, population = preparePopulationFrame(camp)

	# from github issue
	# if not used, set timings to e.g. [0,0] or any other interval of 0 length or outside caluclated window

	control_dict = dict( # contains our 6 different control options. Can choose any combination of these 6. Suggest limit to all occuring at similar times

		# 1
		# if True, reduces transmission rate by params.better_hygiene
		better_hygiene = dict(value = hygiene_value,
							timing = hygiene_timing),

		ICU_capacity = dict(value = icu_capacity/population),
							
		# 4
		# move symptomatic cases off site
		remove_symptomatic = dict(rate = isolate_value/population,  # people per day
								timing = isolate_timing),

		# 5
		# partially separate low and high risk
		# (for now) assumed that if do this, do for entire course of epidemic
		shielding = dict(used= shielding), 

		# 6
		# move uninfected high risk people off site
		remove_high_risk = dict(rate = highrisk_value/population,  # people per day
								n_categories_removed = highrisk_cat, # remove oldest n categories
								timing = highrisk_timing)

	)
	return camp, population_frame, population, control_dict

def one_simulation_scenarios():
	# #hygiene
	# print('runing simulations for hygiene interventions')
	for hygiene_effective in [0.7,0.8,0.9]:
		camp, population_frame, population, control_dict=intialise_control_dict(hygiene_value=hygiene_effective,hygiene_timing=[0,200])
		run_simulation(camp, population_frame, population, control_dict)
	for hygiene_timing in [30,60,90]:
		camp, population_frame, population, control_dict=intialise_control_dict(hygiene_value=0.7,hygiene_timing=[0,hygiene_timing])
		run_simulation(camp, population_frame, population, control_dict)
	#icu capacity
	print('runing simulations for icu interventions')
	for capacity in [12,24,48]:
		camp, population_frame, population, control_dict=intialise_control_dict(icu_capacity=capacity)
		run_simulation(camp, population_frame, population, control_dict)
	#isolate symptomatic infections (this needs to be better built in to account for the 14 days scenario)
	print('runing simulations for isolating symptomatic infections')
	for isolate_per_day in [10,50,100]:
		if isolate_per_day==10:
			for timing in [100,200]:
				camp, population_frame, population, control_dict=intialise_control_dict(isolate_value=isolate_per_day,isolate_timing=[0,timing])
				run_simulation(camp, population_frame, population, control_dict)
		elif isolate_per_day==50:
			for timing in [20,40,80,120]:
				camp, population_frame, population, control_dict=intialise_control_dict(isolate_value=isolate_per_day,isolate_timing=[0,timing])
				run_simulation(camp, population_frame, population, control_dict)
		elif isolate_per_day==100:
			for timing in [20,30,60]:
				camp, population_frame, population, control_dict=intialise_control_dict(isolate_value=isolate_per_day,isolate_timing=[0,timing])
				run_simulation(camp, population_frame, population, control_dict)
	#shielding
	print('runing simulations for shielding')
	camp, population_frame, population, control_dict=intialise_control_dict(shielding=True)
	run_simulation(camp, population_frame, population, control_dict)
	#remove high risk population
	print('runing simulations for high risk popualtions')
	camp, population_frame, population, control_dict=intialise_control_dict(highrisk_value=20,highrisk_timing=[0,30])
	run_simulation(camp, population_frame, population, control_dict)
	camp, population_frame, population, control_dict=intialise_control_dict(highrisk_value=50,highrisk_timing=[0,12])
	run_simulation(camp, population_frame, population, control_dict)
	camp, population_frame, population, control_dict=intialise_control_dict(highrisk_value=100,highrisk_timing=[0,6])
	run_simulation(camp, population_frame, population, control_dict)
	return None


if __name__=='__main__':
    _=one_simulation_scenarios()


