# this is the file that contains the set up for population parameters
import pandas as pd 

raw_data = pd.read_csv('camp_raw_data/moria_params.csv')

def preparePopulationFrame(camp_name):
    
    population_size = raw_data.loc[:,['Variable','Camp','Value']]
    population_size = population_size[population_size.Camp == camp_name]

    population_frame = raw_data[raw_data.Camp==camp_name]
    population_frame = population_frame[population_frame['Variable']=='Population_structure']
    population_frame = population_frame.loc[:,'Age':]


    frac_symptomatic = 0.55 # so e.g. 40% that weren't detected were bc no symptoms and the rest (5%) didn't identify vs e.g. flu
    population_frame = population_frame.assign(p_hospitalised = lambda x: (x.Hosp_given_symptomatic/100)*frac_symptomatic,
                    p_critical = lambda x: (x.Critical_given_hospitalised/100)) 

    population_frame = population_frame.rename(columns={'Value': "Population"})


#     population_size = population_size[population_size['Variable']=='Total_population']
#     population_size = np.float(population_size.Value)
    population_size=20000

    return population_frame, population_size