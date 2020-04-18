"""
This file sets up the parameters for SEIR models used in the cov_functions_AI.py
"""

from math import log, exp
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from math import ceil
import os
cwd = os.getcwd()

raw_data = pd.read_csv(os.path.join(cwd,'camp_params.csv'))
# raw_data = pd.read_csv(os.path.join(cwd,'camp_raw_data/moria_params.csv'))

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


    population_size = population_size[population_size['Variable']=='Total_population']
    population_size = np.float(population_size.Value)


    return population_frame, population_size



# example_population_frame, example_population = preparePopulationFrame('Moria')
example_population_frame, example_population = preparePopulationFrame('Camp_1')
# print(example_population_frame)


#------------------------------------------------------------
# disease params
parameter_csv = pd.read_csv(os.path.join(cwd,'parameters.csv'))
model_params = parameter_csv[parameter_csv['Type']=='Model Parameter']
model_params = model_params.loc[:,['Name','Value']]
# print()

R_0_list                         =   np.asarray(model_params[model_params['Name']=='R0'].Value)


latent_rate      = 1/(np.float(model_params[model_params['Name']=='latent_period'].Value))
removal_rate   = 1/(np.float(model_params[model_params['Name']=='removal_period'].Value))
hosp_rate                   = 1/(np.float(model_params[model_params['Name']=='hosp_period'].Value))
death_rate                  = 1/(np.float(model_params[model_params['Name']=='death_period'].Value))

death_prob          = np.float(model_params[model_params['Name']=='death_prob'].Value)
number_compartments = int(model_params[model_params['Name']=='number_compartments'].Value)

beta_list           = [R_0*removal_rate  for R_0 in R_0_list] # R_0 mu/N, N=1

# infection_matrix    = np.ones((example_population_frame.shape[0],example_population_frame.shape[0]))


# Parameters that may come into play later:
# ICU_capacity = 8/100000
# death prob and period given ICU care:
# import_rate = 1/(30*population) # 1 per month


class Parameters:
    def __init__(self):
        
        self.R_0_list = R_0_list
        self.beta_list = beta_list
        self.removal_rate = removal_rate
        # self.infection_matrix = infection_matrix


        self.number_compartments = number_compartments

        self.latent_rate  = latent_rate
        self.hosp_rate = hosp_rate        
        self.death_rate = death_rate
        self.death_prob     = death_prob

        self.S_ind = 0
        self.E_ind = 1
        self.I_ind = 2
        self.R_ind = 3
        self.H_ind = 4
        self.C_ind = 5
        self.D_ind = 6





params = Parameters()

