"""
This file sets up the parameters for SEIR models used in the cov_functions_AI.py
"""

from math import log, exp
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from math import ceil

def preparePopulationFrame(camp_name):

    raw_data = pd.read_csv('AI_for_good/AI-for-good/camp_params.csv')
    
    population_size = raw_data.loc[:,['Variable',camp_name]]

    covid_data = raw_data.loc[:,'Variable':'Critical_given_hospitalised']
    population_frame = pd.concat([covid_data,raw_data.loc[:,camp_name]],axis=1)
    population_frame = population_frame[population_frame['Variable']=='Population_structure']
    population_frame = population_frame.loc[:,'Age':]


    frac_symptomatic = 0.55 # so e.g. 40% that weren't detected were bc no symptoms and the rest (5%) didn't identify vs e.g. flu
    population_frame = population_frame.assign(p_hospitalised = lambda x: (x.Hosp_given_symptomatic/100)*frac_symptomatic,
                    p_critical = lambda x: (x.Critical_given_hospitalised/100)) 

    population_frame = population_frame.rename(columns={camp_name: "Population"})


    population_size = population_size[population_size['Variable']=='Total_population']
    population_size = np.float(population_size.iloc[-1,-1])

    return population_frame, population_size



example_population_frame, example_population = preparePopulationFrame('Camp_1')


# print(population_frame)
# print(population)
#------------------------------------------------------------
# disease params
N    = 1
non_infectious_rate = 1/7
R_0        = 4.0
beta = R_0*non_infectious_rate/N # R_0 mu/N
infection_matrix = beta*np.ones((example_population_frame.shape[0],example_population_frame.shape[0]))
infectious_rate = 1/2
hosp_rate = 1/8
death_rate = 1/8
death_rate_noICU  = 1/2
death_prob     = 0.5
number_compartments = 7
fact_v = np.concatenate([[0.02,0.1],np.linspace(0.20,1,9)])
max_months_controlling = 18


# Parameters that are going to come into play later:
# ICU_growth = 1
# ICU_capacity = 8/100000
# import_rate = 1/(30*population) # 1 per month
# vaccinate_percent = 0 # 0.9 # vaccinate this many
# vaccinate_rate = 0.55/(365*2/3) #10000/UK_population # per day
# https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0006158&type=printable


class Parameters:
    def __init__(self):
        
        # self.population = population
        # self.dataframe = example_population_frame

        self.R_0 = R_0
        self.N  = N
        self.non_infectious_rate = non_infectious_rate
        beta = R_0*non_infectious_rate/N # R_0 mu/N
        self.infection_matrix = beta*np.ones((example_population_frame.shape[0],example_population_frame.shape[0]))
        self.fact_v = fact_v
        self.max_months_controlling = max_months_controlling

        self.number_compartments = number_compartments

        self.infectious_rate  = infectious_rate
        self.non_infectious_rate = non_infectious_rate
        self.hosp_rate = hosp_rate        
        self.death_rate = death_rate
        self.death_rate_noICU = death_rate_noICU
        self.death_prob     = death_prob

        self.S_ind = 0
        self.E_ind = 1
        self.I_ind = 2
        self.R_ind = 3
        self.H_ind = 4
        self.C_ind = 5
        self.D_ind = 6





params = Parameters()

