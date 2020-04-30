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
# raw_data = pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/camp_params.csv'))
raw_data = pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/moria_params.csv'))

def preparePopulationFrame(camp_name):
    
    population_size = raw_data.loc[:,['Variable','Camp','Value']]
    population_size = population_size[population_size.Camp == camp_name]

    population_frame = raw_data[raw_data.Camp==camp_name]
    population_frame = population_frame[population_frame['Variable']=='Population_structure']
    population_frame = population_frame.loc[:,'Age':]


    # 0.55 # so e.g. 40% that weren't detected were bc no symptoms and the rest (5%) didn't identify vs e.g. flu
    # frac_symptomatic = 1 # now only multiplying by symptomatic category so don't need to do this!
    population_frame = population_frame.assign(p_hospitalised = lambda x: (x.Hosp_given_symptomatic/100), # *frac_symptomatic,
                    p_critical = lambda x: (x.Critical_given_hospitalised/100)) 

    population_frame = population_frame.rename(columns={'Value': "Population"})


    population_size = population_size[population_size['Variable']=='Total_population']
    population_size = np.float(population_size.Value)


    return population_frame, population_size



# example_population_frame, example_population = preparePopulationFrame('Moria')
# example_population_frame, example_population = preparePopulationFrame('Camp_1')
# print(example_population_frame)


#------------------------------------------------------------
# disease params
parameter_csv = pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/parameters.csv'))
model_params = parameter_csv[parameter_csv['Type']=='Model Parameter']
model_params = model_params.loc[:,['Name','Value']]
control_data = parameter_csv[parameter_csv['Type']=='Control']

# print()

R_0_list                         =   np.asarray(model_params[model_params['Name']=='R0'].Value)

asympt_prop = np.float(model_params[model_params['Name']=='asymptomatic proportion'].Value)
latent_rate    = 1/(np.float(model_params[model_params['Name']=='latent period'].Value))
removal_rate   = 1/(np.float(model_params[model_params['Name']=='removal period'].Value))
hosp_rate                   = 1/(np.float(model_params[model_params['Name']=='hosp period'].Value))
death_rate                  = 1/(np.float(model_params[model_params['Name']=='death period'].Value))

death_prob          = np.float(model_params[model_params['Name']=='death prob'].Value)
number_compartments = int(model_params[model_params['Name']=='number_compartments'].Value)

beta_list           = [R_0*removal_rate  for R_0 in R_0_list] # R_0 mu/N, N=1

# infection_matrix    = np.ones((example_population_frame.shape[0],example_population_frame.shape[0]))
# print(control_data)
shield_decrease          = np.float(control_data[control_data['Name']=='Reduction in contact between groups'].Value)
shield_increase          = np.float(control_data[control_data['Name']=='Increase in contact within group'].Value)

better_hygiene = np.float(control_data.Value[control_data.Name=='Better hygiene'])

death_rate_with_ICU = 1/(np.float(model_params[model_params['Name']=='death period with ICU'].Value))
death_prob_with_ICU =    np.float(model_params[model_params['Name']=='death prob with ICU'].Value)



# print(shield_decrease)
# Parameters that may come into play later:
# ICU_capacity = 8/100000
# death prob and period given ICU care:
# import_rate = 1/(30*population) # 1 per month


class Parameters:
    def __init__(self):
        
        self.R_0_list = R_0_list
        self.beta_list = beta_list
        
        self.shield_increase = shield_increase
        self.shield_decrease = shield_decrease
        self.better_hygiene  = better_hygiene


        self.number_compartments = number_compartments

        self.asympt_prop         = asympt_prop

        self.latent_rate         = latent_rate
        self.removal_rate    = removal_rate
        self.hosp_rate           = hosp_rate        
        self.death_rate          = death_rate
        self.death_rate_with_ICU = death_rate_with_ICU
        
        self.death_prob          = death_prob
        self.death_prob_with_ICU = death_prob_with_ICU

        self.S_ind = 0
        self.E_ind = 1
        self.I_ind = 2
        self.A_ind = 3
        self.R_ind = 4
        self.H_ind = 5
        self.C_ind = 6
        self.D_ind = 7
        self.O_ind = 8






params = Parameters()



calculated_categories = ['S',
        'E',
        'I',
        'A',
        'R',
        'H',
        'C',
        'D',
        'O']

change_in_categories = ['C'+ii for ii in calculated_categories] # gives daily change for each category

longname = {'S': 'Susceptible',
            'E': 'Exposed',
            'I': 'Infected (symptomatic)',
            'A': 'Asymptomatically Infected',
            'R': 'Recovered',
            'H': 'Hospitalised',
            'C': 'Critical',
            'D': 'Deaths',
            'O': 'Offsite',
            'CS': 'Change in Susceptible',
            'CE': 'Change in Exposed',
            'CI': 'Change in Infected (symptomatic)',
            'CA': 'Change in Asymptomatically Infected',
            'CR': 'Change in Recovered',
            'CH': 'Change in Hospitalised',
            'CC': 'Change in Critical',
            'CD': 'Change in Deaths',
            'CO': 'Change in Offsite',
            'Ninf': 'Change in total active infections', # sum of E, I, A
}

shortname = {'S': 'Sus.',
             'E':  'Exp.',
             'I':  'Inf. (symp.)',
             'A':  'Asym.',
             'R':  'Rec.',
             'H':  'Hosp.',
             'C':  'Crit.',
             'D':  'Deaths',
             'O': 'Offsite',
             'CS': 'Change in Sus.',
             'CE': 'Change in Exp.',
             'CI': 'Change in Inf. (symp.)',
             'CA': 'Change in Asym.',
             'CR': 'Change in Rec.',
             'CH': 'Change in Hosp.',
             'CC': 'Change in Crit.',
             'CD': 'Change in Deaths',
             'CO': 'Change in Offsite',
             'Ninf': 'New Infected', # newly exposed to the disease = - change in susceptibles
}

colour = {'S':  'rgb(0,0,255)', #'blue',
          'E':  'rgb(255,150,255)', #'pink',
          'I':  'rgb(255,150,50)', #'orange',
          'A':  'rgb(255,50,50)', #'dunno',
          'R':  'rgb(0,255,0)', #'green',
          'H':  'rgb(255,0,0)', #'red',
          'C':  'rgb(50,50,50)', #'black',
          'D':  'rgb(130,0,255)', #'purple',
          'O':  'rgb(130,100,150)', #'dunno',
          'CS': 'rgb(0,0,255)', #'blue',
          'CE': 'rgb(255,150,255)', #'pink',
          'CI': 'rgb(255,150,50)', #'orange',
          'CA': 'rgb(255,50,50)', #'dunno',
          'CR': 'rgb(0,255,0)', #'green',
          'CH': 'rgb(255,0,0)', #'red',
          'CC': 'rgb(50,50,50)', #'black',
          'CD': 'rgb(130,0,255)', #'purple',
          'CO':  'rgb(130,100,150)', #'dunno',
          'Ninf': 'rgb(255,125,100)', #
        }

index = {'S': params.S_ind,
        'E': params.E_ind,
        'I': params.I_ind,
        'A': params.A_ind,
        'R': params.R_ind,
        'H': params.H_ind,
        'C': params.C_ind,
        'D': params.D_ind,
        'O': params.O_ind,
        'CS': 9,
        'CE': 10,
        'CI': 11,
        'CA': 12,
        'CR': 13,
        'CH': 14,
        'CC': 15,
        'CD': 16,
        'CO': 17,
        'Ninf': 18,
        }

categories = {}
for key in longname.keys():
    categories[key] = dict(longname = longname[key],
                           shortname = shortname[key],
                           colour = colour[key],
                           fill_colour = 'rgba' + colour[key][3:-1] + ',0.1)' ,
                           index = index[key]
                        )

