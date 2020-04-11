from math import log, exp
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from math import ceil
#------------------------------------------------------------
# age stats
# https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest

df2 = pd.DataFrame({'Age': ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80+'],
   'Hosp_given_symptomatic': [0.1,0.3,1.2,3.2,4.9,10.2,16.6,24.3,27.3],
   'Crit': [5,5,5,5,6.3,12.2,27.4,43.2,70.9],
   'Pop':  [11.8,9.5,16.2,13.3,14.6,12.1,10.8,7.1,4.6]
   })

frac_symptomatic = 0.55 # so e.g. 40% that weren't detected were bc no symptoms and the rest (5%) didn't identify vs e.g. flu

df2 = df2.assign(p_hosp = lambda x: (x.Hosp_given_symptomatic/100)*frac_symptomatic,
                p_crit = lambda x: (x.Crit/100))
# print(df2.p_hosp[2])
# exit()

#------------------------------------------------------------
# disease params
N    = 1
non_infectious_rate = 1/7
R_0        = 2.4
beta = R_0*non_infectious_rate/N # R_0 mu/N

infection_matrix = beta*np.ones((df2.shape[0],df2.shape[0]))

infectious_rate = 1/2
hosp_rate = 1/8
death_rate = 1/8
death_rate_noICU  = 1/2
death_prob     = 0.5

number_compartments = 7

fact_v = np.concatenate([[0.02,0.1],np.linspace(0.20,1,9)])
max_months_controlling = 18

# ICU_growth = 1
# ICU_capacity = 8/100000

population = 200000
import_rate = 1/(30*population) # 1 per month

vaccinate_percent = 0 # 0.9 # vaccinate this many
vaccinate_rate = 0.55/(365*2/3) #10000/UK_population # per day
# https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0006158&type=printable


class Parameters:
    def __init__(self):
        self.infection_matrix  = infection_matrix
        self.N  = N
        self.population = population

        # self.ICU_capacity = ICU_capacity
        self.fact_v = fact_v
        self.max_months_controlling = max_months_controlling
        self.R_0 = R_0

        # self.vaccinate_percent = vaccinate_percent
        # self.vaccinate_rate = vaccinate_rate
        # self.import_rate = import_rate
        # self.ICU_growth = ICU_growth



        self.number_compartments = number_compartments

        # only for app
        self.infectious_rate  = infectious_rate
        self.non_infectious_rate = non_infectious_rate
        self.hosp_rate = hosp_rate        
        self.death_rate = death_rate
        self.death_rate_noICU = death_rate_noICU
        self.death_prob     = death_prob


        self.age_categories = df2.shape[0]
        self.dataframe = df2
        self.S_ind = 0
        self.E_ind = 1
        self.I_ind = 2
        self.R_ind = 3
        self.H_ind = 4
        self.C_ind = 5
        self.D_ind = 6





params = Parameters()

