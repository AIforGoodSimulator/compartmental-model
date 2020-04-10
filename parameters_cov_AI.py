from math import log, exp
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from math import ceil
#------------------------------------------------------------
# age stats
# https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest

df2 = pd.DataFrame({'Age': ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80+'],
   'Hosp': [0.1,0.3,1.2,3.2,4.9,10.2,16.6,24.3,27.3],
   'Crit': [5,5,5,5,6.3,12.2,27.4,43.2,70.9],
   'Pop':  [11.8,9.5,16.2,13.3,14.6,12.1,10.8,7.1,4.6]
   })

divider = -2

hr_frac = sum(df2.Pop[divider:])/100

df2 = df2.assign(pop_low_prop=lambda x: x.Pop/(100*(1-hr_frac)),
    pop_high_prop=lambda x: x.Pop/(100*(hr_frac)))

df2.loc[  (df2.shape[0]-1+divider): ,'pop_low_prop' ] = 0
df2.loc[ :(df2.shape[0]-1+divider)  ,'pop_high_prop'] = 0

df2 = df2.assign(   weighted_hosp_low=lambda x: (x.Hosp/100)*x.pop_low_prop,
                    weighted_hosp_high=lambda x:(x.Hosp/100)*x.pop_high_prop,
                    weighted_crit_low=lambda x: (x.Crit/100)*x.pop_low_prop,
                    weighted_crit_high=lambda x:(x.Crit/100)*x.pop_high_prop)

######

frac_symptomatic = 0.55 # so e.g. 40% that weren't detected were bc no symptoms and the rest (5%) didn't identify vs e.g. flu

frac_hosp_L = sum(df2.weighted_hosp_low)*frac_symptomatic
frac_hosp_H = sum(df2.weighted_hosp_high)*frac_symptomatic
frac_crit_L = sum(df2.weighted_crit_low)
frac_crit_H = sum(df2.weighted_crit_high)

#------------------------------------------------------------
# disease params
N    = 1
recov_rate = 1/7
R_0        = 2.4
beta = R_0*recov_rate/N # R_0 mu/N

hosp_rate = 1/8
death_rate = 1/8
noICU  = 4 # rate by which death is accelerated without ICU care... so die in 1/(death_rate*noICU) days = 2 without ICU

crit_L      = hosp_rate*frac_crit_L
recover_L   = hosp_rate*(1-frac_crit_L)
crit_H      = hosp_rate*frac_crit_H
recover_H   = hosp_rate*(1-frac_crit_H) 

mu_L    = recov_rate*(1-frac_hosp_L)
gamma_L = recov_rate*frac_hosp_L
mu_H    = recov_rate*(1-frac_hosp_H)
gamma_H = recov_rate*frac_hosp_H

crit_death     =  death_rate*0.5
crit_recovery  =  death_rate*0.5


number_compartments = 6

fact_v = np.concatenate([[0.02,0.1],np.linspace(0.20,1,9)])
max_months_controlling = 18

ICU_growth = 1
ICU_capacity = 8/100000

UK_population = 60 * 10**(6)
import_rate = 1/(30*UK_population) # 1 per month

vaccinate_percent = 0.9 # vaccinate this many
vaccinate_rate = 0.55/(365*2/3) #10000/UK_population # per day
# https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0006158&type=printable


class Parameters:
    def __init__(self):
        self.mu_L  = mu_L
        self.mu_H  = mu_H
        self.gamma_L = gamma_L
        self.gamma_H = gamma_H
        self.beta  = beta
        self.N  = N
        self.hr_frac  = hr_frac
        self.crit_L = crit_L
        self.crit_H = crit_H
        self.recover_L = recover_L
        self.recover_H = recover_H
        self.crit_death = crit_death
        self.crit_recovery = crit_recovery
        self.ICU_capacity = ICU_capacity
        self.fact_v = fact_v
        self.max_months_controlling = max_months_controlling
        self.R_0 = R_0

        # self.UK_population = UK_population
        # self.hospital_production_rate = hospital_production_rate
        # self.T_stop = T_stop
        # self.initial_infections = initial_infections

        self.vaccinate_percent = vaccinate_percent
        self.vaccinate_rate = vaccinate_rate
        self.import_rate = import_rate
        self.ICU_growth = ICU_growth
        self.noICU = noICU



        self.number_compartments = number_compartments

        # only for app
        self.frac_hosp_L = frac_hosp_L
        self.frac_hosp_H = frac_hosp_H
        self.frac_crit_L = frac_crit_L
        self.frac_crit_H = frac_crit_H
        self.recovery_rate = recov_rate
        self.hosp_rate = hosp_rate        
        self.death_rate = death_rate


        self.S_L_ind = 0
        self.I_L_ind = 1
        self.R_L_ind = 2
        self.H_L_ind = 3
        self.C_L_ind = 4
        self.D_L_ind = 5
        self.S_H_ind = number_compartments + 0
        self.I_H_ind = number_compartments + 1
        self.R_H_ind = number_compartments + 2
        self.H_H_ind = number_compartments + 3
        self.C_H_ind = number_compartments + 4
        self.D_H_ind = number_compartments + 5




params = Parameters()



# print(params.frac_hosp_H)
# print(params.frac_hosp_L)
# print(params.frac_crit_H)
# print(params.frac_crit_L)
