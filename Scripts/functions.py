from initialise_parameters import params, control_data, categories, calculated_categories, change_in_categories
from math import exp, ceil, log, floor, sqrt
import numpy as np
from scipy.integrate import ode
from scipy.stats import norm, gamma
import pandas as pd
import statistics
import os
import pickle
from tqdm import tqdm
cwd = os.getcwd()
import pdb

def timing_function(t,time_vector):
    for ii in range(ceil(len(time_vector)/2)):
        if t>=time_vector[2*ii] and t<time_vector[2*ii+1]:
            return True
    # if wasn't in any of these time interval
    return False

##
# -----------------------------------------------------------------------------------
##
class simulator:
    def __init__(self):
        pass
    ##
#-----------------------------------------------------------------
        
    ##
    def ode_system(self,t,y, # state of system
                            infection_matrix,age_categories,symptomatic_prob,hospital_prob,critical_prob,beta, # params
                            latentRate,removalRate,hospRate,deathRateICU,deathRateNoIcu, # more params
                            better_hygiene,remove_symptomatic,remove_high_risk,ICU_capacity # control
                            ):
        ##
        dydt = np.zeros(y.shape)

        I_vec = [ y[params.I_ind+i*params.number_compartments] for i in range(age_categories)]
        # H_vec = [ y[params.H_ind+i*params.number_compartments] for i in range(age_categories)]
        C_vec = [ y[params.C_ind+i*params.number_compartments] for i in range(age_categories)]

        A_vec = [ y[params.A_ind+i*params.number_compartments] for i in range(age_categories)]

        total_I = sum(I_vec)

        # better hygiene
        if timing_function(t,better_hygiene['timing']): # control in place
            control_factor = better_hygiene['value']
        else:
            control_factor = 1
        
        # removing symptomatic individuals
        if timing_function(t,remove_symptomatic['timing']): # control in place
            remove_symptomatic_rate = min(total_I,remove_symptomatic['rate'])  # if total_I too small then can't take this many off site at once
        else:
            remove_symptomatic_rate = 0

        S_removal = 0
        for i in range(age_categories - remove_high_risk['n_categories_removed'],age_categories):
            S_removal += y[params.S_ind + i*params.number_compartments] # add all old people to remove


        for i in range(age_categories):
            # removing symptomatic individuals
            # these are put into Q ('quarantine');
            quarantine_sick = remove_symptomatic_rate * y[params.I_ind + i*params.number_compartments]/total_I # no age bias in who is moved

            # removing susceptible high risk individuals
            # these are moved into O ('offsite')
            if i in range(age_categories - remove_high_risk['n_categories_removed'],age_categories) and timing_function(t,remove_high_risk['timing']):
                remove_high_risk_people = min(remove_high_risk['rate'],S_removal) # only removing high risk (within time control window). Can't remove more than we have
            else:
                remove_high_risk_people = 0
            
            # ICU capacity
            if sum(C_vec)>0: # can't divide by 0
                ICU_for_this_age = ICU_capacity['value'] * y[params.C_ind + i*params.number_compartments]/sum(C_vec) # hospital beds allocated on a first come, first served basis
            else:
                ICU_for_this_age = ICU_capacity['value']




            # ODE system:
            # S
            dydt[params.S_ind + i*params.number_compartments] = (- y[params.S_ind + i*params.number_compartments] * control_factor * beta * (np.dot(infection_matrix[i,:],I_vec) + params.AsymptInfectiousFactor*np.dot(infection_matrix[i,:],A_vec)) 
                                                                    - remove_high_risk_people * y[params.S_ind + i*params.number_compartments] / S_removal )
            # E
            dydt[params.E_ind + i*params.number_compartments] = (  y[params.S_ind + i*params.number_compartments] * control_factor * beta * (np.dot(infection_matrix[i,:],I_vec) + params.AsymptInfectiousFactor*np.dot(infection_matrix[i,:],A_vec))
                                                                - latentRate * y[params.E_ind + i*params.number_compartments])
            # I
            dydt[params.I_ind + i*params.number_compartments] = (latentRate * (1-symptomatic_prob[i]) * y[params.E_ind + i*params.number_compartments]
                                                                  - removalRate * y[params.I_ind + i*params.number_compartments]
                                                                  - quarantine_sick
                                                                  )
            # A
            dydt[params.A_ind + i*params.number_compartments] = (latentRate * symptomatic_prob[i] * y[params.E_ind + i*params.number_compartments]
                                                                 - removalRate * y[params.A_ind + i*params.number_compartments])
            # H
            dydt[params.H_ind + i*params.number_compartments] = (removalRate * (hospital_prob[i]) * y[params.I_ind + i*params.number_compartments]
                                                                 - hospRate * y[params.H_ind + i*params.number_compartments]
                                                                #  + deathRateNoIcu * (1 - params.death_prob) * max(0,y[params.C_ind + i*params.number_compartments] - ICU_for_this_age) # recovered despite no ICU (0, since now assume death_prob is 1)
                                                                 + deathRateICU * (1 - params.death_prob_with_ICU) * min(y[params.C_ind + i*params.number_compartments],ICU_for_this_age) # recovered from ICU
                                                                 + (hospital_prob[i]) * params.quarant_rate * y[params.Q_ind + i*params.number_compartments] # proportion of removed people who were hospitalised once returned
                                                                 )
            # Critical care (ICU)
            dydt[params.C_ind + i*params.number_compartments] = ( min(hospRate  * (critical_prob[i]) * y[params.H_ind + i*params.number_compartments], 
                                                                                max(0, 
                                                                                ICU_for_this_age - y[params.C_ind + i*params.number_compartments]
                                                                                 + deathRateICU * y[params.C_ind + i*params.number_compartments]  # with ICU treatment
                                                                                )
                                                                                ) # amount entering is minimum of: amount of beds available**/number needing it
                                                                                # **including those that will be made available by new deaths
                                                                 - deathRateICU * y[params.C_ind + i*params.number_compartments]  # with ICU treatment
                                                                 )
            
            # Uncared - no ICU
            dydt[params.U_ind + i*params.number_compartments] = ( hospRate  * (critical_prob[i]) * y[params.H_ind + i*params.number_compartments] # number needing care
                                                                 - min(hospRate  * (critical_prob[i]) * y[params.H_ind + i*params.number_compartments],
                                                                     max(0,
                                                                     ICU_for_this_age - y[params.C_ind + i*params.number_compartments]
                                                                    + deathRateICU * y[params.C_ind + i*params.number_compartments] 
                                                                     ) ) # minus number who get it (these entered category C) 
                                                                 - deathRateNoIcu * y[params.U_ind + i*params.number_compartments] # without ICU treatment
                                                                 )
    
            # R
            dydt[params.R_ind + i*params.number_compartments] = (removalRate * (1 - hospital_prob[i]) * y[params.I_ind + i*params.number_compartments]
                                                                 + removalRate * y[params.A_ind + i*params.number_compartments]
                                                                 + hospRate * (1 - critical_prob[i]) * y[params.H_ind + i*params.number_compartments]
                                                                 + (1 - hospital_prob[i]) * params.quarant_rate * y[params.Q_ind + i*params.number_compartments] # proportion of removed people who recovered once returned
                                                                 )
            
            # D
            dydt[params.D_ind + i*params.number_compartments] = (deathRateNoIcu * y[params.U_ind + i*params.number_compartments] # died without ICU treatment (all cases that don't get treatment die)
                                                                + deathRateICU * (params.death_prob_with_ICU) * y[params.C_ind + i*params.number_compartments] # died despite attempted ICU treatment
                                                                )
            # O
            dydt[params.O_ind + i*params.number_compartments] = remove_high_risk_people * y[params.S_ind + i*params.number_compartments] / S_removal

            # Q
            dydt[params.Q_ind + i*params.number_compartments] = quarantine_sick - params.quarant_rate * y[params.Q_ind + i*params.number_compartments]



        return dydt
    ##
    #--------------------------------------------------------------------
    ##
    def run_model(self,T_stop,population,population_frame,infection_matrix,beta,
                control_dict,  # control
                latentRate     = params.latent_rate,
                removalRate    = params.removal_rate,
                hospRate       = params.hosp_rate,
                deathRateICU   = params.death_rate_with_ICU,
                deathRateNoIcu = params.death_rate # more params
                ):
        
        E0 = 0 # exposed
        I0 = 1/population # sympt
        A0 = 1/population # asympt
        R0 = 0 # recovered
        H0 = 0 # hospitalised/needing hospital care
        C0 = 0 # critical (cared)
        D0 = 0 # dead
        O0 = 0 # offsite
        Q0 = 0 # quarantined
        U0 = 0 # critical (uncared)



        S0 = 1 - I0 - R0 - C0 - H0 - D0 - O0 - Q0 - U0
        
        age_categories = int(population_frame.shape[0])

        y0 = np.zeros(params.number_compartments*age_categories) 

        population_vector = np.asarray(population_frame.Population_structure)

        
        
        # initial conditions
        for i in range(age_categories):
            y0[params.S_ind + i*params.number_compartments] = (population_vector[i]/100)*S0
            y0[params.E_ind + i*params.number_compartments] = (population_vector[i]/100)*E0
            y0[params.I_ind + i*params.number_compartments] = (population_vector[i]/100)*I0
            y0[params.A_ind + i*params.number_compartments] = (population_vector[i]/100)*A0
            y0[params.R_ind + i*params.number_compartments] = (population_vector[i]/100)*R0
            y0[params.H_ind + i*params.number_compartments] = (population_vector[i]/100)*H0
            y0[params.C_ind + i*params.number_compartments] = (population_vector[i]/100)*C0
            y0[params.D_ind + i*params.number_compartments] = (population_vector[i]/100)*D0
            y0[params.O_ind + i*params.number_compartments] = (population_vector[i]/100)*O0
            y0[params.Q_ind + i*params.number_compartments] = (population_vector[i]/100)*Q0
            y0[params.U_ind + i*params.number_compartments] = (population_vector[i]/100)*U0



        
        symptomatic_prob = np.asarray(population_frame.p_symptomatic)
        hospital_prob = np.asarray(population_frame.p_hospitalised)
        critical_prob = np.asarray(population_frame.p_critical)


        sol = ode(self.ode_system).set_f_params(infection_matrix,age_categories,symptomatic_prob,hospital_prob,critical_prob,beta, # params
                latentRate,removalRate,hospRate,deathRateICU,deathRateNoIcu, # more params
                control_dict['better_hygiene'],control_dict['remove_symptomatic'],control_dict['remove_high_risk'],control_dict['ICU_capacity'] # control params
                )
        
        tim = np.linspace(0,T_stop, T_stop+1) # 1 time value per day
        
        sol.set_initial_value(y0,tim[0])

        y_out = np.zeros((len(y0),len(tim)))
        
        i2 = 0
        y_out[:,0] = sol.y
        for t in tim[1:]:
                if sol.successful():
                    sol.integrate(t)
                    i2=i2+1
                    y_out[:,i2] = sol.y
                else:
                    raise RuntimeError('ode solver unsuccessful')




        
        y_plot = np.zeros((len(categories.keys()), len(tim) ))
        for name in calculated_categories:

            y_plot[categories[name]['index'],:] = y_out[categories[name]['index'],:]
            for i in range(1, population_frame.shape[0]): # age_categories
                y_plot[categories[name]['index'],:] = y_plot[categories[name]['index'],:] + y_out[categories[name]['index'] + i*params.number_compartments,:]

        for name in change_in_categories: # daily change in
            name_changed_var = name[-1] # name of the variable we want daily change of
            y_plot[categories[name]['index'],:] = np.concatenate([[0],np.diff(y_plot[categories[name_changed_var]['index'],:])])
        
        # finally, 
        E = y_plot[categories['CE']['index'],:]
        I = y_plot[categories['CI']['index'],:]
        A = y_plot[categories['CA']['index'],:]

        y_plot[categories['Ninf']['index'],:] = [E[i] + I[i] + A[i] for i in range(len(E))] # change in total number of people with active infection
        
        return {'y': y_out,'t': tim, 'y_plot': y_plot}

#--------------------------------------------------------------------


def GeneratePercentiles(sols):
    n_time_points = len(sols[0]['t'])
    
    y_plot = np.zeros((len(categories.keys()), len(sols) , n_time_points ))

    for k, sol in enumerate(sols):
        sol['y'] = np.asarray(sol['y'])
        for name in categories.keys():
            y_plot[categories[name]['index'],k,:] = sol['y_plot'][categories[name]['index']]

    y_L95, y_U95, y_LQ, y_UQ, y_median = [np.zeros((len(categories.keys()),n_time_points)) for i in range(5)]

    for name in categories.keys():
        y_L95[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],2.5) for i in range(n_time_points) ])
        y_LQ[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],25) for i in range(n_time_points) ])
        y_UQ[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],75) for i in range(n_time_points) ])
        y_U95[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],97.5) for i in range(n_time_points) ])
        
        y_median[categories[name]['index'],:] = np.asarray([statistics.median(y_plot[categories[name]['index'],:,i]) for i in range(n_time_points) ])
    return [y_U95, y_UQ, y_LQ, y_L95, y_median]

def GenerateInfectionMatrix(population_frame,camp,control_dict):
    infection_matrix = np.asarray(pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/Contact_matrix_' + camp + '.csv'))) #np.ones((population_frame.shape[0],population_frame.shape[0]))
    infection_matrix = infection_matrix[:,1:]

    next_generation_matrix = np.matmul(0.01*np.diag(population_frame.Population_structure) , infection_matrix )
    largest_eigenvalue = max(np.linalg.eig(next_generation_matrix)[0]) # max eigenvalue
    


    beta_list = np.linspace(params.beta_list[0],params.beta_list[2],20)
    beta_list = np.real((1/largest_eigenvalue)* beta_list) # in case eigenvalue imaginary

    if control_dict['shielding']['used']: # increase contact within group and decrease between groups
        divider = -1 # determines which groups separated. -1 means only oldest group separated from the rest
        
        infection_matrix[:divider,:divider]  = params.shield_increase*infection_matrix[:divider,:divider]
        infection_matrix[:divider,divider:]  = params.shield_decrease*infection_matrix[:divider,divider:]
        infection_matrix[divider:,:divider]  = params.shield_decrease*infection_matrix[divider:,:divider]
        infection_matrix[divider:,divider]   = params.shield_increase*infection_matrix[divider:,divider:]
    
    return infection_matrix, beta_list, largest_eigenvalue



def simulate_range_of_R0s(population_frame, population, control_dict, camp, t_stop=200): # gives solution for middle R0, as well as solutions for a range of R0s between an upper and lower bound
    
    infection_matrix, beta_list, largest_eigenvalue = GenerateInfectionMatrix(population_frame,camp,control_dict)

    sols = []
    sols_raw = {}
    for beta in beta_list:
        result=simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,beta=beta,control_dict=control_dict)
        sols.append(result)
        sols_raw[beta*largest_eigenvalue/params.removal_rate]=result

    [y_U95, y_UQ, y_LQ, y_L95, y_median] = GeneratePercentiles(sols)

    StandardSol = []
    StandardSol.append(simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,beta=params.beta_list[1],control_dict=control_dict))
    
    return sols_raw, StandardSol, [y_U95, y_UQ, y_LQ, y_L95, y_median] 





def SimulateOverRangeOfParameters(population_frame, population, control_dict, camp, numberOfIterations, t_stop=200):
    
    infection_matrix, beta_list, largest_eigenvalue = GenerateInfectionMatrix(population_frame,camp,control_dict)

    ParamCsv = pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/GeneratedParams.csv'))

    sols = []
    configDict = []
    sols_raw = {}
    for ii in tqdm(range(min(numberOfIterations,len(ParamCsv)))):
        latentRate  = 1/ParamCsv.LatentPeriod[ii]
        removalRate = 1/ParamCsv.RemovalPeriod[ii]
        
        beta        = removalRate*ParamCsv.R0[ii]/largest_eigenvalue
        
        hospRate       = 1/ParamCsv.HospPeriod[ii]
        deathRateICU   = 1/ParamCsv.DeathICUPeriod[ii]
        deathRateNoIcu = 1/ParamCsv.DeathNoICUPeriod[ii]

        
        result = simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,beta=beta,
                                control_dict= control_dict,
                                latentRate  = latentRate,
                                removalRate = removalRate,
                                hospRate    = hospRate,
                                deathRateICU = deathRateICU,
                                deathRateNoIcu = deathRateNoIcu
                                )
        sols.append(result)

        Dict = dict(beta       = beta,
                latentRate     = latentRate,
                removalRate    = removalRate,
                hospRate       = hospRate,
                deathRateICU   = deathRateICU,
                deathRateNoIcu = deathRateNoIcu
                )
        configDict.append(Dict)
        sols_raw[(ParamCsv.R0[ii],latentRate,removalRate,hospRate,deathRateICU,deathRateNoIcu)]=result

    [y_U95, y_UQ, y_LQ, y_L95, y_median] = GeneratePercentiles(sols)

    # standard run
    StandardSol = []
    StandardSol.append(simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,beta=params.beta_list[1],control_dict=control_dict))
    
    return sols_raw, StandardSol, [y_U95, y_UQ, y_LQ, y_L95, y_median], configDict







def object_dump(file_name,object_to_dump):
    # check if file path exists - if not create
    outdir =  os.path.dirname(file_name)
    if not os.path.exists(outdir):
        os.makedirs(os.path.join(cwd,outdir),exist_ok=True) 
    
    with open(file_name, 'wb') as handle:
        pickle.dump(object_to_dump,handle,protocol=pickle.HIGHEST_PROTOCOL)

    return None



def generate_csv(data_to_save,population_frame,filename,input_type=None,time_vec=None):
    
    category_map = {}
    for key in categories.keys():
        category_map[str(categories[key]['index'])] = key

    print(category_map)

    if input_type=='percentile':
        csv_sol = np.transpose(data_to_save)

        solution_csv = pd.DataFrame(csv_sol)


        col_names = []
        for i in range(csv_sol.shape[1]):
            col_names.append(categories[category_map[str(i)]]['longname'])
            

        solution_csv.columns = col_names
        solution_csv['Time'] = time_vec
        # this is our dataframe to be saved

    elif input_type=='raw':

        final_frame=pd.DataFrame()

        for key, value in tqdm(data_to_save.items()):
            csv_sol = np.transpose(value['y']) # age structured

            solution_csv = pd.DataFrame(csv_sol)

            # setup column names
            col_names = []
            number_categories_with_age = csv_sol.shape[1]
            for i in range(number_categories_with_age):
                ii = i % params.number_compartments
                jj = floor(i/params.number_compartments)

                col_names.append(categories[category_map[str(ii)]]['longname'] +  ': ' + str(np.asarray(population_frame.Age)[jj]) )

            solution_csv.columns = col_names
            solution_csv['Time'] = value['t']

            for j in range(len(categories.keys())): # params.number_compartments
                solution_csv[categories[category_map[str(j)]]['longname']] = value['y_plot'][j] # summary/non age-structured
            
            (R0,latentRate,removalRate,hospRate,deathRateICU,deathRateNoIcu)=key
            solution_csv['R0']=[R0]*solution_csv.shape[0]
            solution_csv['latentRate']=[latentRate]*solution_csv.shape[0]
            solution_csv['removalRate']=[removalRate]*solution_csv.shape[0]
            solution_csv['hospRate']=[hospRate]*solution_csv.shape[0]
            solution_csv['deathRateICU']=[deathRateICU]*solution_csv.shape[0]
            solution_csv['deathRateNoIcu']=[deathRateNoIcu]*solution_csv.shape[0]
            final_frame=pd.concat([final_frame, solution_csv], ignore_index=True)

        solution_csv=final_frame
        #this is our dataframe to be saved

    elif input_type=='solution':
        csv_sol = np.transpose(data_to_save[0]['y']) # age structured

        solution_csv = pd.DataFrame(csv_sol)

        # setup column names
        col_names = []
        number_categories_with_age = csv_sol.shape[1]
        for i in range(number_categories_with_age):
            ii = i % params.number_compartments
            jj = floor(i/params.number_compartments)
            
            col_names.append(categories[category_map[str(ii)]]['longname'] +  ': ' + str(np.asarray(population_frame.Age)[jj]) )

        solution_csv.columns = col_names
        solution_csv['Time'] = data_to_save[0]['t']

        for j in range(len(categories.keys())): # params.number_compartments
            solution_csv[categories[category_map[str(j)]]['longname']] = data_to_save[0]['y_plot'][j] # summary/non age-structured
        # this is our dataframe to be saved


    # save it
    solution_csv.to_csv(os.path.join(os.path.dirname(cwd),'CSV_output/' + filename + '.csv' ))


    return None
