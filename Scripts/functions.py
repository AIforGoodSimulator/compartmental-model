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


##
# -----------------------------------------------------------------------------------
##
class simulator:
    def __init__(self):
        pass
    ##
#-----------------------------------------------------------------
        
    ##
    def ode_system(self,t,y,infection_matrix,age_categories,hospital_prob,critical_prob,beta,better_hygiene,remove_symptomatic,remove_high_risk,ICU_capacity):
        ##
        dydt = np.zeros(y.shape)

        I_vec = [ y[params.I_ind+i*params.number_compartments] for i in range(age_categories)]

        A_vec = [ y[params.A_ind+i*params.number_compartments] for i in range(age_categories)]

        C_vec = [ y[params.C_ind+i*params.number_compartments] for i in range(age_categories)]


        total_I = sum(I_vec)

        # better hygiene
        if t > better_hygiene['timing'][0] and t < better_hygiene['timing'][1]: # control in place
            control_factor = better_hygiene['value']
        else:
            control_factor = 1
        
        # removing symptomatic individuals
        if t > remove_symptomatic['timing'][0] and t < remove_symptomatic['timing'][1]: # control in place
            remove_symptomatic_rate = min(total_I,remove_symptomatic['rate'])  # if total_I too small then can't take this many off site at once
        else:
            remove_symptomatic_rate = 0

        S_removal = 0
        for i in range(age_categories - remove_high_risk['n_categories_removed'],age_categories):
            S_removal += y[params.S_ind + i*params.number_compartments] # add all old people to remove


        for i in range(age_categories):
            # removing symptomatic individuals
            # these are just immediately put into R or H; 
            # no longer infecting new but don't want to 'hide' the fact some of these will die
            # ideally there would be a slight delay
            # but the important thing is that they instantly stop infecting others
            move_sick_offsite = remove_symptomatic_rate * y[params.I_ind + i*params.number_compartments]/total_I # no age bias in who is moved

            # removing susceptible high risk individuals
            # these are moved into 'offsite'
            if i in range(age_categories - remove_high_risk['n_categories_removed'],age_categories) and t > remove_high_risk['timing'][0] and t < remove_high_risk['timing'][1]:
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
            dydt[params.S_ind + i*params.number_compartments] = (- y[params.S_ind + i*params.number_compartments] * control_factor * beta * (np.dot(infection_matrix[i,:],I_vec) + np.dot(infection_matrix[i,:],A_vec)) 
                                                                    - remove_high_risk_people * y[params.S_ind + i*params.number_compartments] / S_removal )
            # E
            dydt[params.E_ind + i*params.number_compartments] = ( y[params.S_ind + i*params.number_compartments] * control_factor * beta * (np.dot(infection_matrix[i,:],I_vec) + np.dot(infection_matrix[i,:],A_vec))
                                                                - params.latent_rate * y[params.E_ind + i*params.number_compartments])
            # I
            dydt[params.I_ind + i*params.number_compartments] = (params.latent_rate * (1-params.asympt_prop) * y[params.E_ind + i*params.number_compartments]
                                                                  - params.removal_rate * y[params.I_ind + i*params.number_compartments]
                                                                  - move_sick_offsite
                                                                  )
            # A
            dydt[params.A_ind + i*params.number_compartments] = (params.latent_rate * params.asympt_prop * y[params.E_ind + i*params.number_compartments]
                                                                 - params.removal_rate * y[params.A_ind + i*params.number_compartments])
            # R
            dydt[params.R_ind + i*params.number_compartments] = (params.removal_rate * (1 - hospital_prob[i]) * y[params.I_ind + i*params.number_compartments]
                                                                 + params.removal_rate * y[params.A_ind + i*params.number_compartments]
                                                                 + params.hosp_rate * (1 - critical_prob[i]) * y[params.H_ind + i*params.number_compartments]
                                                                 + move_sick_offsite  * (1 - hospital_prob[i]) # proportion of removed people who recovered once removed (no delay)
                                                                #  + remove_high_risk_people # now these removed people are just taken out of the system
                                                                 )
            # H
            dydt[params.H_ind + i*params.number_compartments] = (params.removal_rate * (hospital_prob[i]) * y[params.I_ind + i*params.number_compartments]
                                                                 - params.hosp_rate * y[params.H_ind + i*params.number_compartments]
                                                                 + params.death_rate * (1 - params.death_prob) * max(0,y[params.C_ind + i*params.number_compartments] - ICU_for_this_age) # recovered from critical care
                                                                 + params.death_rate_with_ICU * (1 - params.death_prob_with_ICU) * min(y[params.C_ind + i*params.number_compartments],ICU_for_this_age) # ICU
                                                                 + move_sick_offsite  * (hospital_prob[i]) # proportion of removed people who were hospitalised once removed (no delay)

                                                                 )
            # C
            dydt[params.C_ind + i*params.number_compartments] = (params.hosp_rate  * (critical_prob[i]) * y[params.H_ind + i*params.number_compartments]
                                                                 - params.death_rate * max(0,y[params.C_ind + i*params.number_compartments] - ICU_for_this_age) # without ICU
                                                                 - params.death_rate_with_ICU * min(y[params.C_ind + i*params.number_compartments],ICU_for_this_age) # up to hosp capacity get treatment
                                                                 )
            # D
            dydt[params.D_ind + i*params.number_compartments] = (params.death_rate * (params.death_prob) * max(0,y[params.C_ind + i*params.number_compartments] - ICU_for_this_age) # non ICU
                                                                + params.death_rate_with_ICU * (params.death_prob_with_ICU) * min(y[params.C_ind + i*params.number_compartments],ICU_for_this_age) # ICU
                                                                )
            # O
            dydt[params.O_ind + i*params.number_compartments] = remove_high_risk_people * y[params.S_ind + i*params.number_compartments] / S_removal


        return dydt
    ##
    #--------------------------------------------------------------------
    ##
    def run_model(self,T_stop,population,population_frame,infection_matrix,beta,control_dict): # ,beta_L_factor,beta_H_factor,t_control,T_stop,vaccine_time,ICU_grow,let_HR_out):
        
        E0 = 0
        I0 = 1/population
        A0 = 1/population
        R0 = 0
        H0 = 0
        C0 = 0
        D0 = 0
        O0 = 0 # offsite

        S0 = 1 - I0 - R0 - C0 - H0 - D0 - O0
        
        age_categories = int(population_frame.shape[0])

        y0 = np.zeros(params.number_compartments*age_categories) 

        population_vector = np.asarray(population_frame.Population)
        # print(population_vector)

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

        

        hospital_prob = np.asarray(population_frame.p_hospitalised)
        critical_prob = np.asarray(population_frame.p_critical)


        sol = ode(self.ode_system).set_f_params(infection_matrix,age_categories,hospital_prob,critical_prob,beta,control_dict['better_hygiene'],control_dict['remove_symptomatic'],control_dict['remove_high_risk'],control_dict['ICU_capacity'])
        
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





def simulate_range_of_R0s(population_frame, population, control_dict): # gives solution for middle R0, as well as solutions for a range of R0s between an upper and lower bound
    
    t_stop = 200


    # infection_matrix = np.asarray(pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/Contact_matrix.csv'))) #np.ones((population_frame.shape[0],population_frame.shape[0]))
    infection_matrix = np.asarray(pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/moria_contact_matrix.csv'))) #np.ones((population_frame.shape[0],population_frame.shape[0]))
    infection_matrix = infection_matrix[:,1:]

    next_generation_matrix = np.matmul(0.01*np.diag(population_frame.Population) , infection_matrix )
    largest_eigenvalue = max(np.linalg.eig(next_generation_matrix)[0]) # max eigenvalue


    beta_list = np.linspace(params.beta_list[0],params.beta_list[2],20)
    beta_list = (1/largest_eigenvalue)* beta_list

    if control_dict['shielding']['used']: # increase contact within group and decrease between groups
        divider = -1 # determines which groups separated. -1 means only oldest group separated from the rest
        
        infection_matrix[:divider,:divider]  = params.shield_increase*infection_matrix[:divider,:divider]
        infection_matrix[:divider,divider:]  = params.shield_decrease*infection_matrix[:divider,divider:]
        infection_matrix[divider:,:divider]  = params.shield_decrease*infection_matrix[divider:,:divider]
        infection_matrix[divider:,divider]   = params.shield_increase*infection_matrix[divider:,divider:]
        



    sols = []
    sols_raw = {}
    for beta in beta_list:
        result=simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,beta=beta,control_dict=control_dict)
        sols.append(result)
        sols_raw[beta*largest_eigenvalue/params.removal_rate]=result
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

    sols_out = []
    sols_out.append(simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,beta=params.beta_list[1],control_dict=control_dict))
    
    return sols_raw ,sols_out, [y_U95, y_UQ, y_LQ, y_L95, y_median] 




def object_dump(file_name,object_to_dump):
    # check if file path exists - if not create
    outdir =  os.path.dirname(file_name)
    if not os.path.exists(outdir):
        os.makedirs(os.path.join(cwd,outdir),exist_ok=True) 
    
    with open(file_name, 'wb') as handle:
        pickle.dump(object_to_dump,handle,protocol=pickle.HIGHEST_PROTOCOL) # protocol?

    return None




def generate_csv(data_to_save,population_frame,filename,input_type=None,time_vec=None):
    category_map = {    '0':  'S',
                        '1':  'E',
                        '2':  'I',
                        '3':  'A',
                        '4':  'R',
                        '5':  'H',
                        '6':  'C',
                        '7':  'D',
                        '8':  'O',
                        '9':  'CS',
                        '10': 'CE',
                        '11': 'CI',
                        '12': 'CA',
                        '13': 'CR',
                        '14': 'CH',
                        '15': 'CC',
                        '16': 'CD',
                        '17': 'CO',
                        '18': 'Ninf',
                        }


    if input_type=='percentile':
        csv_sol = np.transpose(data_to_save)

        solution_csv = pd.DataFrame(csv_sol)


        col_names = []
        for i in range(csv_sol.shape[1]):
            # ii = i % 8
            # jj = floor(i/8)
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
            
            solution_csv['R0']=[key]*solution_csv.shape[0]
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
