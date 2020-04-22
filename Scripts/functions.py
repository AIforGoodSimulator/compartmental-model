from initialise_parameters import params, control_data
from math import exp, ceil, log, floor, sqrt
import numpy as np
from scipy.integrate import ode
from scipy.stats import norm, gamma
import pandas as pd
import statistics
from plotter import categories
import os
import pickle
##
# -----------------------------------------------------------------------------------
##
class simulator:
    def __init__(self):
        pass
    ##
#-----------------------------------------------------------------
        
    ##
    def ode_system(self,t,y,infection_matrix,age_categories,hospital_prob,critical_prob,control_time,beta,beta_factor):
        ##
        dydt = np.zeros(y.shape)

        I_vec = [ y[params.I_ind+i*params.number_compartments] for i in range(age_categories)]

        A_vec = [ y[params.A_ind+i*params.number_compartments] for i in range(age_categories)]


        if t > control_time[0] and t < control_time[1]: # control in place
            control_factor = beta_factor
        else:
            control_factor = 1
            
        


        for i in range(age_categories): # age_categories
            # S
            dydt[params.S_ind + i*params.number_compartments] = - y[params.S_ind + i*params.number_compartments] * control_factor * beta * (np.dot(infection_matrix[i,:],I_vec) + np.dot(infection_matrix[i,:],A_vec)) 
            # E
            dydt[params.E_ind + i*params.number_compartments] = ( y[params.S_ind + i*params.number_compartments] * control_factor * beta * (np.dot(infection_matrix[i,:],I_vec) + np.dot(infection_matrix[i,:],A_vec))
                                                                - params.latent_rate * y[params.E_ind + i*params.number_compartments])
            # I
            dydt[params.I_ind + i*params.number_compartments] = (params.latent_rate * (1-params.asympt_prop) * y[params.E_ind + i*params.number_compartments] - 
                                                                  params.removal_rate * y[params.I_ind + i*params.number_compartments])
            # A
            dydt[params.A_ind + i*params.number_compartments] = (params.latent_rate * params.asympt_prop * y[params.E_ind + i*params.number_compartments] - 
                                                                  params.removal_rate * y[params.A_ind + i*params.number_compartments])
            # R
            dydt[params.R_ind + i*params.number_compartments] = (params.removal_rate * (1 - hospital_prob[i]) * y[params.I_ind + i*params.number_compartments] +
                                                                 params.removal_rate * y[params.A_ind + i*params.number_compartments] +
                                                                 params.hosp_rate * (1 - critical_prob[i]) * y[params.H_ind + i*params.number_compartments] + 
                                                                 params.death_rate * (1 - params.death_prob) * y[params.C_ind + i*params.number_compartments])
            # H
            dydt[params.H_ind + i*params.number_compartments] = (params.removal_rate * (hospital_prob[i]) * y[params.I_ind + i*params.number_compartments] -
                                                                  params.hosp_rate * y[params.H_ind + i*params.number_compartments])
            # C
            dydt[params.C_ind + i*params.number_compartments] = (params.hosp_rate  * (critical_prob[i]) * y[params.H_ind + i*params.number_compartments] -
                                                                  params.death_rate * y[params.C_ind + i*params.number_compartments])
            # D
            dydt[params.D_ind + i*params.number_compartments] = params.death_rate * (params.death_prob) * y[params.C_ind + i*params.number_compartments]

        return dydt
    ##
    #--------------------------------------------------------------------
    ##
    def run_model(self,T_stop,population,population_frame,infection_matrix,control_time,beta,beta_factor): # ,beta_L_factor,beta_H_factor,t_control,T_stop,vaccine_time,ICU_grow,let_HR_out):
        
        E0 = 0
        I0 = 1/population
        A0 = 1/population
        R0 = 0
        H0 = 0
        C0 = 0
        D0 = 0
        S0 = 1 - I0 - R0 - C0 - H0 - D0
        
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
        

        hospital_prob = np.asarray(population_frame.p_hospitalised)
        critical_prob = np.asarray(population_frame.p_critical)

        sol = ode(self.ode_system,jac=None).set_integrator('dopri5').set_f_params(infection_matrix,age_categories,hospital_prob,critical_prob,control_time,beta,beta_factor)
        
        tim = np.linspace(0,T_stop, 301) # use 141 time values
        
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
        
        return {'y': y_out,'t': tim}

#--------------------------------------------------------------------


# print(simulator().run_model(200))




def simulate_range_of_R0s(preset,timings,camp,population_frame, population): # gives solution as well as upper and lower bounds
    
    t_stop = 200

    beta_factor = np.float(control_data.Value[control_data.Name==preset])

    # population_frame, population = preparePopulationFrame(camp)
    
    infection_matrix = np.ones((population_frame.shape[0],population_frame.shape[0]))
    beta_list = np.linspace(params.beta_list[0],params.beta_list[2],20)
    sols = []
    for beta in beta_list:
        sols.append(simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,control_time=timings,beta=beta,beta_factor=beta_factor))

    n_time_points = len(sols[0]['t'])

    y_plot = np.zeros((len(categories.keys()), len(sols) , n_time_points ))

    for k, sol in enumerate(sols):
        for name in categories.keys():
            sol['y'] = np.asarray(sol['y'])

            # print(name,categories[name]['index'])
            y_plot[categories[name]['index'],k,:] = sol['y'][categories[name]['index'],:]
            for i in range(1, population_frame.shape[0]): # age_categories
                y_plot[categories[name]['index'],k,:] = y_plot[categories[name]['index'],k,:] + sol['y'][categories[name]['index'] + i*params.number_compartments,:]


    y_L95, y_U95, y_LQ, y_UQ, y_median = [np.zeros((params.number_compartments,n_time_points)) for i in range(5)]
    # y_max = np.zeros((params.number_compartments,n_time_points))

    for name in categories.keys():
        y_L95[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],2.5) for i in range(n_time_points) ])
        y_LQ[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],25) for i in range(n_time_points) ])
        y_UQ[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],75) for i in range(n_time_points) ])
        y_U95[categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[categories[name]['index'],:,i],97.5) for i in range(n_time_points) ])
        
        y_median[categories[name]['index'],:] = np.asarray([statistics.median(y_plot[categories[name]['index'],:,i]) for i in range(n_time_points) ])

        # y_min[categories[name]['index'],:] = [min(y_plot[categories[name]['index'],:,i]) for i in range(n_time_points)]
        # y_max[categories[name]['index'],:] = [max(y_plot[categories[name]['index'],:,i]) for i in range(n_time_points)]

    sols_out = []
    sols_out.append(simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,control_time=timings,beta=params.beta_list[1],beta_factor=beta_factor))
    
    return sols_out, [y_U95, y_UQ, y_LQ, y_L95, y_median] 




def object_dump(file_name,object_to_dump):
    # check if file path exists - if not create
    outdir =  os.path.dirname(file_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True) 
    
    with open(file_name, 'wb') as handle:
        pickle.dump(object_to_dump,handle,protocol=pickle.HIGHEST_PROTOCOL) # protocol?

    return None



