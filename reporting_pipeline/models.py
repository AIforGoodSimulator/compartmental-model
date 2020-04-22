import numpy as np
from scipy.integrate import ode
from scipy.stats import norm, gamma
import pandas as pd

class DesterministicSEIRv1:
    """

    This is a deterministic version of the SEIR model with additional compartments in hospitalisation, critical cases and death
    Age compartments are also built into this model so each age group is modeled as a sub model under the big model.

    """ 
    def __init__(self,model_params,population_params):
        #seperate out all the parameters carried by the params varaible
        _,_,_=model_params
        _,_,_=population_params
        #store the relevant parameters as class variables
        self._=_
    ##
#-----------------------------------------------------------------
        
    ##
    def ode_system(self,t,y,population_frame,control_time,beta_factor):
        ##
        dydt = np.zeros(y.shape)

        I_vec = [ y[self.params.I_ind+i*self.params.number_compartments] for i in range(population_frame.shape[0])] # age_categories

        if t > control_time[0] and t < control_time[1]: # control in place
            control_factor = beta_factor
        else:
            control_factor = 1
            


        for i in range(population_frame.shape[0]): # age_categories
            # S
            dydt[self.params.S_ind + i*self.params.number_compartments] = - y[self.params.S_ind + i*self.params.number_compartments] * control_factor * (np.dot(self.params.infection_matrix[i,:],I_vec)) 
            # E
            dydt[self.params.E_ind + i*self.params.number_compartments] = ( y[self.params.S_ind + i*self.params.number_compartments] * control_factor * (np.dot(self.params.infection_matrix[i,:],I_vec))
                                                                - self.params.become_infectious_rate * y[self.params.E_ind + i*self.params.number_compartments])
            # I
            dydt[self.params.I_ind + i*self.params.number_compartments] = (self.params.become_infectious_rate * y[params.E_ind + i*params.number_compartments] - 
                                                                  self.params.no_longer_infectious_rate * y[params.I_ind + i*params.number_compartments])
            # R
            dydt[self.params.R_ind + i*self.params.number_compartments] = (self.params.no_longer_infectious_rate * (1 - population_frame.p_hospitalised[i]) * y[self.params.I_ind + i*self.params.number_compartments] +
                                                                  self.params.hosp_rate * (1 - population_frame.p_critical[i]) * y[self.params.H_ind + i*self.params.number_compartments] + 
                                                                  self.params.death_rate * (1 - self.params.death_prob) * y[self.params.C_ind + i*self.params.number_compartments])
            # H
            dydt[self.params.H_ind + i*self.params.number_compartments] = (self.params.no_longer_infectious_rate * (population_frame.p_hospitalised[i]) * y[self.params.I_ind + i*self.params.number_compartments] -
                                                                  self.params.hosp_rate * y[self.params.H_ind + i*self.params.number_compartments])
            # C
            dydt[self.params.C_ind + i*self.params.number_compartments] = (self.params.hosp_rate  * (population_frame.p_critical[i]) * y[self.params.H_ind + i*self.params.number_compartments] -
                                                                  self.params.death_rate * y[self.params.C_ind + i*self.params.number_compartments])
            # D
            dydt[self.params.D_ind + i*self.params.number_compartments] = self.params.death_rate * (self.params.death_prob) * y[self.params.C_ind + i*self.params.number_compartments]

        return dydt
    ##
    #--------------------------------------------------------------------
    ##
    def run_model(self,T_stop,population,population_frame,control_time,beta_factor): # ,beta_L_factor,beta_H_factor,t_control,T_stop,vaccine_time,ICU_grow,let_HR_out):
        
        E0 = 0
        I0 = 1/population
        R0 = 0
        H0 = 0
        C0 = 0
        D0 = 0
        S0 = 1 - I0 - R0 - C0 - H0 - D0

        y0 = np.zeros(self.params.number_compartments*population_frame.shape[0]) # age_categories

        for i in range(population_frame.shape[0]):
            y0[self.params.S_ind + i*self.params.number_compartments] = (population_frame.Population[i]/100)*S0
            y0[self.params.E_ind + i*self.params.number_compartments] = (population_frame.Population[i]/100)*E0
            y0[self.params.I_ind + i*self.params.number_compartments] = (population_frame.Population[i]/100)*I0
            y0[self.params.R_ind + i*self.params.number_compartments] = (population_frame.Population[i]/100)*R0
            y0[self.params.H_ind + i*self.params.number_compartments] = (population_frame.Population[i]/100)*H0
            y0[self.params.C_ind + i*self.params.number_compartments] = (population_frame.Population[i]/100)*C0
            y0[self.params.D_ind + i*self.params.number_compartments] = (population_frame.Population[i]/100)*D0

        sol = ode(self.ode_system,jac=None).set_integrator('dopri5').set_f_params(population_frame,control_time,beta_factor)
        
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

class StochasticSEIR:
    def __init__(self,model_params,population_params):
        #seperate out all the parameters carried by the params varaible
        _,_,_=model_params
        _,_,_=population_params
        #store the relevant parameters as class variables
        self._=_
    def run_model():
        pass

#--------------------------------------------------------------------

class NetworkSEIR:
    def __init__(self,model_params,population_params):
        #seperate out all the parameters carried by the params varaible
        _,_,_=model_params
        _,_,_=population_params
        #store the relevant parameters as class variables
        self._=_
    def run_model():
        pass
