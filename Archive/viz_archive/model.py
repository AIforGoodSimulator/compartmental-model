import ipywidgets as widgets
from ipywidgets import fixed,interactive,Layout
import numpy as np
import pandas as pd
from math import floor

def simulate_R0_unmitigated(R_0,column,t_stop=200): # gives solution for middle R0, as well as solutions for a range of R0s between an upper and lower bound
    from plots import plot_by_age
    import sys, os
    cwd = os.getcwd()
    sys.path.append(os.path.abspath(os.path.join('..', 'Scripts')))
    from functions import simulator
    from configs.baseline import camp, population_frame, population, control_dict
    from initialise_parameters import params,categories
    # infection_matrix = np.asarray(pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/Contact_matrix.csv'))) #np.ones((population_frame.shape[0],population_frame.shape[0]))
    infection_matrix = np.asarray(pd.read_csv(os.path.join(os.path.dirname(cwd),'Parameters/Contact_matrix_' + camp + '.csv'))) #np.ones((population_frame.shape[0],population_frame.shape[0]))
    infection_matrix = infection_matrix[:,1:]

    next_generation_matrix = np.matmul(0.01*np.diag(population_frame.Population_structure) , infection_matrix )
    largest_eigenvalue = max(np.linalg.eig(next_generation_matrix)[0]) # max eigenvalue
    beta=R_0*params.removal_rate
    beta = np.real((1/largest_eigenvalue)* beta) # in case eigenvalue imaginary
    sols_raw = {}
    result=simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,beta=beta,control_dict=control_dict)
    sols_raw[beta*largest_eigenvalue/params.removal_rate]=result
    final_frame=pd.DataFrame()
    category_map = {    '0':  'S',
                        '1':  'E',
                        '2':  'I',
                        '3':  'A',
                        '4':  'R',
                        '5':  'H',
                        '6':  'C',
                        '7':  'D',
                        '8':  'O',
                        '9':  'CS', # change in S
                        '10': 'CE', # change in E
                        '11': 'CI', # change in I
                        '12': 'CA', # change in A
                        '13': 'CR', # change in R
                        '14': 'CH', # change in H
                        '15': 'CC', # change in C
                        '16': 'CD', # change in D
                        '17': 'CO', # change in O
                        '18': 'Ninf',
                        }
    for key, value in sols_raw.items():
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
    plot_by_age(column,solution_csv)

def simulate_R0_unmitigated_plot(simulate_R0_unmitigated):
    w = interactive(simulate_R0_unmitigated,
                    R_0=widgets.FloatSlider(min=1, max=7, step=0.1,
                    value=2,
                    description='R0:',continuous_update=False),
                    column=widgets.Dropdown(
                    options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],
                    value='Infected (symptomatic)',
                    description='Category:'
                    ))
    words = widgets.Label('Impact of R0 value on the do nothing scenario outcomes')
    container=widgets.VBox([words,w])
    container.layout.width = '100%'
    container.layout.border = '2px solid grey'
    container.layout.justify_content = 'space-around'
    container.layout.align_items = 'center'
    return container

