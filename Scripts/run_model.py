import pandas as pd
from math import floor, ceil, exp
from initialise_parameters import params, parameter_csv, categories
import numpy as np
import plotly.graph_objects as go
from functions import simulator, simulate_range_of_R0s, object_dump, generate_csv
from plotter import figure_generator, age_structure_plot, stacked_bar_plot, uncertainty_plot
import pickle
import os

#import the config file for the experimental setup 
# baseline experiment
from configs.baseline import camp, population_frame, population, control_dict
# better hygiene from day 0
# from configs.better_hygiene import camp, population_frame, population, control_dict
# build up ICU capacity from 6 to 100
# from configs.icu100 import camp, population_frame, population, control_dict
# remove people form the camp (here we vary the parameters in the config file to explore the number of people removed and to which period of time removing people is still effective)
# from configs.remove_symptomatic import camp, population_frame, population, control_dict
# shielding the old population/high risk
# from configs.shielding import camp, population_frame, population, control_dict
# remove high risk people form the camp (here we vary the parameters in the config file to explore the number of people removed and to which period of time removing people is still effective)
# from configs.remove_highrisk import camp, population_frame, population, control_dict

def run_simulation(mode='experiment'):
    # cd into Scripts
    cwd = os.getcwd()

    ##----------------------------------------------------------------
    # load a saved solution?
    load = True
    # save generated solution? Only generates new if not loading old
    # saves as a python pickle object
    save = True
    save_csv = True
    # plot output?
    plot_output = False
    save_plots  = False # needs plot_output to be True
    #simulation timestep
    t_sim=200

    ##----------------------------------------------------------------
    param_string = "Camp=%s_%shygieneT=%s_remInfRate=%s_remInfT=%s_Shield=%s_RemHrRate=%s_RemHrTime=%s_ICU=%s" %(camp,
                                                                                                                control_dict['better_hygiene']['value'],
                                                                                                               control_dict['better_hygiene']['timing'],
                                                                                                               ceil(population*control_dict['remove_symptomatic']['rate']),
                                                                                                               control_dict['remove_symptomatic']['timing'],
                                                                                                               control_dict['shielding']['used'],
                                                                                                               ceil(population*control_dict['remove_high_risk']['rate']),
                                                                                                               control_dict['remove_high_risk']['timing'],
                                                                                                               ceil(population*control_dict['ICU_capacity']['value']))
    solution_name   = 'Solution_' + param_string    
    percentile_name = 'Percentiles_'  + param_string

    already_exists_soln       = os.path.isfile(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name   + '.pickle'))
    already_exists_percentile = os.path.isfile(os.path.join(os.path.dirname(cwd),'saved_runs/' + percentile_name + '.pickle'))



    if not load or not (already_exists_soln and already_exists_percentile): # generate solution if not wanting to load, or if wanting to load but at least one file missing
        # run model - change inputs via 'config.py'
        print('running the model to produce results')
        sols_raw,sols, percentiles = simulate_range_of_R0s(population_frame, population, control_dict, camp,t_stop=t_sim) # returns solution for middle R0 and then minimum and maximum values by scanning across a range defined by low and high R0
        if save:
            object_dump(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name + '_all.pickle'),  sols_raw)
            # object_dump(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name   + '.pickle'),  sols)
            # object_dump(os.path.join(os.path.dirname(cwd),'saved_runs/' + percentile_name + '.pickle'),  percentiles)
    else:
        print('retrieving results from saved runs')
        sols_raw    = pickle.load(open(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name   + '_all.pickle'), 'rb'))
        # sols        = pickle.load(open(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name   + '.pickle'), 'rb'))
        percentiles = pickle.load(open(os.path.join(os.path.dirname(cwd),'saved_runs/' + percentile_name + '.pickle'), 'rb'))

    if mode=='test':
        return sols_raw

    # example of generating csv (currently for medium R0 value and for certain percentiles)
    # might want to include all age structure info for percentiles as well as medium R0
    if save_csv:
        print('saving outputs as csv files')

        U95 = percentiles[0]
        L95 = percentiles[3]
        median = percentiles[4]
        # print(median.shape)
        # print(sols)
        
        # generate_csv(median,population_frame,'median_'+solution_name,input_type='percentile',time_vec=sols[0]['t'])
        # generate_csv(L95,population_frame,'L95_'+solution_name,input_type='percentile',time_vec=sols[0]['t'])
        # generate_csv(U95,population_frame,'U95_'+solution_name,input_type='percentile',time_vec=sols[0]['t'])

        generate_csv(sols_raw,population_frame,'all_R0_'+solution_name,input_type='raw')
        # generate_csv(sols,population_frame,'middle_R0_'+solution_name,input_type='solution')




    ## ----------------------------------------------------------------------------------------
    if plot_output:
        # plots - change outputs via these below
        print('generating dynamic plots in plotly')
        multiple_categories_to_plot    = ['E','A','I','R','H','C','D','O'] # categories to plot
        single_category_to_plot        = 'D'           # categories to plot in final 3 plots

        # plot graphs
        fig_multi_lines   = go.Figure(  figure_generator(sols,multiple_categories_to_plot,population,population_frame))   # plot with lots of lines
        fig_age_structure    = go.Figure(age_structure_plot(sols,single_category_to_plot,    population,population_frame))   # age structure
        fig_bar_chart     = go.Figure(  stacked_bar_plot(sols,single_category_to_plot,    population,population_frame))                      # bar chart (age structure)
        fig_uncertainty   = go.Figure(  uncertainty_plot(sols,single_category_to_plot,    population,population_frame,percentiles)) # uncertainty


        # view
        fig_multi_lines.show()
        fig_age_structure.show()
        fig_bar_chart.show()
        fig_uncertainty.show()


        # save
        if save_plots:
            plotString = "_%s_%s" %(categories[single_category_to_plot]['longname'],param_string)
            fig_multi_lines.write_image(   os.path.join(os.path.dirname(cwd), "Figs/Disease_progress_%s.png" % param_string))
            fig_age_structure.write_image( os.path.join(os.path.dirname(cwd), "Figs/Age_structure" + plotString + ".png" ))
            fig_bar_chart.write_image(     os.path.join(os.path.dirname(cwd), "Figs/Age_structure_(bar_chart)" + plotString + ".png" ))
            fig_uncertainty.write_image(   os.path.join(os.path.dirname(cwd), "Figs/Uncertainty" + plotString + ".png" ))

    return None

if __name__=='__main__':
    _=run_simulation()
