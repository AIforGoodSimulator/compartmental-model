
import pandas as pd
from math import floor, ceil, exp
from initialise_parameters import params, parameter_csv, categories
import numpy as np
import plotly.graph_objects as go
from functions import simulator, simulate_range_of_R0s, object_dump, generate_csv
from plotter import figure_generator, age_structure_plot, stacked_bar_plot, uncertainty_plot
from config import camp, population_frame, population, control_dict
import pickle
import os

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
plot_output = True
save_plots  = True # needs plot_output to be True

##----------------------------------------------------------------
param_string = "Camp=%s_hygieneT=%s_remInfRate=%s_remInfT=%s_Shield=%s_RemHrRate=%s_RemHrTime=%s_ICU=%s_" %(camp,
                                                                                                           control_dict['better_hygiene']['timing'],
                                                                                                           round(population*control_dict['remove_symptomatic']['rate'],1),
                                                                                                           control_dict['remove_symptomatic']['timing'],
                                                                                                           control_dict['shielding']['used'],
                                                                                                           round(population*control_dict['remove_high_risk']['rate'],1),
                                                                                                           control_dict['remove_high_risk']['timing'],
                                                                                                           round(population*control_dict['ICU_capacity']['value'],1))
solution_name   = 'Solution_' + param_string    
percentile_name = 'Percentiles_'  + param_string

already_exists_soln       = os.path.isfile(os.path.join(cwd,'saved_runs/' + solution_name))
already_exists_percentile = os.path.isfile(os.path.join(cwd,'saved_runs/' + percentile_name))


if not load or not (already_exists_soln and already_exists_percentile): # generate solution if not wanting to load, or if wanting to load but at least one file missing
    # run model - change inputs via 'config.py'
    print('running the model to produce results')
    sols_raw,sols, percentiles =simulate_range_of_R0s(population_frame, population, control_dict) # returns solution for middle R0 and then minimum and maximum values by scanning across a range defined by low and high R0
    if save:
        object_dump(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name+'_all')  ,  sols_raw)
        object_dump(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name)  ,  sols)
        object_dump(os.path.join(os.path.dirname(cwd),'saved_runs/' + percentile_name),  percentiles)
else:
    print('retrieving results from saved runs')
    sols        = pickle.load(open(os.path.join(os.path.dirname(cwd),'saved_runs/' + solution_name), 'rb'))
    percentiles = pickle.load(open(os.path.join(os.path.dirname(cwd),'saved_runs/' + percentile_name), 'rb'))


# example of generating csv (currently for middle R0 value)
# might also want to do save specific percentiles
# might want to include all age structure info (as currently is)
# or might want to just sum over all age classes to get a total
if save_csv:
    print('saving outputs as csv files')

    U95 = percentiles[0]
    L95 = percentiles[3]
    median = percentiles[4]
    # print(median.shape)
    # print(sols)
    
    generate_csv(median,population_frame,'median_'+solution_name,input_type='percentile',time_vec=sols[0]['t'])
    generate_csv(L95,population_frame,'L95_'+solution_name,input_type='percentile',time_vec=sols[0]['t'])
    generate_csv(U95,population_frame,'U95_'+solution_name,input_type='percentile',time_vec=sols[0]['t'])

    generate_csv(sols,population_frame,'middle_R0_'+solution_name,input_type='solution')




## ----------------------------------------------------------------------------------------
if plot_output:
    # plots - change outputs via these below
    print('generating dynamic plots in plotly')
    multiple_categories_to_plot    = ['E','A','I','R','H','C','D'] # categories to plot
    single_category_to_plot        = 'R'           # categories to plot in final 3 plots

    no_control = True # otherwise plots blue bar

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