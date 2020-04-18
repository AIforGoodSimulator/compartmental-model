import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from flask import Flask
from gevent.pywsgi import WSGIServer
import pandas as pd
from math import floor, ceil, exp
from parameters_cov_AI import params, parameter_csv, preparePopulationFrame
import numpy as np
import plotly.graph_objects as go
# from plotly.validators.scatter.marker import SymbolValidator
import copy
from cov_functions_AI import simulator
import flask
import datetime
import json

from app_AI import index, longname, control_data, figure_generator, age_structure_plot, stacked_bar_plot


def find_sol2(preset,timings,camp): # gives solution as well as upper and lower bounds
    
    t_stop = 200

    beta_factor = np.float(control_data.Value[control_data.Name==preset])

    population_frame, population = preparePopulationFrame(camp)
    
    infection_matrix = np.ones((population_frame.shape[0],population_frame.shape[0]))
    beta_list = np.linspace(params.beta_list[0],params.beta_list[2],20)
    sols = []
    for beta in beta_list:
        sols.append(simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,control_time=timings,beta=beta,beta_factor=beta_factor))

    n_time_points = len(sols[0]['t'])

    y_plot = np.zeros((len(longname.keys()), len(sols) , n_time_points ))

    for k, sol in enumerate(sols):
        for name in longname.keys():
            sol['y'] = np.asarray(sol['y'])

            y_plot[index[name],k,:] = sol['y'][index[name],:]
            for i in range(1, population_frame.shape[0]): # age_categories
                y_plot[index[name],k,:] = y_plot[index[name],k,:] + sol['y'][index[name] + i*params.number_compartments,:]


    y_min = np.zeros((params.number_compartments,n_time_points))
    y_max = np.zeros((params.number_compartments,n_time_points))

    for name in longname.keys():
        y_min[index[name],:] = [min(y_plot[index[name],:,i]) for i in range(n_time_points)]
        y_max[index[name],:] = [max(y_plot[index[name],:,i]) for i in range(n_time_points)]

    sols_out = []
    sols_out.append(simulator().run_model(T_stop=t_stop,infection_matrix=infection_matrix,population=population,population_frame=population_frame,control_time=timings,beta=params.beta_list[1],beta_factor=beta_factor))
    
    return sols_out, [y_min,y_max]








# find solutions
preset = 'No control'
# print([control_data.Name==preset])
camp = 'Camp_1'
timings = [10,100] # control timings
sols, upper_lower_bounds =find_sol2(preset, timings, camp) # returns solution for middle R0 and then minimum and maximum values by scanning across a range defined by low and high R0


cats = ['I','R','D'] # categories to plot
cats2 = 'D' # categories to plot in final 3 plots


population_frame, population = preparePopulationFrame(camp)
# preset = control_data.Name[preset]

no_control = False
if preset=='No control':
    no_control = True

# plot graphs
fig    = go.Figure(figure_generator(sols,cats,population,population_frame,timings,no_control)) # plot with lots of lines
fig_U  = go.Figure(figure_generator(sols,cats2,population,population_frame,timings,no_control,upper_lower_bounds)) # uncertainty
fig2   = go.Figure(age_structure_plot(sols,cats2,population,population_frame,timings,no_control)) # age structure
fig3   = go.Figure(stacked_bar_plot(sols,cats2,population,population_frame)) # bar chart (age structure)

fig.show()
fig_U.show()
fig2.show()
fig3.show()