from initialise_parameters import preparePopulationFrame
import numpy as np

# camp
camp = 'Camp_2'
population_frame, population = preparePopulationFrame(camp)

# control timings
timings = [10,100]

# choice of no control, or better hygeine
control_type = 'No control'

# move symptomatic cases off site
taken_offsite_rate = 100 # people per day

# move uninfected high risk people off site
remove_high_risk = 100 # people per day

