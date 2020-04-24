from initialise_parameters import preparePopulationFrame
import numpy as np

control_type = 'No control'
camp = 'Camp_2'
timings = [10,100] # control timings
taken_offsite_rate = 100 # people per day
remove_high_risk = 100 # people per day
population_frame, population = preparePopulationFrame(camp)




##
# create dict for removal
# Ages = np.asarray(population_frame.Age)
# remove_people = [remove_high_risk if i ==len(Ages)-1 else 0 for i in range(len(Ages))]

# remove_people = {}
# for i, age in enumerate(Ages):
#     remove_people[age] = removal[i]

