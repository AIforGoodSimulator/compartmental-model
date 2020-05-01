from initialise_parameters import preparePopulationFrame, params
import numpy as np

# camp
camp = 'Moria'
#need to put the population frame in use here too for testing purposes
population_frame, population = preparePopulationFrame(camp)

# from github issue
# if not used, set timings to e.g. [0,0] or any other interval of 0 length or outside caluclated window

control_dict = dict( # contains our 6 different control options. Can choose any combination of these 6. Suggest limit to all occuring at similar times

    # 1
    # if True, reduces transmission rate by params.better_hygiene
    better_hygiene = dict(value = params.better_hygiene,
                        timing = [10,200]),

    ICU_capacity = dict(value = 20/population),
                        
    # 4
    # move symptomatic cases off site
    remove_symptomatic = dict(rate = 10/population,  # people per day
                            timing = [10,200]),

    # 5
    # partially separate low and high risk
    # (for now) assumed that if do this, do for entire course of epidemic
    shielding = dict(used= True), 

    # 6
    # move uninfected high risk people off site
    remove_high_risk = dict(rate = 20/population,  # people per day
                            timing = [0,60])

)