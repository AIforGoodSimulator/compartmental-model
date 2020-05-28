from initialise_parameters import preparePopulationFrame, params

# camp
camp = 'Moria'
population_frame, population = preparePopulationFrame(camp)

# from github issue
# if not used, set timings to e.g. [0,0] or any other interval of 0 length or outside caluclated window

control_dict = dict( # contains our 6 different control options. Can choose any combination of these 6. Suggest limit to all occuring at similar times

    # 1
    # if True, reduces transmission rate by params.better_hygiene
    better_hygiene = dict(value = params.better_hygiene,
                        timing = [0,0]),

    ICU_capacity = dict(value = 6/population),
                        
    # 4
    # move symptomatic cases off site
    remove_symptomatic = dict(rate = 10/population,  # people per day
                            timing = [0,0]),

    # 5
    # partially separate low and high risk
    # (for now) assumed that if do this, do for entire course of epidemic
    shielding = dict(used= False), 

    # 6
    # move uninfected high risk people off site
    # in total in the age brackets there are ~ people 620
    remove_high_risk = dict(rate = 100/population,  # people per day
                            n_categories_removed = 2, # remove oldest n categories
                            timing = [0,6])
    #amount of people per day 20 50 100
    #timing from 0-30 0-12 0-6

)