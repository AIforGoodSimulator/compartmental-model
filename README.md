# Compartment model from AIforGood Simulator

[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

AIforGood Simulator aims to provide epidimology modelling help to humanitarian NGOs who are working to contain coronavirus outbreak in refugee camps. Please fill out the contact form on [our website](https://www.aiforgoodsimulator.com) if you want to get in touch or volunteer with us.

# Model description

(pending a write up from Nick)

# Running the repo

clone the repo and then change to the git directory and then:

```python
pip install -r requirements.txt
```

Here is the workflow in generating CSVs for the report

1. Prepare population level data

⋅⋅⋅Go to **Parameters** folder and open up **camp_params.csv** where you use the input the camp information and fill up the proportion for each age bracket and save it. Don't modify other camps information!

2. Prepare contact matrix data (optional)

⋅⋅⋅Again in the **Parameters** folder, open up **GenerateRandomParams.R** where you can synthesize the contact matrix for the camp of interest. Save it as a seperate csv file.

3. To run the baseline experiment
* go to **Scripts/fucntions.py** and find the GenerateInfectionMatrix function to change where it is pointing to the contact matrix csv file
* define the config file used to run the experiment and set up the name of the camp correctly as set out in the camp_params.csv
* for a single run and just run the run_model.py file but notice the control parameters at the top (maybe we can build in some command line control here rather than modifying the py file every time)
```python
python run_model.py
```

4. To run different interventions

* Modify the **config.py** under the **Scripts** directory to make copies with it along with appropriate naming convention

For example:
```python
#running simulation with no intervention
from configs.baseline import camp, population_frame, population, control_dict
```

```python
#running simulation with improving hygiene as an intervention
from configs.hygiene import camp, population_frame, population, control_dict
```

* Can use the **run_model_intervention.py** file to run interventions in bulk (maybe we can build in multiprocessing here to speed up the run time)

5. Retrieve outputs from **CSV_output** folder

⋅⋅⋅ These are the csvs that we will use to make a plot
