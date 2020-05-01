# Compartment model from AIforGood Simulator

AIforGood Simulator aims to provide epidimology modelling help to humanitarian NGOs who are working to contain coronavirus outbreak in refugee camps. Please fill out the contact form on [our website](https://www.aiforgoodsimulator.com)if you want to get in touch or volunteer with us.

# Model description

(pending a write up from Nick)

# Running the repo

clone the repo and then change to the git directory and then:

```python
pip install -r requirements.txt
```

Here is the workflow in generating CSVs for the report

1. Prepare population level data

⋅⋅⋅Go to **Parameters** folder and open up **Prepare camp parameters.ipynb** where you use the **camp_params_template.csv** modify it with the age data from the camp of interest and save it as a seperate csv file

2. Prepare contact matrix data

⋅⋅⋅Again in the **Parameters** folder, open up **Prepare contact matrix.ipynb** where you use the **Contact_matrix_wuhan.csv** modify it with the csv just being created on camp population level data. Save it as a seperate csv file

3. Change the directories pointing towards the population level data and contact matrix data
* go to **Scripts** and open **initalise_parameters.py** and change the csv import for population data
* go to **Scripts** and open **functions.py** and change the csv import for contact matrix

4. Define the config file used to run the experiment

⋅⋅⋅Can modify the **config.py** under the **Scripts** directory or go into **configs** folder to set up the file and go to **run_model.py** to adjust for the import.

For example:
```python
#running simulation with no intervention
from configs.baseline import camp, population_frame, population, control_dict
```

```python
#running simulation with improving hygiene as an intervention
from configs.hygiene import camp, population_frame, population, control_dict
```

5. Set up the run_model,py (change settings on *load*,*save*,*plot_outputs* and *save_plots*) and run the model from command line
⋅⋅⋅ make sure your current directory is in **Scripts**

```python
python run_model.py
```

6. Retrieve outputs from **CSV_output** folder

⋅⋅⋅ These are the csvs that we will use to make a plot
