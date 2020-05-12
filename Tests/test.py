#this is where the file where all tests will come together

#Here I am going to dump a few testing ideas to start with

#tracking population total over the course of simulation to see if the total numbers are right
#tracking the population within each age compartment to see if the number remain constant with the time step passing to observe is there is any abnormal flow patter

# Bring your packages onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Scripts')))

# Now do your import
from run_model import *

tol=1e-5

solution=run_simulation(mode='test')

#here is a test class taht runs some simple tests on the output dict solution
class TestBasicOutput:
	def test_num_runs(self,solution=solution):
		#test the number of R0 simulated 
		assert len(solution)==20

	def test_r0_range(self,solution=solution,params=params):
		#test if the biggest R0 and the smallest R0 simulated in the runs are in the specified range
		real_numbers=[num.real for num in list(solution.keys())]
		imag_numbers=[num.imag for num in list(solution.keys())]
		assert min(real_numbers)==params.R_0_list[0]
		assert max(real_numbers)==params.R_0_list[-1]
		assert sum(np.array(imag_numbers)!=0)==0

	def test_outputframes(self,solution=solution,params=params,population_frame=population_frame):
		#assemble all the runs' data into data frames and use hardcoded column numbers to see if the totals within each compartment agree
		category_map = {    '0':  'S',
						'1':  'E',
						'2':  'I',
						'3':  'A',
						'4':  'R',
						'5':  'H',
						'6':  'C',
						'7':  'D',
						'8':  'O',
						'9':  'CS',
						'10': 'CE',
						'11': 'CI',
						'12': 'CA',
						'13': 'CR',
						'14': 'CH',
						'15': 'CC',
						'16': 'CD',
						'17': 'CO',
						'18': 'Ninf',
						}
		for data in solution.values():
			csv_sol = np.transpose(data['y']) # age structured
			solution_csv = pd.DataFrame(csv_sol)
			# setup column names
			col_names = []
			number_categories_with_age = csv_sol.shape[1]
			for i in range(number_categories_with_age):
				ii = i % params.number_compartments
				jj = floor(i/params.number_compartments)
				col_names.append(categories[category_map[str(ii)]]['longname'] +  ': ' + str(np.asarray(population_frame.Age)[jj]) )

			solution_csv.columns = col_names
			solution_csv['Time'] = data['t']

			for j in range(len(categories.keys())): # params.number_compartments
				solution_csv[categories[category_map[str(j)]]['longname']] = data['y_plot'][j] # summary/non age-structured

			assert abs(100-sum(population_frame.Population))<tol
			#sum across all the people across age compartments to see if that equals to all the people in different disease department
			assert sum(np.absolute(solution_csv.loc[:,solution_csv.columns[:72]].sum(axis=1)-solution_csv.loc[:,solution_csv.columns[73:82]].sum(axis=1))<tol)==len(solution_csv)
			#make sure the changes are correct
			assert np.all(solution_csv.loc[1:,solution_csv.columns[82:-1]].values==solution_csv.loc[:,solution_csv.columns[73:82]].diff().loc[1:,:].values)==True
