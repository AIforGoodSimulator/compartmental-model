#for plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from ipywidgets import fixed,interactive,Layout
from preprocess import read_preprocess_file,load_interventions,intervention_dict
import ipywidgets as widgets
from textwrap import wrap
import numpy as np
import pandas as pd

#---------------------------------------------------------------------------
# Monkey patch seaborn to color the error bands with maxmin and iqr options
#---------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from seaborn import utils
from seaborn.utils import (categorical_order, get_color_cycle, ci_to_errsize,
                    remove_na, locator_to_legend_entries)
from seaborn.algorithms import bootstrap
from seaborn.palettes import (color_palette, cubehelix_palette,
                       _parse_cubehelix_args, QUAL_PALETTES)
from seaborn.axisgrid import FacetGrid, _facet_docs

class LinePlotter_custom(sns.relational._RelationalPlotter):
    _legend_attributes = ["color", "linewidth", "marker", "dashes"]
    _legend_func = "plot"

    def __init__(self,
                 x=None, y=None, hue=None, size=None, style=None, data=None,
                 palette=None, hue_order=None, hue_norm=None,
                 sizes=None, size_order=None, size_norm=None,
                 dashes=None, markers=None, style_order=None,
                 units=None, estimator=None, ci=None, n_boot=None, seed=None,
                 sort=True, err_style=None, err_kws=None, legend=None):

        plot_data = self.establish_variables(
            x, y, hue, size, style, units, data
        )

        self._default_size_range = (
            np.r_[.5, 2] * mpl.rcParams["lines.linewidth"]
        )

        self.parse_hue(plot_data["hue"], palette, hue_order, hue_norm)
        self.parse_size(plot_data["size"], sizes, size_order, size_norm)
        self.parse_style(plot_data["style"], markers, dashes, style_order)

        self.units = units
        self.estimator = estimator
        self.ci = ci
        self.n_boot = n_boot
        self.seed = seed
        self.sort = sort
        self.err_style = err_style
        self.err_kws = {} if err_kws is None else err_kws

        self.legend = legend
    def aggregate(self, vals, grouper, units=None):
        """Compute an estimate and confidence interval using grouper."""
        func = self.estimator
        ci = self.ci
        n_boot = self.n_boot
        seed = self.seed

        # Define a "null" CI for when we only have one value
        null_ci = pd.Series(index=["low", "high"], dtype=np.float)

        # Function to bootstrap in the context of a pandas group by
        def bootstrapped_cis(vals):

            if len(vals) <= 1:
                return null_ci

            boots = bootstrap(vals, func=func, n_boot=n_boot, seed=seed)
            cis = utils.ci(boots, ci)
            return pd.Series(cis, ["low", "high"])

        # Group and get the aggregation estimate
        grouped = vals.groupby(grouper, sort=self.sort)
        est = grouped.agg(func)

        # Exit early if we don't want a confidence interval
        if ci is None:
            return est.index, est, None

        # Compute the error bar extents
        if ci == "sd":
            sd = grouped.std()
            cis = pd.DataFrame(np.c_[est - sd, est + sd],
                               index=est.index,
                               columns=["low", "high"]).stack()
        elif ci=='maxmin':
            cis = pd.DataFrame(np.c_[grouped.min(), grouped.max()],
                               index=est.index,
                               columns=["low", "high"]).stack()
        elif ci=='iqr':
            cis = pd.DataFrame(np.c_[grouped.quantile(0.25), grouped.quantile(0.75)],
                               index=est.index,
                               columns=["low", "high"]).stack()
        else:
            cis = grouped.apply(bootstrapped_cis)

        # Unpack the CIs into "wide" format for plotting
        if cis.notnull().any():
            cis = cis.unstack().reindex(est.index)
        else:
            cis = None

        return est.index, est, cis
    def plot(self, ax, kws):
        """Draw the plot onto an axes, passing matplotlib kwargs."""

        # Draw a test plot, using the passed in kwargs. The goal here is to
        # honor both (a) the current state of the plot cycler and (b) the
        # specified kwargs on all the lines we will draw, overriding when
        # relevant with the data semantics. Note that we won't cycle
        # internally; in other words, if ``hue`` is not used, all elements will
        # have the same color, but they will have the color that you would have
        # gotten from the corresponding matplotlib function, and calling the
        # function will advance the axes property cycle.

        scout, = ax.plot([], [], **kws)

        orig_color = kws.pop("color", scout.get_color())
        orig_marker = kws.pop("marker", scout.get_marker())
        orig_linewidth = kws.pop("linewidth",
                                 kws.pop("lw", scout.get_linewidth()))

        orig_dashes = kws.pop("dashes", "")

        kws.setdefault("markeredgewidth", kws.pop("mew", .75))
        kws.setdefault("markeredgecolor", kws.pop("mec", "w"))

        scout.remove()

        # Set default error kwargs
        err_kws = self.err_kws.copy()
        if self.err_style == "band":
            err_kws.setdefault("alpha", .2)
        elif self.err_style == "bars":
            pass
        elif self.err_style is not None:
            err = "`err_style` must be 'band' or 'bars', not {}"
            raise ValueError(err.format(self.err_style))

        # Loop over the semantic subsets and draw a line for each

        for semantics, data in self.subset_data():

            hue, size, style = semantics
            x, y, units = data["x"], data["y"], data.get("units", None)

            if self.estimator is not None:
                if self.units is not None:
                    err = "estimator must be None when specifying units"
                    raise ValueError(err)
                x, y, y_ci = self.aggregate(y, x, units)
            else:
                y_ci = None

            kws["color"] = self.palette.get(hue, orig_color)
            kws["dashes"] = self.dashes.get(style, orig_dashes)
            kws["marker"] = self.markers.get(style, orig_marker)
            kws["linewidth"] = self.sizes.get(size, orig_linewidth)

            line, = ax.plot([], [], **kws)
            line_color = line.get_color()
            line_alpha = line.get_alpha()
            line_capstyle = line.get_solid_capstyle()
            line.remove()

            # --- Draw the main line

            x, y = np.asarray(x), np.asarray(y)

            if self.units is None:
                line, = ax.plot(x, y, **kws)

            else:
                for u in units.unique():
                    rows = np.asarray(units == u)
                    ax.plot(x[rows], y[rows], **kws)

            # --- Draw the confidence intervals

            if y_ci is not None:

                low, high = np.asarray(y_ci["low"]), np.asarray(y_ci["high"])

                if self.err_style == "band":

                    ax.fill_between(x, low, high, color=line_color, **err_kws)

                elif self.err_style == "bars":

                    y_err = ci_to_errsize((low, high), y)
                    ebars = ax.errorbar(x, y, y_err, linestyle="",
                                        color=line_color, alpha=line_alpha,
                                        **err_kws)

                    # Set the capstyle properly on the error bars
                    for obj in ebars.get_children():
                        try:
                            obj.set_capstyle(line_capstyle)
                        except AttributeError:
                            # Does not exist on mpl < 2.2
                            pass

        # Finalize the axes details
        self.label_axes(ax)
        if self.legend:
            self.add_legend_data(ax)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
                
sns.relational._LinePlotter=LinePlotter_custom
#---------------------------------------------------------------------------
# Monkey patch seaborn to color the error bands with maxmin and iqr options
#---------------------------------------------------------------------------

# def plot_by_age(column,df):
# 	fig, ax = plt.subplots(1, 9, sharex='col', sharey='row',figsize=(20,5),constrained_layout=True)
# 	for key in df.columns:
# 		if key==column:
# 			sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[0])
# 			ax[0].set_title('all ages')
# 		elif '0-9' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[1])
# 				ax[1].set_title('<9 years')
# 		elif '10-19' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[2])
# 				ax[2].set_title('10-19 years')
# 		elif '20-29' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[3])
# 				ax[3].set_title('20-29 years')
# 		elif '30-39' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[4])
# 				ax[4].set_title('30-39 years')
# 		elif '40-49' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[5])
# 				ax[5].set_title('40-49 years')
# 		elif '50-59' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[6])
# 				ax[6].set_title('50-59 years')
# 		elif '60-69' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[7])
# 				ax[7].set_title('60-69 years')
# 		elif '70+' in key:
# 			if key.startswith(column):
# 				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[8])
# 				ax[8].set_title('70+ years')

# def plot_by_age_interactive(plot_by_age,df):
# 	w = interactive(plot_by_age,column=widgets.Dropdown(
# 				options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],
# 				value='Infected (symptomatic)',
# 				description='Category:'
# 				),df=fixed(df))
# 	words = widgets.Label('Plot the do nothing scenario in four different categories split by age groups')
# 	container=widgets.VBox([words,w])
# 	container.layout.width = '100%'
# 	container.layout.border = '2px solid grey'
# 	container.layout.justify_content = 'space-around'
# 	container.layout.align_items = 'center'
# 	return container

def plot_all(df):
	fig, ax = plt.subplots(1,4, sharex='col',figsize=(16,9),constrained_layout=True)

	columns_to_plot=['Infected (symptomatic)','Hospitalised','Critical','Deaths']
	i=0
	fontdict={'fontsize':15}
	for column in columns_to_plot:
		sns.lineplot(x="Time", y=column,ci='iqr',data=df,ax=ax[i],estimator=np.median)
		ax[i].set_title(column,fontdict)
		i+=1
	fig.suptitle('Fig.1 Plots of changes in symptomatically infected cases, hopitalisation cases, critical care cases and death incidents over the course of simulation datys',
		fontsize=20)

def plot_by_age_all(df):
	fig, ax = plt.subplots(4, 9, sharex='col', sharey='row',figsize=(32,18),constrained_layout=True)
	columns_to_plot=['Infected (symptomatic)','Hospitalised','Critical','Deaths']
	i=0
	fontdict={'fontsize':15}
	for column in columns_to_plot:
		for key in df.columns:
			if key==column:
				sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,0],estimator=np.median)
				ax[i,0].set_title('all ages',fontdict)
				ax[i,0].set_xlabel('Time',fontsize=15)
			elif '0-9' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,1],estimator=np.median)
					ax[i,1].set_title('<9 years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
			elif '10-19' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,2],estimator=np.median)
					ax[i,2].set_title('10-19 years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
			elif '20-29' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,3],estimator=np.median)
					ax[i,3].set_title('20-29 years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
			elif '30-39' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,4],estimator=np.median)
					ax[i,4].set_title('30-39 years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
			elif '40-49' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,5],estimator=np.median)
					ax[i,5].set_title('40-49 years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
			elif '50-59' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,6],estimator=np.median)
					ax[i,6].set_title('50-59 years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
			elif '60-69' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,7],estimator=np.median)
					ax[i,7].set_title('60-69 years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
			elif '70+' in key:
				if key.startswith(column):
					sns.lineplot(x="Time", y=key,ci='iqr',data=df,ax=ax[i,8],estimator=np.median)
					ax[i,8].set_title('70+ years',fontdict)
					ax[i,0].set_xlabel('Time',fontsize=15)
		i+=1
	fig.suptitle('Fig.1 Plots of changes in symptomatically infected cases, hopitalisation cases, critical care cases and death incidents over the course of simulation datys',
		fontsize=20)


def plot_one_intervention_horizontal(column,baseline,one_intervention_dict):
	fig, ax = plt.subplots(1, len(one_intervention_dict)+1, sharex='col', sharey='row',figsize=(4*len(one_intervention_dict),5))
	sns.lineplot(x="Time", y=column,ci='iqr',data=baseline,ax=ax[0],estimator=np.median)
	ax[0].set_title('Baseline')
	i=1
	for key,value in one_intervention_dict.items():
		sns.lineplot(x="Time", y=column,ci='iqr',data=value,ax=ax[i],estimator=np.median)
		ax[i].set_title("\n".join(wrap(intervention_dict[key], 30)))
		i+=1

def plot_one_intervention_horizontal_interactive(plot_one_intervention_horizontal,baseline):
	folder_path='./model_outcomes/one_intervention/'
	one_intervention_dict=load_interventions(folder_path)
	#sort the collection of interventions by their keys
	one_intervention_dict={k: v for k, v in sorted(one_intervention_dict.items(), key=lambda item: item[0])}
	w = interactive(plot_one_intervention_horizontal,
					column=widgets.Select(
					options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],
					value='Infected (symptomatic)',
					description='Category:'
					),
					baseline=fixed(baseline),
					one_intervention_dict=fixed(one_intervention_dict))
	words = widgets.Label('Contrast the peak case count between do nothing and a single intervention strategy in place')
	container=widgets.VBox([words,w])
	container.layout.width = '100%'
	container.layout.border = '2px solid grey'
	container.layout.justify_content = 'space-around'
	container.layout.align_items = 'center'
	return container

def plot_one_intervention_vertical(column,one_intervention_dict,top_5=False):
	peak_values={}
	for key,value in one_intervention_dict.items():
		peak_values[key]=value.groupby(['R0','latentRate','removalRate','hospRate','deathRateICU','deathRateNoIcu'])[column].max().mean()
	peak_values_sorted={k: v for k, v in sorted(peak_values.items(), key=lambda item: item[1])}
	if not top_5:
		fig, ax = plt.subplots(len(one_intervention_dict), 1, sharex='row', sharey=True,figsize=(15,len(one_intervention_dict)*3))
	elif top_5:
		fig, ax = plt.subplots(5, 1, sharex='row', sharey=True,figsize=(32,18))
	i=0
	for key in peak_values_sorted.keys():
		sns.lineplot(x="Time", y=column,ci='iqr',data=one_intervention_dict[key],ax=ax[i],estimator=np.median)
		ax[i].text(0.5,0.5,intervention_dict[key],verticalalignment='center',horizontalalignment='center',fontsize=15,color='green',transform=ax[i].transAxes)
		ax[i].set_ylabel('')    
		i+=1
		if top_5:
			if i==5:
				break

def plot_one_intervention_vertical_interactive(plot_one_intervention_vertical):
	folder_path='./model_outcomes/one_intervention/'
	one_intervention_dict=load_interventions(folder_path)
	w = interactive(plot_one_intervention_vertical,
					column=widgets.Select(
					options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],
					value='Infected (symptomatic)',
					description='Category:'
					),
					one_intervention_dict=fixed(one_intervention_dict),
					top_5=fixed(True))
	words = widgets.Label('Plot the case counts when one of the intervention is in place and the intervention plots are places by ascending order of effectiveness in reducing peak case counts')
	container=widgets.VBox([words,w])
	container.layout.width = '100%'
	container.layout.border = '2px solid grey'
	container.layout.justify_content = 'space-around'
	container.layout.align_items = 'center'
	return w

def plot_intervention_comparison(scenarioDict,firstIntervention,secondIntervention,selectedCategory):
	fig, ax = plt.subplots(1, 2, sharex='col', sharey='row',figsize=(25,5))
	sns.lineplot(x="Time", y=selectedCategory,ci='iqr',data=scenarioDict[firstIntervention],ax=ax[0],estimator=np.median)
	ax[0].set_title(firstIntervention)
	sns.lineplot(x="Time", y=selectedCategory,ci='iqr',data=scenarioDict[secondIntervention],ax=ax[1],estimator=np.median)
	ax[1].set_title(secondIntervention)

def plot_intervention_comparison_interactive(plot_intervention_comparison,baseline,folder_path='./model_outcomes/one_intervention/'):
	selectedInterventions=load_interventions(folder_path)
	selectedInterventions['do nothing']=baseline
	first = widgets.Dropdown(options=selectedInterventions.keys(),value='do nothing',description='Compare:',disabled=False)
	second = widgets.Dropdown(options=selectedInterventions.keys(),description='With:',disabled=False)
	category = widgets.Dropdown(options=['Infected (symptomatic)','Hospitalised','Critical','Deaths'],description='Category:',disabled=False)
	w = interactive(plot_intervention_comparison,scenarioDict=fixed(selectedInterventions),firstIntervention=first,secondIntervention=second,
				selectedCategory=category)
	controls = widgets.HBox(w.children[:-1], layout = Layout(flex_flow='row wrap'))
	output = w.children[-1]
	words = widgets.Label('Compare two different intervention strategies or compare a particular intervention strategy with do nothing scenario')
	container=widgets.VBox([words,controls,output])
	container.layout.width = '100%'
	container.layout.border = '2px solid grey'
	container.layout.justify_content = 'space-around'
	container.layout.align_items = 'center'
	return container

def plot_hygiene_intervention_horizontal(baseline,column='Infected (symptomatic)',timing=True):
	folder_path='./model_outcomes/one_intervention/'
	if timing:
		selectedInterventions=load_interventions(folder_path,prefix='hygiene0.7')
		selectedInterventions={k: v for k, v in sorted(selectedInterventions.items(), key=lambda item: int(item[0].split('-')[1]))}
		return plot_one_intervention_horizontal(column,baseline,selectedInterventions)
	else:
		selectedInterventions=load_interventions(folder_path,prefix='hygiene',suffix='200')
		selectedInterventions={k: v for k, v in sorted(selectedInterventions.items(), key=lambda item: float(item[0].split('-')[0].split('giene')[1]))}
		return plot_one_intervention_horizontal(column,baseline,selectedInterventions)

def plot_iso_intervention_horizontal(baseline,column='Infected (symptomatic)',timing=True):
	folder_path='./model_outcomes/one_intervention/'
	if timing:
		selectedInterventions=load_interventions(folder_path,prefix='isolate50')
		return plot_one_intervention_horizontal(column,baseline,selectedInterventions)
	else:
		selectedInterventions_100=load_interventions(folder_path,prefix='isolate100')
		selectedInterventions_50=load_interventions(folder_path,prefix='isolate50',suffix='40')
		selectedInterventions_10=load_interventions(folder_path,prefix='isolate10')
		selectedInterventions = {**selectedInterventions_100, **selectedInterventions_50,**selectedInterventions_10}
		return plot_one_intervention_horizontal(column,baseline,selectedInterventions)

def plot_onetype_intervention_horizontal(baseline,prefix,column='Infected (symptomatic)'):
	folder_path='./model_outcomes/one_intervention/'
	selectedInterventions=load_interventions(folder_path,prefix=prefix)
	return plot_one_intervention_horizontal(column,baseline,selectedInterventions)
	
def plot_onetype_intervention_vertical(prefix,column='Infected (symptomatic)',top_5=False):
	folder_path='./model_outcomes/one_intervention/'
	selectedInterventions=load_interventions(folder_path,prefix=prefix)
	return plot_one_intervention_vertical(column,selectedInterventions,top_5=top_5)
	

