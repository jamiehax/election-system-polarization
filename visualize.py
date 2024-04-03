import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from model import ElectionSystem
from run import save_data
import numpy as np


def main():
    model_data = pd.read_csv('data/model_data.csv')
    agent_data = pd.read_csv('data/agent_data.csv')
    agent_data_bc = pd.read_csv('data/agent_data_bc.csv')
    agent_data_avg = pd.read_csv('data/agent_data_avg.csv')

    # make visualizations output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # PLOT PARAMETER SWEEP
    parameter_sweep()

    # PLOT AGENT LOCATIONS IN 1D
    #agent_locations_1d(agent_data)

    # PLOT OPINION VARIANCE IN 1D
    #opinion_variance_1d(agent_data_bc, agent_data_avg, 3000)

    # # PLOT AGENT LOCATIONS IN 2D
    # time_steps = [0, 500, 1000, 3000, 10000]
    # agent_locations_2d(agent_data, time_steps)

    # # PLOT OPINION VARIANCE IN 2D
    # opinion_variance(agent_data, time_steps=10000)


def parameter_sweep():

    # create sweeps output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations/sweeps')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # sweep data frame
    num_sweeps = 18
    responsiveness_list = np.linspace(0.1, 1, num_sweeps, endpoint=False)
    tolerance_list = np.linspace(0.1, 1, num_sweeps, endpoint=False)
    responsiveness_list = np.round(responsiveness_list, decimals=2)
    tolerance_list = np.round(tolerance_list, decimals=2)
    df = pd.DataFrame(columns=['responsiveness', 'tolerance', 'variance'])

    # set figure dimensions
    plt.figure(
        figsize=(6, 6), 
        dpi = 600
    )

    for responsiveness in tqdm(responsiveness_list):
        for tolerance in tolerance_list:

            model = ElectionSystem(
                    #seed=0,
                    num_voters=100,
                    num_voters_to_activate=1,
                    initial_num_candidates=3,
                    min_candidates=3,
                    max_candidates=5,
                    term_limit=2,
                    num_opinions=1,
                    election_system='plurality', # plurality, rc, score
                    voter_voter_interaction_fn='ar', # avg, bc, bc1, ar
                    voter_candidate_interaction_fn='ar', # avg, bc, bc1, ar
                    voter_noise_factor=0.01,
                    initial_exit_probability=0.33,
                    exit_probability_decrease_factor=0.25,     
                    initial_threshold=tolerance,
                    threshold_increase_factor=0.1,
                    num_candidates_to_benefit=2,
                    num_rounds_before_election=10,
                    mu=0.5, # mu in bounded confidence (controls magnitude of opinion update)
                    radius=0.1, # r in radius of support function (controls inflection point)
                    learning_rate=0.001, # learning rate for candidate gradient ascent
                    gamma=10, # gamma in radius of support function (controls steepness which is effectively variation in voting probabilities)
                    beta=1,
                    second_choice_weight_factor=0.5,
                    exposure=0.2,
                    responsiveness=responsiveness
                )

            num_steps = 100
            for _ in range(num_steps):
                model.step()

            save_data(model)

            agent_data = pd.read_csv('data/agent_data.csv')
            variance = float(agent_data.loc[agent_data['Step'] == num_steps][['opinion1']].var())
            df.loc[df.shape[0]] = [responsiveness, tolerance, variance]

    sns.heatmap(data=df.pivot('responsiveness', 'tolerance', 'variance'), cmap='rocket')
    plt.xlabel('Tolerance')
    plt.ylabel('Responsiveness')
    plt.gca().invert_yaxis()
    plt.title('Heatmap of Variance')

    # save plot
    plt.savefig('visualizations/sweeps/sweep.png')



def opinion_variance_1d(agent_data_bc, agent_data_avg, time_steps):
    
    # create agent_locations output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations/variance')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    variance = {'Step': [], 'Model Type': [], 'var_opinion': []}
    for step in range(time_steps):
        # variance['Step'].append(step)
        # variance['Model Type'].append('Bounded Confidence')
        # var_bc = agent_data_bc.loc[agent_data_bc['Step'] == step]['opinion1'].var()
        # variance['var_opinion'].append(var_bc)
        variance['Step'].append(step)
        variance['Model Type'].append('Averaging')
        var_avg = agent_data_avg.loc[agent_data_avg['Step'] == step]['opinion1'].var()
        variance['var_opinion'].append(var_avg)

    var_df = pd.DataFrame(variance)

    plt.figure(
        figsize=(12, 6), 
        dpi = 600
    )

    # set plot styles
    sns.set_style('ticks')
    colors = {'Bounded Confidence': sns.color_palette('Paired')[1], 'Averaging': sns.color_palette('Paired')[6]}

    # plot
    var_plot = sns.lineplot(
        data=var_df,
        x='Step',
        y='var_opinion',
        hue='Model Type',
        palette=colors
    )

    var_plot.spines[['right', 'top']].set_visible(False)
    var_plot.set(
        xlabel='Time Step',
        ylabel='Variance in Opinion',
        title='Variance in Opinion over Time',
        ylim=(0, 0.1)
    )

    # save plot
    plt.savefig('visualizations/variance/var.png')


def agent_locations_1d(agent_data):

    # create agent_locations output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations/1d_locations')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    # set plot styles
    sns.set_style('ticks')

    plt.figure(
        figsize=(12, 6), 
        dpi = 600
    )

    # set plot styles
    sns.set_style('ticks')
    sizes = {'voter': 0.5, 'candidate': 1.5}
    colors = {'voter': sns.color_palette('Set2')[7], 'candidate': sns.color_palette('rocket')[1]}

    location_plot = sns.lineplot(
        data=agent_data,
        x="Step",
        y="opinion1",
        hue="type",
        palette=colors,
        size='type',
        sizes=sizes,
        units="AgentID",
        estimator=None
    )

    location_plot.spines[['right', 'top']].set_visible(False)
    location_plot.set(
        xlabel='Time Step',
        ylabel='Opinion Value',
        title="Change in Agent's Opinion from Averaging Opinion Updates"
    )

    # save plot
    plt.savefig('visualizations/1d_locations/1dloc.png')


def opinion_variance(agent_data, time_steps):

    # create agent_locations output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations/variance')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    variances = {'Step': [], 'var_opinion1': [], 'var_opinion2': []}
    for step in range(time_steps):
        var = agent_data.loc[agent_data['Step'] == step][['opinion1', 'opinion2']].var()
        variances['Step'].append(step)
        variances['var_opinion1'].append(var[0])
        variances['var_opinion2'].append(var[1])

    var_df = pd.DataFrame(variances)

    plt.figure(
        figsize=(12, 6), 
        dpi = 600
    )

    # set plot styles
    sns.set_style('ticks')

    # plot
    var_plot = sns.lineplot(
        data=var_df,
        x='Step',
        y='var_opinion1',
        label='Opinion 1 Variance'
    )
    var_plot = sns.lineplot(
        data=var_df,
        x='Step',
        y='var_opinion2',
        label='Opinion 2 Variance'
    )

    var_plot.spines[['right', 'top']].set_visible(False)
    var_plot.set(
        xlabel='Time Step',
        ylabel='Variance in Opinion',
        title='Variance in Opinions over Time',
        ylim=(0, 0.1)
    )

    # save plot
    plt.savefig('visualizations/variance/var.png')


def agent_locations_2d(agent_data, time_steps):

    # create agent_locations output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations/agent_locations')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # set plot styles
    sns.set_style('ticks')
    markers = {'voter': 'X', 'candidate': 'o'}
    sizes = {'voter': 50, 'candidate': 150}

    for t in tqdm(time_steps):
        agent_locations = agent_data.loc[agent_data['Step'] == t]

        # CREATE A NEW FIGURE SO PLOTS DO NOT OVERLAY
        plt.figure(
            figsize=(6, 6), 
            dpi = 600
        )

        if t == 0:
            plot = sns.scatterplot(
                data=agent_locations,
                x='opinion1',
                y='opinion2',
                size='type',
                sizes=sizes,
                style='type',
                markers=markers,
                legend=False
            )
        else:
            plot = sns.scatterplot(
                data=agent_locations,
                x='opinion1',
                y='opinion2',
                size='type',
                sizes=sizes,
                style='type',
                markers=markers,
                hue='voted_for',
                legend=False
            )

        #plot.spines[['right', 'top']].set_visible(False)
        plot.set(
            xlabel='Opinion 1',
            ylabel='Opinion 2',
            title='Agent Distribution in Opinion Space at Time Step '+str(t),
            xlim=(0, 1),
            ylim=(0, 1)
        )

        # save plot
        plt.savefig('visualizations/agent_locations/t'+str(t)+'.png')


if __name__ == '__main__':
    main()