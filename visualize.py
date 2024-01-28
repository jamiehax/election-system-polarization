import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def main():
    model_data = pd.read_csv('data/model_data.csv')
    agent_data = pd.read_csv('data/agent_data.csv')

    # make visualizations output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # PLOT AGENT LOCATIONS IN 1D
    agent_locations_1d(agent_data)

    # # PLOT AGENT LOCATIONS IN 2D
    # time_steps = [0, 500, 1000, 3000, 10000]
    # agent_locations_2d(agent_data, time_steps)

    # # PLOT OPINION VARIANCE
    # opinion_variance(agent_data, time_steps=10000)


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
    colors = {'voter': sns.color_palette('rocket')[4], 'candidate': sns.color_palette('rocket')[1]}

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
        ylabel='Opinion',
        title='Agent Opinions over Time'
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