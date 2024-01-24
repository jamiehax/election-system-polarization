import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def agent_locations(agent_data, time_steps):

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
    model_data = pd.read_csv('data/model_data.csv')
    agent_data = pd.read_csv('data/agent_data.csv')

    # make visualizations output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'visualizations')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # PLOT AGENT LOCATIONS   
    time_steps = [0, 500, 1000, 2000, 4000]
    agent_locations(agent_data, time_steps)