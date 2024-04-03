from model import ElectionSystem
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():

    # run all sweeps
    big_boy_sweeps()

    # make all sweep figures
    sweep_figures()


def big_boy_sweeps():
    """
    Run ALL parameter sweeps...
    """

    default_params = {
        #seed:0,
        'num_steps': 100, # number of model steps
        'num_runs': 10, # number of runs to average variance for
        'num_voters': 100,
        'num_voters_to_activate': 1,
        'initial_num_candidates': 3,
        'min_candidates': 3,
        'max_candidates': 5,
        'term_limit': 2,
        'num_opinions': 1,
        'voter_noise_factor': 0.01,
        'initial_exit_probability': 0.33,
        'exit_probability_decrease_factor': 0.25,     
        'initial_threshold': 0.2,
        'threshold_increase_factor': 0.1,
        'num_candidates_to_benefit': 2,
        'num_rounds_before_election': 10,
        'mu': 0.5, # mu in bounded confidence (controls magnitude of opinion update)
        'radius': 0.1, # r in radius of support function (controls inflection point)
        'learning_rate': 0.001, # learning rate for candidate gradient ascent
        'gamma': 10, # gamma in radius of support function (controls steepness which is effectively variation in voting probabilities)
        'beta': 1,
        'second_choice_weight_factor': 0.5,
        'exposure': 0.2,
        'responsiveness': 0.25
    }

    # SWEEPS for EXIT PROBABLITY DECREASE FACTOR and THRESHOLD INCREASE FACTOR
    # Attraction-Repulsion / Plurality
    sweep_exit_probability_decrease_threshold_increase(
        v_v_interaction_fn='ar',
        v_c_interaction_fn='ar',
        election_system='plurality',
        default_params=default_params
    )

    # Bounded Confidence / Plurality
    sweep_exit_probability_decrease_threshold_increase(
        v_v_interaction_fn='bc',
        v_c_interaction_fn='bc1',
        election_system='plurality',
        default_params=default_params
    )
    # Attraction-Repulsion / Ranked Choice
    sweep_exit_probability_decrease_threshold_increase(
        v_v_interaction_fn='ar',
        v_c_interaction_fn='ar',
        election_system='rc',
        default_params=default_params
    )

    # Bounded Confidence / Ranked Choice
    sweep_exit_probability_decrease_threshold_increase(
        v_v_interaction_fn='bc',
        v_c_interaction_fn='bc1',
        election_system='rc',
        default_params=default_params
    )

    # SWEEPS for GAMMA and BETA
    # Attraction-Repulsion / Plurality
    sweep_gamma_beta(
        v_v_interaction_fn='ar',
        v_c_interaction_fn='ar',
        election_system='plurality',
        default_params=default_params
    )

    # Bounded Confidence / Plurality
    sweep_gamma_beta(
        v_v_interaction_fn='bc',
        v_c_interaction_fn='bc1',
        election_system='plurality',
        default_params=default_params
    )
    # Attraction-Repulsion / Ranked Choice
    sweep_gamma_beta(
        v_v_interaction_fn='ar',
        v_c_interaction_fn='ar',
        election_system='rc',
        default_params=default_params
    )

    # Bounded Confidence / Ranked Choice
    sweep_gamma_beta(
        v_v_interaction_fn='bc',
        v_c_interaction_fn='bc1',
        election_system='rc',
        default_params=default_params
    )

    # SWEEPS for RADIUS and SECOND CHOICE WEIGHT FACTOR
    # Attraction-Repulsion / Plurality
    sweep_radius_second_choice_weight(
        v_v_interaction_fn='ar',
        v_c_interaction_fn='ar',
        election_system='plurality',
        default_params=default_params
    )

    # Bounded Confidence / Plurality
    sweep_radius_second_choice_weight(
        v_v_interaction_fn='bc',
        v_c_interaction_fn='bc1',
        election_system='plurality',
        default_params=default_params
    )
    # Attraction-Repulsion / Ranked Choice
    sweep_radius_second_choice_weight(
        v_v_interaction_fn='ar',
        v_c_interaction_fn='ar',
        election_system='rc',
        default_params=default_params
    )

    # Bounded Confidence / Ranked Choice
    sweep_radius_second_choice_weight(
        v_v_interaction_fn='bc',
        v_c_interaction_fn='bc1',
        election_system='rc',
        default_params=default_params
    )


def sweep_figures():

    # make sure storage directory is set
    storage_directory = os.environ.get('STORAGE')
    if not storage_directory:
        raise EnvironmentError("$STORAGE environment variable not set.")

    # find sweeps data directory
    sweeps_path = os.path.join(storage_directory, r'election-simulation/data/sweeps')

    # figure output directory 
    output_dir = 'sweeps'

    # EXIT PROBABILITY DECREASE FACTOR and THRESHOLD INCREASE FACTOR
    # attraction repulsion plurality
    file_name = 'exit_pr_decrease-threshold_increase-ar-plurality.png'
    df_path = os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-ar-plurality.csv')
    make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Attraction-Repulsion Plurality Variance Heatmap')

    # bounded confidence plurality
    file_name = 'exit_pr_decrease-threshold_increase-bc-plurality.png'
    df_path = os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-bc-plurality.csv')
    make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Bounded Confidence Plurality Variance Heatmap')

    # attraction repulsion ranked choice
    file_name = 'exit_pr_decrease-threshold_increase-ar-rc.png'
    df_path = os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-ar-rc.csv')
    make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Attraction-Repulsion Ranked Choice Variance Heatmap')

    # bounded confidence ranked choice
    file_name = 'exit_pr_decrease-threshold_increase-bc-rc.png'
    df_path = os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-bc-rc.csv')
    make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Bounded Confidence Ranked Choice Variance Heatmap')


    # GAMMA and BETA
    # attraction repulsion plurality
    file_name = 'gamma-beta-ar-plurality.png'
    df_path = os.path.join(sweeps_path, 'gamma-beta-ar-plurality.csv')
    make_heatmap(df_path, output_dir, file_name, 'gamma', 'beta', 'Gamma', 'Beta', 'Attraction-Repulsion Plurality Variance Heatmap')

    # bounded confidence plurality
    file_name = 'gamma-beta-bc-plurality.png'
    df_path = os.path.join(sweeps_path, 'gamma-beta-bc-plurality.csv')
    make_heatmap(df_path, output_dir, file_name, 'gamma', 'beta', 'Gamma', 'Beta', 'Bounded Confidence Plurality Variance Heatmap')

    # attraction repulsion ranked choice
    file_name = 'gamma-beta-ar-rc.png'
    df_path = os.path.join(sweeps_path, 'gamma-beta-ar-rc.csv')
    make_heatmap(df_path, output_dir, file_name, 'gamma', 'beta', 'Gamma', 'Beta', 'Attraction-Repulsion Ranked Choice Variance Heatmap')

    # bounded confidence ranked choice
    file_name = 'gamma-beta-bc-plurality.png'
    df_path = os.path.join(sweeps_path, 'gamma-beta-bc-rc.csv')
    make_heatmap(df_path, output_dir, file_name, 'gamma', 'beta', 'Gamma', 'Beta', 'Bounded Confidence Ranked Choice Variance Heatmap')


    # RADIUS and SECOND CHOICE WEIGHT
    # attraction repulsion plurality
    file_name = 'radius-second_choice_weight-ar-plurality.png'
    df_path = os.path.join(sweeps_path, 'radius-second_choice_weight-ar-plurality.csv')
    make_heatmap(df_path, output_dir, file_name, 'radius', 'second_choice_weight', 'Radius', 'Second Choice Weight Factor', 'Attraction-Repulsion Plurality Variance Heatmap')

    # bounded confidence plurality
    file_name = 'radius-second_choice_weight-bc-plurality.png'
    df_path = os.path.join(sweeps_path, 'radius-second_choice_weight-bc-plurality.csv')
    make_heatmap(df_path, output_dir, file_name, 'radius', 'second_choice_weight', 'Radius', 'Second Choice Weight Factor', 'Bounded Confidence Plurality Variance Heatmap')

    # attraction repulsion ranked choice
    file_name = 'radius-second_choice_weight-ar-rc.png'
    df_path = os.path.join(sweeps_path, 'radius-second_choice_weight-ar-rc.csv')
    make_heatmap(df_path, output_dir, file_name, 'radius', 'second_choice_weight', 'Radius', 'Second Choice Weight Factor', 'Attraction-Repulsion Ranked Choice Variance Heatmap')

    # bounded confidence ranked choice
    file_name = 'radius-second_choice_weight-bc-rc.png'
    df_path = os.path.join(sweeps_path, 'radius-second_choice_weight-bc-rc.csv')
    make_heatmap(df_path, output_dir, file_name, 'radius', 'second_choice_weight', 'Radius', 'Second Choice Weight Factor', 'Attraction-Repulsion Ranked Choice Variance Heatmap')


def sweep_exit_probability_decrease_threshold_increase(v_v_interaction_fn, v_c_interaction_fn, election_system, default_params):
    """
    Parameter sweep over exit probability decrease factor and threshold increase factor.
    """

    # number of parameter combinations
    start = 0.01
    stop = 1
    num_combos = 99

    exit_probability_decrease_list = np.linspace(start, stop, num_combos, endpoint=False)
    threshold_increase_list = np.linspace(start, stop, num_combos, endpoint=False)
    exit_probability_decrease_list = np.round(exit_probability_decrease_list, decimals=4)
    threshold_increase_list = np.round(threshold_increase_list, decimals=4)
    var_df = pd.DataFrame(columns=['exit_probability_decrease_factor', 'threshold_increase_factor', 'variance'])

    for exit_pr in exit_probability_decrease_list:
        for threshold in threshold_increase_list:
            run_df = pd.DataFrame(columns=['run', 'variance'])
            for run in default_params['num_runs']:
                model = ElectionSystem(
                        #seed=default_params['seed'],
                        num_voters=default_params['num_voters'],
                        num_voters_to_activate=default_params['num_voters_to_activate'],
                        initial_num_candidates=default_params['initial_num_candidates'],
                        min_candidates=default_params['min_candidates'],
                        max_candidates=default_params['max_candidates'],
                        term_limit=default_params['term_limit'],
                        num_opinions=default_params['num_opinions'],
                        election_system=election_system, # varied by run
                        voter_voter_interaction_fn=v_v_interaction_fn, # varied by run
                        voter_candidate_interaction_fn=v_c_interaction_fn, # varied by run
                        voter_noise_factor=default_params['voter_noise_factor'],
                        initial_exit_probability=default_params['initial_exit_probability'],
                        exit_probability_decrease_factor=exit_pr, # varied in run
                        initial_threshold=default_params['initial_threshold'],
                        threshold_increase_factor=threshold, # varied in run
                        num_candidates_to_benefit=default_params['num_candidates_to_benefit'],
                        num_rounds_before_election=default_params['num_rounds_before_election'],
                        mu=default_params['mu'],
                        radius=default_params['radius'],
                        learning_rate=default_params['learning_rate'],
                        gamma=default_params['gamma'],
                        beta=default_params['beta'],
                        second_choice_weight_factor=default_params['second_choice_weight_factor'],
                        exposure=default_params['exposure'],
                        responsiveness=default_params['responsiveness']
                    )
                
                # run the model
                for _ in range(default_params['num_steps']):
                    model.step()

                # add variance of this run to run_df
                agent_data = model.datacollector.get_agent_vars_dataframe()
                variance = float(agent_data.loc[agent_data['Step'] == default_params['num_steps']][['opinion1']].var())
                run_df.loc[run_df.shape[0]] = [run, variance]

            # add the average variance over all runs for that parameter combo to var_df
            var_df.loc[var_df.shape[0]] = [exit_pr, threshold, run_df['run'].mean()]

    output_directory = 'sweeps'
    run_type = f"{v_v_interaction_fn}-{election_system}"
    file_name = f'exit_pr_decrease-threshold_increase-{run_type}.csv'
    save_df(var_df, output_directory, file_name)


def sweep_gamma_beta(v_v_interaction_fn, v_c_interaction_fn, election_system, default_params):
    """
    Parameter sweep over exit probability decrease factor and threshold increase factor.
    """

    # number of parameter combinations
    start = 0.01
    stop = 1
    num_combos = 99

    gamma_list = np.linspace(start, stop, num_combos, endpoint=False)
    beta_list = np.linspace(start, stop, num_combos, endpoint=False)
    gamma_list = np.round(gamma_list, decimals=4)
    beta_list = np.round(beta_list, decimals=4)
    var_df = pd.DataFrame(columns=['gamma', 'beta', 'variance'])

    for gamma in gamma_list:
        for beta in beta_list:
            run_df = pd.DataFrame(columns=['run', 'variance'])
            for run in default_params['num_runs']:
                model = ElectionSystem(
                        #seed=default_params['seed'],
                        num_voters=default_params['num_voters'],
                        num_voters_to_activate=default_params['num_voters_to_activate'],
                        initial_num_candidates=default_params['initial_num_candidates'],
                        min_candidates=default_params['min_candidates'],
                        max_candidates=default_params['max_candidates'],
                        term_limit=default_params['term_limit'],
                        num_opinions=default_params['num_opinions'],
                        election_system=election_system, # varied by run
                        voter_voter_interaction_fn=v_v_interaction_fn, # varied by run
                        voter_candidate_interaction_fn=v_c_interaction_fn, # varied by run
                        voter_noise_factor=default_params['voter_noise_factor'],
                        initial_exit_probability=default_params['initial_exit_probability'],
                        exit_probability_decrease_factor=default_params['exit_pr'],
                        initial_threshold=default_params['initial_threshold'],
                        threshold_increase_factor=default_params['threshold'],
                        num_candidates_to_benefit=default_params['num_candidates_to_benefit'],
                        num_rounds_before_election=default_params['num_rounds_before_election'],
                        mu=default_params['mu'],
                        radius=default_params['radius'],
                        learning_rate=default_params['learning_rate'],
                        gamma=gamma, # varied in run
                        beta=beta, # varied in run
                        second_choice_weight_factor=default_params['second_choice_weight_factor'],
                        exposure=default_params['exposure'],
                        responsiveness=default_params['responsiveness']
                    )
                
                # run the model
                for _ in range(default_params['num_steps']):
                    model.step()

                # add variance of this run to run_df
                agent_data = model.datacollector.get_agent_vars_dataframe()
                variance = float(agent_data.loc[agent_data['Step'] == default_params['num_steps']][['opinion1']].var())
                run_df.loc[run_df.shape[0]] = [run, variance]

            # add the average variance over all runs for that parameter combo to var_df
            var_df.loc[var_df.shape[0]] = [gamma, beta, run_df['run'].mean()]

    output_directory = 'sweeps'
    run_type = f"{v_v_interaction_fn}-{election_system}"
    file_name = f'gamma-beta-{run_type}.csv'
    save_df(var_df, output_directory, file_name)


def sweep_radius_second_choice_weight(v_v_interaction_fn, v_c_interaction_fn, election_system, default_params):
    """
    Parameter sweep over exit probability decrease factor and threshold increase factor.
    """

    # number of parameter combinations
    start = 0.01
    stop = 1
    num_combos = 99

    radius_list = np.linspace(start, stop, num_combos, endpoint=False)
    second_choice_weight_list = np.linspace(start, stop, num_combos, endpoint=False)
    radius_list = np.round(radius_list, decimals=4)
    second_choice_weight_list = np.round(second_choice_weight_list, decimals=4)
    var_df = pd.DataFrame(columns=['radius', 'second_choice_weight', 'variance'])

    for radius in radius_list:
        for weight in second_choice_weight_list:
            run_df = pd.DataFrame(columns=['run', 'variance'])
            for run in default_params['num_runs']:
                model = ElectionSystem(
                        #seed=default_params['seed'],
                        num_voters=default_params['num_voters'],
                        num_voters_to_activate=default_params['num_voters_to_activate'],
                        initial_num_candidates=default_params['initial_num_candidates'],
                        min_candidates=default_params['min_candidates'],
                        max_candidates=default_params['max_candidates'],
                        term_limit=default_params['term_limit'],
                        num_opinions=default_params['num_opinions'],
                        election_system=election_system, # varied by run
                        voter_voter_interaction_fn=v_v_interaction_fn, # varied by run
                        voter_candidate_interaction_fn=v_c_interaction_fn, # varied by run
                        voter_noise_factor=default_params['voter_noise_factor'],
                        initial_exit_probability=default_params['initial_exit_probability'],
                        exit_probability_decrease_factor=default_params['exit_pr'],
                        initial_threshold=default_params['initial_threshold'],
                        threshold_increase_factor=default_params['threshold'],
                        num_candidates_to_benefit=default_params['num_candidates_to_benefit'],
                        num_rounds_before_election=default_params['num_rounds_before_election'],
                        mu=default_params['mu'],
                        radius=radius, # varied in run
                        learning_rate=default_params['learning_rate'],
                        gamma=default_params['gamma'],
                        beta=default_params['beta'], 
                        second_choice_weight_factor=weight, # varied in run
                        exposure=default_params['exposure'],
                        responsiveness=default_params['responsiveness']
                    )
                
                # run the model
                for _ in range(default_params['num_steps']):
                    model.step()

                # add variance of this run to run_df
                agent_data = model.datacollector.get_agent_vars_dataframe()
                variance = float(agent_data.loc[agent_data['Step'] == default_params['num_steps']][['opinion1']].var())
                run_df.loc[run_df.shape[0]] = [run, variance]

            # add the average variance over all runs for that parameter combo to var_df
            var_df.loc[var_df.shape[0]] = [radius, weight, run_df['run'].mean()]

    output_directory = 'sweeps'
    run_type = f"{v_v_interaction_fn}-{election_system}"
    file_name = f'radius-second_choice_weight-{run_type}.csv'
    save_df(var_df, output_directory, file_name)


def make_heatmap(df_path, output_dir, file_name, param1, param2, xlab=None, ylab=None, title=None):
    """
    Create and save a heatmap of the variance from the sweep dataframe passed in the df_path.
    Saves to:

    /storage/$USER/election-simulation/figures/$OUTPUT_DIR/$FILE_NAME.csv
    """

    # set figure dimensions
    plt.figure(
        figsize=(6, 6), 
        dpi = 600
    )

    # load in datafram
    df = pd.read_csv(df_path)

    # make heatmap
    sns.heatmap(data=df.pivot(param1, param2, 'variance'), cmap='rocket')

    # set labels
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    plt.gca().invert_yaxis()
    if title:
        plt.title('Heatmap of Variance')

    # save plot
    storage_directory = os.environ.get('STORAGE')
    if not storage_directory:
        raise EnvironmentError("$STORAGE environment variable not set.")

    # make election simulation directory in storage
    simulation_directory = os.path.join(storage_directory, r'election-simulation')
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)

    # make election simulation figure directory in storage
    figure_directory = os.path.join(simulation_directory, r'figures')
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    final_directory = os.path.join(figure_directory, output_dir)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # make final file path
    file_path = os.path.join(final_directory, file_name)

    # save figure
    plt.savefig(file_path)


def save_df(df, output_dir, file_name):
    """
    Save the dataframe to the $STORAGE directory saved in the $STORAGE environment variable.
    The model data and agent data output files are stored to:

    /storage/$USER/election-simulation/data/$OUTPUT_DIR/$FILE_NAME.csv
    """

    # make sure storage directory is set
    storage_directory = os.environ.get('STORAGE')
    if not storage_directory:
        raise EnvironmentError("$STORAGE environment variable not set.")

    # make election simulation directory in storage
    simulation_directory = os.path.join(storage_directory, r'election-simulation')
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)

    # make election simulation figure directory in storage
    data_directory = os.path.join(simulation_directory, r'data')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # make final data output directory
    final_directory = os.path.join(data_directory, output_dir)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # make final file path
    file_path = os.path.join(final_directory, file_name)

    # save csv files of dataframe
    df.to_csv(file_path, index=False)


def save_model_data(model, output_dir, run_extension, save_model_data=True, save_agent_data=True):
    """
    Save the data from the model to the $STORAGE directory saved in the $STORAGE environment variable.
    model_data and agent_data flags for saving those csv files.
    The model data and agent data output files are stored to:

    /storage/$USER/election-simulation/data/$OUTPUT_DIR/data_$RUN_EXTENSION.csv
    """

    # make sure storage directory is set
    storage_directory = os.environ.get('STORAGE')
    if not storage_directory:
        raise EnvironmentError("$STORAGE environment variable not set.")

    # make election simulation directory in storage
    simulation_directory = os.path.join(storage_directory, r'election-simulation')
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)

    # make election simulation data directory in storage
    data_directory = os.path.join(simulation_directory, r'data')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # make final data output directory
    final_directory = os.path.join(data_directory, output_dir)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # make filenames with extension for specific run
    model_data_file_name = f"model_data_{run_extension}.csv"
    agent_data_file_name = f"agent_data_{run_extension}.csv"

    # make final file paths
    model_data_file_path = os.path.join(final_directory, model_data_file_name)
    agent_data_file_path = os.path.join(final_directory, agent_data_file_name)

    # save csv files of agent and model data
    if save_model_data:
        model.datacollector.get_model_vars_dataframe().to_csv(model_data_file_path)

    if save_agent_data:
        model.datacollector.get_agent_vars_dataframe().to_csv(agent_data_file_path)


if __name__ == '__main__':
    main()

