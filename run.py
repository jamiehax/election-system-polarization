from model import ElectionSystem
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
from multiprocessing import Pool


class Parameters:
    """
    Run class - attribute self.params holds default parameters for the run.
    """
    
    def __init__(self) -> None:
        self.default_params = {
            #seed:0,
            'device': 'cpu', # device to compute tensors on ('cpu' or 'cuda')
            'num_steps': 10000, # number of model steps
            'num_steps_before_ascent': 20, # number of steps before running candidate ascent
            'num_runs': 20, # number of runs to average variance for
            'num_voters': 100,
            'num_voters_to_activate': 1,
            'initial_num_candidates': 3,
            'min_candidates': 3,
            'max_candidates': 5,
            'term_limit': 2,
            'num_opinions': 1,
            'voter_noise_factor': 0.001,
            'initial_exit_probability': 0.33,
            'exit_probability_decrease_factor': 0.25,     
            'initial_threshold': 0.2,
            'threshold_increase_factor': 0.25,
            'num_candidates_to_benefit': 2,
            'num_rounds_before_election': 500,
            'mu': 0.25, # mu in bounded confidence (controls magnitude of opinion update) - ALSO R IN AR MODEL
            'radius': 0.1, # r in radius of support function (controls inflection point)
            'learning_rate': 0.00002, # learning rate for candidate gradient ascent
            'gamma': 10, # gamma in radius of support function (controls steepness which is effectively variation in voting probabilities)
            'beta': 1, # variablity in the vote probabilities - higher beta less random
            'second_choice_weight_factor': 0.75,
            'exposure': 0.2, # E in AR model (controls probability of interacting)
        }
        


class OpinionDynamics:
    """
    Methods for simulating single runs and making corresponding opinion dyanmics plots
    """


    def __init__(self, params) -> None:
        self.params = params


    def run_model(self, v_v_interaction_fn, v_c_interaction_fn, election_system):
        model = ElectionSystem(
                #seed=self.params['seed'],
                num_voters=self.params['num_voters'],
                num_voters_to_activate=self.params['num_voters_to_activate'],
                initial_num_candidates=self.params['initial_num_candidates'],
                min_candidates=self.params['min_candidates'],
                max_candidates=self.params['max_candidates'],
                term_limit=self.params['term_limit'],
                num_opinions=self.params['num_opinions'],
                election_system=election_system, # varied by run
                voter_voter_interaction_fn=v_v_interaction_fn, # varied by run
                voter_candidate_interaction_fn=v_c_interaction_fn, # varied by run
                voter_noise_factor=self.params['voter_noise_factor'],
                initial_exit_probability=self.params['initial_exit_probability'],
                exit_probability_decrease_factor=self.params['exit_probability_decrease_factor'],
                initial_threshold=self.params['initial_threshold'],
                threshold_increase_factor=self.params['threshold_increase_factor'],
                num_candidates_to_benefit=self.params['num_candidates_to_benefit'],
                num_rounds_before_election=self.params['num_rounds_before_election'],
                mu=self.params['mu'],
                radius=self.params['radius'],
                learning_rate=self.params['learning_rate'],
                gamma=self.params['gamma'],
                beta=self.params['beta'],
                second_choice_weight_factor=self.params['second_choice_weight_factor'],
                exposure=self.params['exposure'],
            )
                        
        # run the model
        for _ in range(self.params['num_steps']):
            model.step()     

        # calculate and return variance at each step
        variances = model.datacollector.get_model_vars_dataframe()['variance']
        return variances.iloc[1:].reset_index(drop=True)


    def run_model_save_data(self, v_v_interaction_fn, v_c_interaction_fn, election_system, num_simulations):
        for run in range(num_simulations):
            model = ElectionSystem(
                    #seed=self.params['seed'],
                    num_voters=self.params['num_voters'],
                    num_voters_to_activate=self.params['num_voters_to_activate'],
                    initial_num_candidates=self.params['initial_num_candidates'],
                    min_candidates=self.params['min_candidates'],
                    max_candidates=self.params['max_candidates'],
                    term_limit=self.params['term_limit'],
                    num_opinions=self.params['num_opinions'],
                    election_system=election_system, # varied by run
                    voter_voter_interaction_fn=v_v_interaction_fn, # varied by run
                    voter_candidate_interaction_fn=v_c_interaction_fn, # varied by run
                    voter_noise_factor=self.params['voter_noise_factor'],
                    initial_exit_probability=self.params['initial_exit_probability'],
                    exit_probability_decrease_factor=self.params['exit_probability_decrease_factor'],
                    initial_threshold=self.params['initial_threshold'],
                    threshold_increase_factor=self.params['threshold_increase_factor'],
                    num_candidates_to_benefit=self.params['num_candidates_to_benefit'],
                    num_rounds_before_election=self.params['num_rounds_before_election'],
                    mu=self.params['mu'],
                    radius=self.params['radius'],
                    learning_rate=self.params['learning_rate'],
                    gamma=self.params['gamma'],
                    beta=self.params['beta'],
                    second_choice_weight_factor=self.params['second_choice_weight_factor'],
                    exposure=self.params['exposure'],
                )
                            
            # run the model
            for _ in range(self.params['num_steps']):
                model.step()     

            # ensure directories exist
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
            output_dir = f'runs/{v_v_interaction_fn}/{election_system}'
            final_directory = os.path.join(data_directory, output_dir)
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)

            
            file_name = f"run{run}.csv"
            save_path = os.path.join(final_directory, file_name)

            # save run data
            model.datacollector.get_agent_vars_dataframe().to_csv(save_path)
    
    
    def compare_election_system_variance(self):

        # compute variances across runs for bounded confidence - plurality
        avg_variances_bc_plurality = np.empty((0, self.params['num_steps']))
        for _ in range(1):
            args = [('bc', 'bc1', 'plurality')] * self.params['num_runs']
            with Pool() as pool:
                run_variances = pool.starmap(self.run_model, args)

            for run_variance in run_variances:
                avg_variances_bc_plurality = np.vstack([avg_variances_bc_plurality, run_variance])

        avg_variances_bc_plurality = np.mean(avg_variances_bc_plurality, axis=0)

        # compute variances across runs for bounded confidence - ranked choice
        avg_variances_bc_rc = np.empty((0, self.params['num_steps']))
        for _ in range(40):
            args = [('bc', 'bc1', 'rc')] * self.params['num_runs']
            with Pool() as pool:
                run_variances = pool.starmap(self.run_model, args)

            for run_variance in run_variances:
                avg_variances_bc_rc = np.vstack([avg_variances_bc_rc, run_variance])

        avg_variances_bc_rc = np.mean(avg_variances_bc_rc, axis=0)

        # compute variances across runs for attraction repulsion - plurality
        avg_variances_ar_plurality = np.empty((0, self.params['num_steps']))
        for _ in range(40):
            args = [('ar', 'ar', 'plurality')] * self.params['num_runs']
            with Pool() as pool:
                run_variances = pool.starmap(self.run_model, args)

            for run_variance in run_variances:
                avg_variances_ar_plurality = np.vstack([avg_variances_ar_plurality, run_variance])

        avg_variances_ar_plurality = np.mean(avg_variances_ar_plurality, axis=0)

        # compute variances across runs for attraction repulsion ranked choice
        avg_variances_ar_rc = np.empty((0, self.params['num_steps']))
        for _ in range(40):
            args = [('ar', 'ar', 'rc')] * self.params['num_runs']
            with Pool() as pool:
                run_variances = pool.starmap(self.run_model, args)

            for run_variance in run_variances:
                avg_variances_ar_rc = np.vstack([avg_variances_ar_rc, run_variance])

        avg_variances_ar_rc = np.mean(avg_variances_ar_rc, axis=0)

        # print final variances
        print(f'final bounded confidence plurality average variance: {avg_variances_bc_plurality[-1]}')
        print(f'final bounded confidence ranked choice average variance: {avg_variances_bc_rc[-1]}')
        print(f'final attraction-repulsion plurality average variance: {avg_variances_ar_plurality[-1]}')
        print(f'final attraction-repulsion ranked choice average variance: {avg_variances_ar_rc[-1]}')

        # df for all run types
        model_types = ['Bounded Confidence Plurality', 'Bounded Confidence Ranked Choice', 'Attraction-Repulsion Plurality', 'Attraction-Repulsion Ranked Choice']
        data = {
            'Step': np.tile(np.arange(self.params['num_steps']),  4),
            'Variance': np.concatenate([avg_variances_bc_plurality, avg_variances_bc_rc, avg_variances_ar_plurality, avg_variances_ar_rc]),
            'Model Type': np.repeat(model_types, self.params['num_steps'])
        }
        var_df = pd.DataFrame(data)

        # make plot
        plt.figure(
            figsize=(12, 6), 
            dpi = 600
        )

        # set plot styles
        sns.set_style('ticks')
        colors = {
            'Bounded Confidence Plurality': sns.color_palette('Paired')[4],
            'Bounded Confidence Ranked Choice': sns.color_palette('Paired')[5],
            'Attraction-Repulsion Plurality': sns.color_palette('Paired')[8],
            'Attraction-Repulsion Ranked Choice': sns.color_palette('Paired')[9]
        }

        # plot
        var_plot = sns.lineplot(
            data=var_df,
            x='Step',
            y='Variance',
            hue='Model Type',
            palette=colors
        )

        # Add legend
        plt.legend(title='Model Types')

        var_plot.spines[['right', 'top']].set_visible(False)
        var_plot.set(
            xlabel='Time Step',
            ylabel='Variance in Opinion',
            title='Variance in Opinion over Time'
        )

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

        final_directory = os.path.join(figure_directory, r'variances')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        # make final file path
        file_path = os.path.join(final_directory, 'compare_variance.png')

        # save figure
        plt.savefig(file_path)


    def opinion_dynamics(self, num_simulations):
        """
        Run all run type combinations for the given number of simulations.
        """

        # run simulations
        # self.run_model_save_data('bc', 'bc1', 'plurality', num_simulations=num_simulations)
        # self.run_model_save_data('ar', 'ar', 'plurality', num_simulations=num_simulations)
        # self.run_model_save_data('bc', 'bc1', 'rc', num_simulations=num_simulations)
        # self.run_model_save_data('ar', 'ar', 'rc', num_simulations=num_simulations)

        # make sure storage directory is set
        storage_directory = os.environ.get('STORAGE')
        if not storage_directory:
            raise EnvironmentError("$STORAGE environment variable not set.")

        # find runs data directory
        runs_path = os.path.join(storage_directory, r'election-simulation/data/runs')

        # get directory of dataframe for bounded confidence - plurality
        bc_plurality_path = os.path.join(runs_path, 'bc/plurality')
        for df_path in os.listdir(bc_plurality_path):
            # get output file_name
            file_name = os.path.basename(df_path)
            file_name_out = os.path.splitext(file_name)[0] + ".png"
            df_path = os.path.join(bc_plurality_path, df_path)
            self.make_opinion_dynamics_plots(df_path=df_path, file_name=file_name_out, run_type='bc-plurality')

        # get directory of dataframe for attraction repulsion - plurality
        ar_plurality_path = os.path.join(runs_path, 'ar/plurality')
        for df_path in os.listdir(ar_plurality_path):
            # get output file_name
            file_name = os.path.basename(df_path)
            file_name_out = os.path.splitext(file_name)[0] + ".png"
            df_path = os.path.join(ar_plurality_path, df_path)
            self.make_opinion_dynamics_plots(df_path=df_path, file_name=file_name_out, run_type='ar-plurality')

        # get directory of dataframe for bounded confidence - ranked choice
        bc_rc_path = os.path.join(runs_path, 'bc/rc')
        for df_path in os.listdir(bc_rc_path):
            # get output file_name
            file_name = os.path.basename(df_path)
            file_name_out = os.path.splitext(file_name)[0] + ".png"
            df_path = os.path.join(bc_rc_path, df_path)
            self.make_opinion_dynamics_plots(df_path=df_path, file_name=file_name_out, run_type='bc-rc')

        # get directory of dataframe for attraction repulsion - ranked choice
        ar_rc_path = os.path.join(runs_path, 'ar/rc')
        for df_path in os.listdir(ar_rc_path):
            # get output file_name
            file_name = os.path.basename(df_path)
            file_name_out = os.path.splitext(file_name)[0] + ".png"
            df_path = os.path.join(ar_rc_path, df_path)
            self.make_opinion_dynamics_plots(df_path=df_path, file_name=file_name_out, run_type='ar-rc')


    def make_opinion_dynamics_plots(self, df_path, file_name, run_type):
        """
        Create and save a heatmap of the variance from the sweep dataframe passed in the df_path.
        Saves to:

        run_type: 'interaction_rule-election_system'

        /storage/$USER/election-simulation/figures/$OUTPUT_DIR/$FILE_NAME.csv
        """

        if run_type == 'bc-plurality':
            title = 'Plurality Election System with Bounded Confidence Opinion Updates'
        if run_type == 'ar-plurality':
            title = 'Plurality Election System with Attraction-Repulsion Opinion Updates'
        if run_type == 'bc-rc':
            title = 'Ranked Choice Election System with Bounded Confidence Opinion Updates '
        if run_type == 'ar-rc':
            title = 'Ranked Choice Election System with Attraction-Repulsion Opinion Updates'

        plt.figure(
            figsize=(12, 6), 
            dpi = 600
        )

        # load in dataframe
        df = pd.read_csv(df_path)

        # set plot styles
        sns.set_style('ticks')

        # plot voters lines
        voters = df[df['type'] == 'voter']
        location_plot = sns.lineplot(
            data=voters,
            x="Step",
            y="opinion1",
            color = sns.color_palette('Set2')[7],
            size='threshold',
            units="AgentID",
            estimator=None,
            legend=False
        )

        # plot candidate lines
        candidates = df[df['type'] == 'candidate']
        winner_colors = {True: sns.color_palette('rocket')[3], False: sns.color_palette('rocket')[1]}
        location_plot = sns.lineplot(
            data=candidates,
            x="Step",
            y="opinion1",
            hue="is_winner",
            palette=winner_colors,
            size='threshold',
            sizes=(0.25, 5),
            units="AgentID",
            estimator=None,
            legend=False
        )

        location_plot.spines[['right', 'top']].set_visible(False)
        location_plot.set(
            xlabel='Time Step',
            ylabel='Opinion Value',
            title=f"Agent Opinion Distribution in {title}"
        )

        # save plot
        storage_directory = os.environ.get('STORAGE')
        if not storage_directory:
            raise EnvironmentError("$STORAGE environment variable not set.")

        # make election simulation directory in storage
        simulation_directory = os.path.join(storage_directory, r'election-simulation')
        if not os.path.exists(simulation_directory):
            os.makedirs(simulation_directory)

        # make election simulation figure directory in storage
        figure_directory = os.path.join(simulation_directory, r'figures/runs')
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        final_directory = os.path.join(figure_directory, run_type)
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        # make final file path
        file_path = os.path.join(final_directory, file_name)

        # save figure
        plt.savefig(file_path)



class Sweeps:
    """
    Methods for running parameter sweeps
    """


    def __init__(self, params) -> None:
        self.params = params


    def run_model(self, params):

        model = ElectionSystem(
                #seed=self.params['seed'],
                num_voters=params['num_voters'],
                num_voters_to_activate=params['num_voters_to_activate'],
                initial_num_candidates=params['initial_num_candidates'],
                min_candidates=params['min_candidates'],
                max_candidates=params['max_candidates'],
                term_limit=params['term_limit'],
                num_opinions=params['num_opinions'],
                election_system=params['election_system'], # varied by run
                voter_voter_interaction_fn=params['v_v_interaction_fn'], # varied by run
                voter_candidate_interaction_fn=params['v_c_interaction_fn'], # varied by run
                voter_noise_factor=params['voter_noise_factor'],
                initial_exit_probability=params['initial_exit_probability'],
                exit_probability_decrease_factor=params['exit_probability_decrease_factor'],
                initial_threshold=params['initial_threshold'],
                threshold_increase_factor=params['threshold_increase_factor'],
                num_candidates_to_benefit=params['num_candidates_to_benefit'],
                num_rounds_before_election=params['num_rounds_before_election'],
                mu=params['mu'],
                radius=params['radius'],
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                beta=params['beta'],
                second_choice_weight_factor=params['second_choice_weight_factor'],
                exposure=params['exposure'],
            )
                        
        # run the model
        for _ in range(self.params['num_steps']):
            model.step()     


        # return variance at last step
        model_vars = model.datacollector.get_model_vars_dataframe()
        variance = float(model_vars.tail(1)['variance'])
        return variance

    
    def et_sweep(self):
        """
        Run ALL run type combos for Exit Probability Decrease Factor and Threshold Increase Factor parameter sweep.
        """

        # Attraction-Repulsion / Plurality
        self.sweep_exit_probability_decrease_threshold_increase(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='plurality',
        )

        # Bounded Confidence / Plurality
        self.sweep_exit_probability_decrease_threshold_increase(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='plurality',
        )
        # Attraction-Repulsion / Ranked Choice
        self.sweep_exit_probability_decrease_threshold_increase(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='rc',
        )

        # Bounded Confidence / Ranked Choice
        self.sweep_exit_probability_decrease_threshold_increase(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='rc',
        ) 
    

    def sweep_exit_probability_decrease_threshold_increase(self, v_v_interaction_fn, v_c_interaction_fn, election_system):
        """
        Parameter sweep over exit probability decrease factor and threshold increase factor.
        """

        # number of parameter combinations
        start = 0.08
        stop = 2
        num_combos = 25

        exit_probability_decrease_list = np.linspace(start, stop, num_combos, endpoint=True)
        threshold_increase_list = np.linspace(start, stop, num_combos, endpoint=True)
        exit_probability_decrease_list = np.round(exit_probability_decrease_list, decimals=5)
        threshold_increase_list = np.round(threshold_increase_list, decimals=5)
        var_df = pd.DataFrame(columns=['exit_probability_decrease_factor', 'threshold_increase_factor', 'variance'])

        for exit_pr in exit_probability_decrease_list:
            for threshold in threshold_increase_list:

                params = self.params
                params['v_v_interaction_fn'] = v_v_interaction_fn
                params['v_c_interaction_fn'] = v_c_interaction_fn
                params['election_system'] = election_system
                params['exit_probability_decrease_factor'] = exit_pr
                params['threshold_increase_factor'] = threshold

                # multiprocess simulations to run concurrently
                args = [(params)] * self.params['num_runs']
                with Pool() as pool:
                    variances = pool.map(self.run_model, args)

                # add the average variance from all runs to var_df
                avg_var = sum(variances) / len(variances)
                var_df.loc[var_df.shape[0]] = [exit_pr, threshold, avg_var]


        output_directory = 'sweeps'
        run_type = f"{v_v_interaction_fn}-{election_system}"
        file_name = f'exit_pr_decrease-threshold_increase-{run_type}.csv'
        self.save_df(var_df, output_directory, file_name)


    def gr_sweep(self):
        """
        Run ALL run type combos for Gamma and Radius Parameter sweep.
        """

        # Attraction-Repulsion / Plurality
        self.sweep_gamma_radius(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='plurality',
        )

        # Bounded Confidence / Plurality
        self.sweep_gamma_radius(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='plurality',
        )
        # Attraction-Repulsion / Ranked Choice
        self.sweep_gamma_radius(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='rc',
        )

        # Bounded Confidence / Ranked Choice
        self.sweep_gamma_radius(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='rc',
        )


    def sweep_gamma_radius(self, v_v_interaction_fn, v_c_interaction_fn, election_system):
        """
        Parameter sweep over exit probability decrease factor and threshold increase factor.
        """

        # number of parameter combinations
        num_combos = 25

        gamma_list = np.linspace(0.8, 20, num_combos, endpoint=True)
        radius_list = np.linspace(0.04, 1, num_combos, endpoint=True)
        gamma_list = np.round(gamma_list, decimals=5)
        radius_list = np.round(radius_list, decimals=5)
        var_df = pd.DataFrame(columns=['gamma', 'radius', 'variance'])

        for gamma in gamma_list:
            for radius in radius_list:

                params = self.params
                params['v_v_interaction_fn'] = v_v_interaction_fn
                params['v_c_interaction_fn'] = v_c_interaction_fn
                params['election_system'] = election_system
                params['gamma'] = gamma
                params['radius'] = radius

                # multiprocess simulations to run concurrently
                args = [(params)] * self.params['num_runs']
                with Pool() as pool:
                    variances = pool.map(self.run_model, args)

                # add the average variance from all runs to var_df
                avg_var = sum(variances) / len(variances)
                var_df.loc[var_df.shape[0]] = [gamma, radius, avg_var]


        output_directory = 'sweeps'
        run_type = f"{v_v_interaction_fn}-{election_system}"
        file_name = f'gamma-radius-{run_type}.csv'
        self.save_df(var_df, output_directory, file_name)


    def bs_sweep(self):
        """
        Run ALL run type combos for Beta and Second Choice Weight Factor parameter sweep.
        """

        # Attraction-Repulsion / Plurality
        self.sweep_beta_second_choice_weight(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='plurality',
        )

        # Bounded Confidence / Plurality
        self.sweep_beta_second_choice_weight(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='plurality',
        )
        # Attraction-Repulsion / Ranked Choice
        self.sweep_beta_second_choice_weight(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='rc',
        )

        # Bounded Confidence / Ranked Choice
        self.sweep_beta_second_choice_weight(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='rc',
        )


    def sweep_beta_second_choice_weight(self, v_v_interaction_fn, v_c_interaction_fn, election_system):
        """
        Parameter sweep over exit probability decrease factor and threshold increase factor.
        """

        # number of parameter combinations
        start = 0.04
        stop = 1
        num_combos = 25

        beta_list = np.linspace(0.2, 5, num_combos, endpoint=True)
        second_choice_weight_list = np.linspace(start, stop, num_combos, endpoint=True)
        beta_list = np.round(beta_list, decimals=5)
        second_choice_weight_list = np.round(second_choice_weight_list, decimals=5)
        var_df = pd.DataFrame(columns=['beta', 'second_choice_weight', 'variance'])

        for beta in beta_list:
            for weight in second_choice_weight_list:

                params = self.params
                params['v_v_interaction_fn'] = v_v_interaction_fn
                params['v_c_interaction_fn'] = v_c_interaction_fn
                params['election_system'] = election_system
                params['beta'] = beta
                params['second_choice_weight_factor'] = weight

                # multiprocess simulations to run concurrently
                args = [(params)] * self.params['num_runs']
                with Pool() as pool:
                    variances = pool.map(self.run_model, args)

                # add the average variance from all runs to var_df
                avg_var = sum(variances) / len(variances)
                var_df.loc[var_df.shape[0]] = [beta, weight, avg_var]

    
        output_directory = 'sweeps'
        run_type = f"{v_v_interaction_fn}-{election_system}"
        file_name = f'beta-second_choice_weight-{run_type}.csv'
        self.save_df(var_df, output_directory, file_name)


    def tm_sweep(self):
        """
        Run ALL run type combos for Initial Threshold and mu parameter sweep.
        """

        # Attraction-Repulsion / Plurality
        self.sweep_threshold_mu(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='plurality',
        )

        # Attraction-Repulsion / Ranked Choice
        self.sweep_threshold_mu(
            v_v_interaction_fn='ar',
            v_c_interaction_fn='ar',
            election_system='rc',
        )

        # Bounded Confidence / Plurality
        self.sweep_threshold_mu(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='plurality',
        )

        # Bounded Confidence / Ranked Choice
        self.sweep_threshold_mu(
            v_v_interaction_fn='bc',
            v_c_interaction_fn='bc1',
            election_system='rc',
        )


    def sweep_threshold_mu(self, v_v_interaction_fn, v_c_interaction_fn, election_system):
        """
        Parameter sweep over initial threshold and mu.
        """

        # number of parameter combinations
        start = 0.038
        stop = 0.95
        num_combos = 25

        threshold_list = np.linspace(start, stop, num_combos, endpoint=True)
        mu_list = np.linspace(start, stop, num_combos, endpoint=True)
        threshold_list = np.round(threshold_list, decimals=5)
        mu_list = np.round(mu_list, decimals=5)
        var_df = pd.DataFrame(columns=['threshold', 'mu', 'variance'])

        for threshold in threshold_list:
            for mu in mu_list:

                params = self.params
                params['v_v_interaction_fn'] = v_v_interaction_fn
                params['v_c_interaction_fn'] = v_c_interaction_fn
                params['election_system'] = election_system
                params['initial_threshold'] = threshold
                params['mu'] = mu

                # multiprocess simulations to run concurrently
                args = [(params)] * self.params['num_runs']
                with Pool() as pool:
                    variances = pool.map(self.run_model, args)

                # add the average variance from all runs to var_df
                avg_var = sum(variances) / len(variances)
                var_df.loc[var_df.shape[0]] = [threshold, mu, avg_var]

    
        output_directory = 'sweeps'
        run_type = f"{v_v_interaction_fn}-{election_system}"
        file_name = f'threshold-mu-{run_type}.csv'
        self.save_df(var_df, output_directory, file_name)


    def make_figures(self):

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
        self.make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Attraction-Repulsion Plurality Variance Heatmap')

        # bounded confidence plurality
        file_name = 'exit_pr_decrease-threshold_increase-bc-plurality.png'
        df_path = os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-bc-plurality.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Bounded Confidence Plurality Variance Heatmap')

        # attraction repulsion ranked choice
        file_name = 'exit_pr_decrease-threshold_increase-ar-rc.png'
        df_path = os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-ar-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Attraction-Repulsion Ranked Choice Variance Heatmap')

        # bounded confidence ranked choice
        file_name = 'exit_pr_decrease-threshold_increase-bc-rc.png'
        df_path = os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-bc-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Bounded Confidence Ranked Choice Variance Heatmap')


        # GAMMA and RADIUS
        # attraction repulsion plurality
        file_name = 'gamma-radius-ar-plurality.png'
        df_path = os.path.join(sweeps_path, 'gamma-radius-ar-plurality.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'gamma', 'radius', 'Gamma', 'Radius', 'Attraction-Repulsion Plurality Variance Heatmap')

        # bounded confidence plurality
        file_name = 'gamma-radius-bc-plurality.png'
        df_path = os.path.join(sweeps_path, 'gamma-radius-bc-plurality.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'gamma', 'radius', 'Gamma', 'Radius', 'Bounded Confidence Plurality Variance Heatmap')

        # attraction repulsion ranked choice
        file_name = 'gamma-radius-ar-rc.png'
        df_path = os.path.join(sweeps_path, 'gamma-radius-ar-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'gamma', 'radius', 'Gamma', 'Radius', 'Attraction-Repulsion Ranked Choice Variance Heatmap')

        # bounded confidence ranked choice
        file_name = 'gamma-radius-bc-rc.png'
        df_path = os.path.join(sweeps_path, 'gamma-radius-bc-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'gamma', 'radius', 'Gamma', 'Radius', 'Bounded Confidence Ranked Choice Variance Heatmap')


        # BETA and SECOND CHOICE WEIGHT FACTOR
        # attraction repulsion plurality
        file_name = 'beta-second_choice_weight-ar-plurality.png'
        df_path = os.path.join(sweeps_path, 'beta-second_choice_weight-ar-plurality.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'beta', 'second_choice_weight', 'Beta', 'Second Choice Weight Factor', 'Attraction-Repulsion Plurality Variance Heatmap')

        # bounded confidence plurality
        file_name = 'beta-second_choice_weight-bc-plurality.png'
        df_path = os.path.join(sweeps_path, 'beta-second_choice_weight-bc-plurality.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'beta', 'second_choice_weight', 'Beta', 'Second Choice Weight Factor', 'Bounded Confidence Plurality Variance Heatmap')

        # attraction repulsion ranked choice
        file_name = 'beta-second_choice_weight-ar-rc.png'
        df_path = os.path.join(sweeps_path, 'beta-second_choice_weight-ar-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'beta', 'second_choice_weight', 'Beta', 'Second Choice Weight Factor', 'Attraction-Repulsion Ranked Choice Variance Heatmap')

        # bounded confidence ranked choice
        file_name = 'beta-second_choice_weight-bc-rc.png'
        df_path = os.path.join(sweeps_path, 'beta-second_choice_weight-bc-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'beta', 'second_choice_weight', 'Beta', 'Second Choice Weight Factor', 'Attraction-Repulsion Ranked Choice Variance Heatmap')

        # INITIAL THRESHOLD and MU
        # attraction repulsion plurality
        file_name = 'threshold-mu-ar-plurality.png'
        df_path = os.path.join(sweeps_path, 'threshold-mu-ar-plurality.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'threshold', 'mu', 'Threshold', 'Mu', 'Attraction-Repulsion Plurality Variance Heatmap')

        # attraction repulsion ranked choice
        file_name = 'threshold-mu-ar-rc.png'
        df_path = os.path.join(sweeps_path, 'threshold-mu-ar-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'threshold', 'mu', 'Threshold', 'Mu', 'Attraction-Repulsion Ranked Choice Variance Heatmap')


        # bounded confidence plurality
        file_name = 'threshold-mu-bc-plurality.png'
        df_path = os.path.join(sweeps_path, 'threshold-mu-bc-plurality.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'threshold', 'mu', 'Threshold', 'Mu', 'Bounded Confidence Plurality Variance Heatmap')

        # bounded confidence ranked choice
        file_name = 'threshold-mu-bc-rc.png'
        df_path = os.path.join(sweeps_path, 'threshold-mu-bc-rc.csv')
        self.make_heatmap(df_path, output_dir, file_name, 'threshold', 'mu', 'Threshold', 'Mu', 'Bounded Confidence Ranked Choice Variance Heatmap')


    def make_heatmap(self, df_path, output_dir, file_name, param1, param2, xlab=None, ylab=None, title=None):
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
        sns.heatmap(data=df.pivot(columns=param1, index=param2, values='variance'), cmap='rocket')

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


    def make_figure_grids(self):
        """
        Make sweep grids for all parameter sweeps

        IMPORTANT:
            df1 is bounded confidence plurality (top left)
            df2 is bounded confidence ranked choice (top right)
            df3 is attraction-repulsion plurality (bottom left)
            df4 is attraction-repulsion ranked choice (bottom right)
        """

        # make sure storage directory is set
        storage_directory = os.environ.get('STORAGE')
        if not storage_directory:
            raise EnvironmentError("$STORAGE environment variable not set.")

        # find sweeps data directory
        sweeps_path = os.path.join(storage_directory, r'election-simulation/data/sweeps')

        # EXIT PROBABILITY DECREASE FACTOR and THRESHOLD INCREASE FACTOR
        file_name = 'exit_pr_decrease-threshold_increase-grid.png'
        df1 = pd.read_csv(os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-bc-plurality.csv'))
        df2 = pd.read_csv(os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-bc-rc.csv'))
        df3 = pd.read_csv(os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-ar-plurality.csv'))
        df4 = pd.read_csv(os.path.join(sweeps_path, 'exit_pr_decrease-threshold_increase-ar-rc.csv'))
        self.make_heatmap_grid((df1, df2, df3, df4), 'exit_probability_decrease_factor', 'threshold_increase_factor', 'Exit Probability Decrease Factor', 'Threshold Increase Factor', 'Variance Heatmap', file_name)

        # GAMMA and RADIUS
        file_name = 'gamma-radius-grid.png'
        df1 = pd.read_csv(os.path.join(sweeps_path, 'gamma-radius-bc-plurality.csv'))
        df2 = pd.read_csv(os.path.join(sweeps_path, 'gamma-radius-bc-rc.csv'))
        df3 = pd.read_csv(os.path.join(sweeps_path, 'gamma-radius-ar-plurality.csv'))
        df4 = pd.read_csv(os.path.join(sweeps_path, 'gamma-radius-ar-rc.csv'))
        self.make_heatmap_grid((df1, df2, df3, df4), 'gamma', 'radius', 'Gamma', 'Radius', 'Variance Heatmap', file_name)

        # BETA and SECOND CHOICE WEIGHT FACTOR
        file_name = 'beta-second_choice_weight-grid.png'
        df1 = pd.read_csv(os.path.join(sweeps_path, 'beta-second_choice_weight-bc-plurality.csv'))
        df2 = pd.read_csv(os.path.join(sweeps_path, 'beta-second_choice_weight-bc-rc.csv'))
        df3 = pd.read_csv(os.path.join(sweeps_path, 'beta-second_choice_weight-ar-plurality.csv'))
        df4 = pd.read_csv(os.path.join(sweeps_path, 'beta-second_choice_weight-ar-rc.csv'))
        self.make_heatmap_grid((df1, df2, df3, df4), 'beta', 'second_choice_weight', 'Beta', 'Second Choice Weight Factor', 'Variance Heatmap', file_name)

        # THRESHOLD and MU
        file_name = 'threshold-mu-grid.png'
        df1 = pd.read_csv(os.path.join(sweeps_path, 'threshold-mu-bc-plurality.csv'))
        df2 = pd.read_csv(os.path.join(sweeps_path, 'threshold-mu-bc-rc.csv'))
        df3 = pd.read_csv(os.path.join(sweeps_path, 'threshold-mu-ar-plurality.csv'))
        df4 = pd.read_csv(os.path.join(sweeps_path, 'threshold-mu-ar-rc.csv'))
        self.make_heatmap_grid((df1, df2, df3, df4), 'threshold', 'mu', 'Threshold', 'Mu', 'Variance Heatmap', file_name)


    def make_heatmap_grid(self, dfs, col_param, row_param, xlab, ylab, title, file_name):
        """
        Make 2x2 grid of heatmaps for same parameter sweep.
        """
        
        df1, df2, df3, df4 = dfs
        figure, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10), dpi=1000)

        # bounded confidence variance baounds for color legend
        row_1_min = min(df['variance'].min() for df in (df1, df2))
        row_1_max = max(df['variance'].max() for df in (df1, df2))

        # bounded confidence plurality
        sns.heatmap(
            data=df1.pivot(columns=col_param, index=row_param, values='variance'), 
            cmap='rocket',
            ax=axes[0, 0], # top left
            vmin=row_1_min,
            vmax=row_1_max,
            cbar=False
        )
        axes[0, 0].set_title('Bounded Confidence Plurality')

        # bounded confidence ranked choice
        sns.heatmap(
            data=df2.pivot(columns=col_param, index=row_param, values='variance'), 
            cmap='rocket',
            ax=axes[0, 1], # top right
            vmin=row_1_min,
            vmax=row_1_max,
        )
        axes[0, 1].set_title('Bounded Confidence Ranked Choice')

        # attraction-repulsion variance bounds for color legend
        row_2_min = min(df['variance'].min() for df in (df3, df4))
        row_2_max = max(df['variance'].max() for df in (df3, df4))

        # attraction-repulsion plurality
        sns.heatmap(
            data=df3.pivot(columns=col_param, index=row_param, values='variance'), 
            cmap='rocket',
            ax=axes[1, 0], # bottom left
            vmin=row_2_min,
            vmax=row_2_max,
            cbar=False
        )
        axes[1, 0].set_title('Attraction-Repulsion Plurality')

        # attraction-repulsion ranked choice
        sns.heatmap(
            data=df4.pivot(columns=col_param, index=row_param, values='variance'), 
            cmap='rocket',
            ax=axes[1, 1], # bottom right
            vmin=row_2_min,
            vmax=row_2_max,
        )
        axes[1, 1].set_title('Attraction-Repulsion Ranked Choice')

        # invert y axis
        plt.gca().invert_yaxis()

        # set labels
        for ax in axes.flat:
            ax.set(xlabel=xlab, ylabel=ylab)

        # remove inner labels and tick marks
        axes[0, 1].set_xlabel('') 
        axes[0, 1].set_ylabel('')  
        axes[1, 1].set_ylabel('') 
        for ax in figure.get_axes():
            ax.label_outer()

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

        final_directory = os.path.join(figure_directory, r'sweeps/grids')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        # make final file path
        file_path = os.path.join(final_directory, file_name)

        # save figure
        figure.savefig(file_path)


    def save_df(self, df, output_dir, file_name):
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