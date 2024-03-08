from model import ElectionSystem
import os
from tqdm import tqdm

def main():
    params = {
            'num_voters': 10,
            'num_voters_to_activate': 1,
            'initial_num_candidates': 3,
            'min_candidates': 3,
            'max_candidates': 5,
            'num_candidates_to_activate': 3, # since num candidates changes, activates no more than this number
            'term_limit': 2,
            'num_opinions': 3,
            'election_system': 'rc', # plurality, rc, score
            'voter_voter_interaction_fn': 'bc', # avg, bc
            'voter_candidate_interaction_fn': 'bc', # avg, bc
            'candidate_voter_interaction_fn': 'avg', # avg, bc, kmean, ascent
            'candidate_candidate_interaction_fn': None,
            'initial_exit_probability': 0.33,
            'exit_probability_decrease_factor': 0.25,     
            'initial_threshold': 0.2,
            'threshold_increase_factor': 0.25,
            'num_candidates_to_benefit': 2,
            'num_rounds_before_election': 3,
            'mu': 0.5,
            'radius': 0.2,
            'beta': 0.2,
            'C': 0.2,
            'second_choice_weight_factor': 0.1
        }

    model = build_model(params)
    run_model(model, time_steps=15)
    save_data(model)


def build_model(params):
    """
    THINGS WE WANT TO BE MODULAR (I.E. WE CAN CONTROL THROUGH PARAMETERS)
        - the election system (plurality, ranked choice, cardinal)
        - the method for agents to find other agents to interact with
    """

    model = ElectionSystem(
        seed=0,
        num_voters=params['num_voters'],
        initial_num_candidates=params['initial_num_candidates'],
        term_limit=params['term_limit'],
        num_opinions=params['num_opinions'],
        election_system=params['election_system'],
        voter_voter_interaction_fn=params['voter_voter_interaction_fn'],
        voter_candidate_interaction_fn=params['voter_candidate_interaction_fn'],
        candidate_voter_interaction_fn=params['candidate_voter_interaction_fn'],
        candidate_candidate_interaction_fn=params['candidate_candidate_interaction_fn'],
        num_voters_to_activate=params['num_voters_to_activate'],
        num_candidates_to_activate=params['num_candidates_to_activate'],
        initial_exit_probability=params['initial_exit_probability'],
        exit_probability_decrease_factor=params['exit_probability_decrease_factor'],
        min_candidates=params['min_candidates'],
        max_candidates=params['max_candidates'],
        initial_threshold=params['initial_threshold'],
        threshold_increase_factor=params['threshold_increase_factor'],
        num_candidates_to_benefit=params['num_candidates_to_benefit'],
        mu=params['mu'],
        num_rounds_before_election=params['num_rounds_before_election'],
        radius=params['radius'],
        beta=params['beta'],
        C=params['C'],
        second_choice_weight_factor=params['second_choice_weight_factor']
    )
    return model


def run_model(model, time_steps):
    """
    Run the model for the specified number of time steps.
    """
    for _ in tqdm(range(time_steps)):
        model.step()


def save_data(model):
    """
    Save the data from the model
    """
    # make data output directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'data')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # save csv files of agent and model data
    model.datacollector.get_model_vars_dataframe().to_csv('data/model_data.csv')
    model.datacollector.get_agent_vars_dataframe().to_csv('data/agent_data.csv')


if __name__ == '__main__':
    main()

