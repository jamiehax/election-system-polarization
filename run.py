from model import ElectionSystem
import os
from tqdm import tqdm


def main():
    params = {
            'num_voters': 100,
            'num_candidates': 5,
            'width': 1,
            'height': 1,
            'num_opinions': 1,
            'voter_voter_interaction_fn': 'avg',
            'voter_candidate_interaction_fn': None,
            'candidate_voter_interaction_fn': 'avg',
            'candidate_candidate_interaction_fn': None,
            'num_voters_to_activate': 1,
            'num_candidates_to_activate': 5,
            'threshold': 0.2,
            'mu': 0.5,
        }

    model = build_model(params)
    run_model(model, time_steps=3000)
    save_data(model)


def build_model(params):
    """
    THINGS WE WANT TO BE MODULAR (I.E. WE CAN CONTROL THROUGH PARAMETERS)
        - the election system (plurality, ranked choice, cardinal)
        - the method for agents to find other agents to interact with
    """

    model = ElectionSystem(
        num_voters=params['num_voters'],
        num_candidates=params['num_candidates'],
        width=params['width'],
        height=params['height'],
        num_opinions=params['num_opinions'],
        voter_voter_interaction_fn=params['voter_voter_interaction_fn'],
        voter_candidate_interaction_fn=params['voter_candidate_interaction_fn'],
        candidate_voter_interaction_fn=params['candidate_voter_interaction_fn'],
        candidate_candidate_interaction_fn=params['candidate_candidate_interaction_fn'],
        num_voters_to_activate=params['num_voters_to_activate'],
        num_candidates_to_activate=params['num_candidates_to_activate'],
        threshold=params['threshold'],
        mu=params['mu'],
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

