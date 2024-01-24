from model import ElectionSystem
import os
from tqdm import tqdm


def build_model(params):
    """
    THINGS WE WANT TO BE MODULAR (I.E. WE CAN CONTROL THROUGH PARAMETERS)
        - the election system (plurality, ranked choice, cardinal)
        - the method for agents to find other agents to interact with

    Default Model Parameters:
    num_voters = 100
    num_candidates = 5
    width = 1
    height = 1
    num_opinions = 2
    voter_interaction_fn = averaging
    candidate_interaction_fn = bounded confidence
    num_voters_to_activate = 1
    num_candidatess_to_activate = 1
    d = 0.2
    mu = 0.5
    """

    model = ElectionSystem(
        num_voters=params.get('num_voters', 100),
        num_candidates=params.get('num_candidates', 5),
        width=params.get('width', 1),
        height=params.get('height', 1),
        num_opinions=params.get('num_opinions', 2),
        voter_interaction_fn=params.get('voter_interaction_fn', 'avg'),
        candidate_interaction_fn=params.get('candidate_interaction_fn', 'avg'),
        num_voters_to_activate=params.get('num_voters_to_activate', '1'),
        num_candidates_to_activate=params.get('num_candidates_to_activate', '1'),
        d=params.get('d', 0.2),
        mu=params.get('mu', 0.5)
    )
    return model


def run_model(model, time_steps):
    """
    Run the model for the number of time steps passed in the time_steps argument
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
    params = {
        'num_voters': 100,
        'num_candidates': 5,
        'width': 1,
        'height': 1,
        'num_opinions': 2,
        'voter_interaction_fn': 'bc',
        'candidate_interaction_fn': 'bc',
        'num_voters_to_activate': 1,
        'num_candidates_to_activate': 1,
        'threshold': 0.2,
        'mu': 0.5,
    }

    model = build_model(params)
    run_model(model, time_steps=4000)
    save_data(model)

