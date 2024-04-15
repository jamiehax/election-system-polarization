from model import ElectionSystem
import os
from tqdm import tqdm

def main():
    model = ElectionSystem(
        #seed=0,
        num_voters=250,
        num_voters_to_activate=1,
        initial_num_candidates=3,
        min_candidates=3,
        max_candidates=5,
        term_limit=2,
        num_opinions=1,
        election_system='plurality', # plurality, rc, score
        voter_voter_interaction_fn='ar', # avg, bc, bc1, ar
        voter_candidate_interaction_fn='ar', # avg, bc, bc1, ar
        voter_noise_factor=0.001,
        initial_exit_probability=0.33,
        exit_probability_decrease_factor=0.25,     
        initial_threshold=0.3,
        threshold_increase_factor=0.1,
        num_candidates_to_benefit=2,
        num_rounds_before_election=350,
        mu=0.25, # mu in bounded confidence (controls magnitude of opinion update)
        radius=0.1, # r in radius of support function (controls inflection point)
        learning_rate=0.000001, # learning rate for candidate gradient ascent
        gamma=10, # gamma in radius of support function (controls steepness which is effectively variation in voting probabilities)
        beta=1,
        second_choice_weight_factor=0.5,
        exposure=0.2, # E in AR model (controls probability of interacting)
        responsiveness=0.25 # R in AR model (controls magnitude of opinion update)
    )

    run_model(model, time_steps=5000)
    save_data(model)


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