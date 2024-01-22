from model import ElectionSystem

"""
THINGS WE WANT TO BE MODULAR (I.E. WE CAN CONTROL THROUGH PARAMETERS)
    - number of voters
    - number of candidates
    - the size of opinion space
    - number of opinion dimensions
    - the election system (plurality, ranked choice, cardinal)
    - the method for agents to find other agents to interact with
    - the method for how agents interact with other agents
"""

# ElectionSystem(num_voters, num_candidates, width,  height)
model = ElectionSystem(100, 5, x_max=5, y_max=5)
for i in range(100):
    model.step()

# save data
model.datacollector.get_model_vars_dataframe().to_csv('model_data.csv')
model.datacollector.get_agent_vars_dataframe().to_csv('agent_data.csv')