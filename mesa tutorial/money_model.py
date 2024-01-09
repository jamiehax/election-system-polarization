import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

class MoneyAgent(mesa.Agent):
    """An Agent with fixed inital Wealth"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.wealth = 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore = True,
            include_center = False,
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        cellmates.pop(cellmates.index(self))  # Ensure agent is not giving money to itself
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1


class MoneyModel(mesa.Model):
    """A Model with some number of Agents"""

    def __init__(self, N, width, height):
        self.num_agents = N

        # create scheduler
        self.schedule = mesa.time.RandomActivation(self)

        # create grtid
        self.grid = mesa.space.MultiGrid(width, height, True)

        # create data collector
        self.datacollector = mesa.DataCollector(
            model_reporters = {"Gini": self.compute_gini}, 
            agent_reporters = {"Wealth": "wealth"}
        )

        # variable for conditional shutoff of batch_run
        self.running = True
        
        # create agents
        for i in range(self.num_agents):
            agent = MoneyAgent(i, self)

            # add agent to scheduler
            self.schedule.add(agent)

            # add agent to grid
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        
    def compute_gini(self):
        agent_wealths = [agent.wealth for agent in self.schedule.agents]
        x = sorted(agent_wealths)
        N = self.num_agents
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
        return 1 + (1 / N) - 2 * B

    def step(self):
        """Advance model by one step"""

        self.schedule.step()

        # collect data
        self.datacollector.collect(self)