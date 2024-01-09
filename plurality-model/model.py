import mesa
import numpy as np

class Voter(mesa.Agent):
    """
    An agent representing the behaviour of a voter.
    """

    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        super().__init__(unique_id, model)

        # 2d vector representing the opinion of the voter
        self.opinion = np.zeroes(2)


    def step(self):
        """
        TODO: Add logic for voter to cast their vote to the nearest candidate
        TODO: Determine how a voter moves between elections
        """
        pass


    def nearest_candidate(self):
        """
        Returns the nearest candidate to the Voter.
        """


class Candidate(mesa.Agent):
    """
    An agent representing the behaviour of a cadndiate.
    """


    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        super().__init__(unique_id, model)

        # 2d vector representing the opinion of the candidate
        self.opinion = np.zeroes(2)

        # number of votes this candidate has, initialized to 0
        self.votes = 0


    def step(self):
        """
        TODO: Determine how a candidate moves between elections
        """
        pass


class ElectionSystem(mesa.Model):
    """
    The model representing the plurality election system.
    """


    def __init__(self, num_voters, num_candidates, width, height):
        super().__init__()
        self.num_agents = num_voters + num_candidates
        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(width=width, height=height, torus=True)

        # initialize voters
        for i in range(self.num_voters):
            agent = Voter(i, self)
            self.schedule.add(agent)

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

            # voter opinion is dictated by their location
            agent.opinion = np.array([x, y])

        # initialize candidates
        for i in range(self.num_candidates):
            agent = Candidate(i, self)
            self.schedule.add(agent)

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

            # voter opinion is dictated by their location
            agent.opinion = np.array([x, y])

        # data collector
        self.datacollector = mesa.datacollection.DataCollector()

        self.running = True
        self.datacollector.collect(self)


    def step(self):
        """
        TODO: Count each candidates votes to determine winner.
        """
        self.datacollector.collect(self)
        self.schedule.step()

