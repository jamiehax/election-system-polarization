import mesa
import numpy as np
import math

class Voter(mesa.Agent):
    """
    An agent representing the behavior of a voter.

    uid: the unique ID of the voter
    opinion: the array representing the voter's opinions as well as their position
    voted_for: who the candidate voted for
    """

    def __init__(self, uid, model):
        super().__init__(uid, model)
        self.type = 'voter'
        self.unique_id = uid
        self.opinion = np.zeros(2)
        self.pos
        self.voted_for = None


    def step(self):
        """
        This function gets called at every time step in the model.
        """
        candidate = self.get_nearest_candidate()
        self.voted_for = candidate
        if candidate:
            candidate.num_votes += 1
        self.move()


    def move(self):
        """
        TODO: Determine how a voter moves between elections
        """
        pass


    def get_nearest_candidate(self):
        """
        Returns the nearest candidate to the Voter.
        """
        candidates = self.model.agents[Candidate]
        min_candidate = None
        min_dist = math.inf
        for candidate in candidates:
            dist = np.linalg.norm(self.opinion - candidate.opinion)
            if dist < min_dist:
                min_dist = dist
                min_candidate = candidate

        return min_candidate



class Candidate(mesa.Agent):
    """
    An agent representing the behavior of a cadndiate.

    uid: the unique ID of the candidate
    opinion: the array representing the candidate's opinions as well as their position
    num_votes: the number of votes that candidate recieved
    """


    def __init__(self, uid, model):
        super().__init__(uid, model)
        self.type = 'candidate'
        self.unique_id = uid
        self.opinion = np.zeros(2)
        self.pos
        self.num_votes = 0


    def step(self):
        """
        This function gets called at every time step in the model.
        """
        self.move()


    def move(self):
        """
        TODO: Determine how a candidate moves between elections
        """
        pass



class ElectionSystem(mesa.Model):
    """
    The model representing the plurality election system.

    num_agents: total number of agents
    num_voters: number of voting agents
    num_candidates: number of candidate agents
    winner: the winning candidate of the election
    agents: all agents in model stored as a dict of {"Agent Type":[agent1, agent2, ..., agentN]}
    """


    def __init__(self, num_voters, num_candidates, **kwargs):
        super().__init__()
        
        # get key word arguments
        width = kwargs.get('x_max', 100)
        height = kwargs.get('y_max', 100)
        opinions = kwargs.get('opinions', None)

        self.num_agents = num_voters + num_candidates
        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.winner = None
        self.agents = {}
        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(x_max=width, y_max=height, torus=False)

        # initialize voters
        self.agents[Voter] = []
        for i in range(self.num_voters):
            voter = Voter(i, self)
            self.schedule.add(voter)
            self.agents[Voter].append(voter)

            # set voter opinion and location
            if opinions: 
                opinion = opinions[i]
            else:
                opinion = (self.random.randrange(self.space.width), self.random.randrange(self.space.height))
            
            voter.pos = opinion
            self.space.place_agent(voter, opinion)
            voter.opinion = np.array(opinion)

        # initialize candidates
        self.agents[Candidate] = []
        for i in range(self.num_voters, self.num_agents):
            candidate = Candidate(i, self)
            self.schedule.add(candidate)
            self.agents[Candidate].append(candidate)

            # set candidate opinion and location
            if opinions: 
                opinion = opinions[i]
            else:
                opinion = (self.random.randrange(self.space.width), self.random.randrange(self.space.height))
            
            candidate.pos = opinion
            self.space.place_agent(candidate, opinion)
            candidate.opinion = np.array(opinion)


        # data collector
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters = {"winner": "winner"},
            agent_reporters = {
                'voted_for': lambda a: a.voted_for if a.type == 'voter' else None,
                'num_votes': lambda a: a.num_votes if a.type == 'candidate' else None
            }
        )

        self.running = True
        self.datacollector.collect(self)


    def step(self):
        """
        This function gets called at every step in the model.
        """
        self.schedule.step()
        self.datacollector.collect(self)
        
        # count votes to determine winner
        candidates = self.agents[Candidate]
        most_votes = 0
        for candidate in candidates:
            if candidate.num_votes > most_votes:
                self.winner = candidate
                most_votes = candidate.num_votes

        print(self.winner.unique_id)
