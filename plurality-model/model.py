import mesa
import numpy as np


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
        self.interaction_function = self.model.average
        self.get_interaction_agents = self.model.get_random_agent_set


    def voters_move(self):
        """
        Right now Voters move by:
            - interacting with candidates first in some random order and taking the average opinion of each interaction
            - interacting with voters second in some random order and taking the average opinion of each interaction
        """
        interaction_agents = self.get_interaction_agents()

        # Voter-Candidate interaction
        for candidate in interaction_agents['candidates']:
            new_opinion = self.interaction_function(self, candidate)
            self.opinion = new_opinion
            self.pos = self.opinion

        # Voter-Voter interaction
        for voter in interaction_agents['voters']:
            new_opinion = self.interaction_function(self, voter)
            self.opinion = new_opinion
            self.pos = self.opinion


    def vote(self):
        candidates = self.get_candidate_distances()
        if candidates[0]:
            self.voted_for = candidates[0][0].unique_id
            candidates[0][0].num_votes += 1

    
    def get_candidate_distances(self):
        """
        Returns a list of tuples of candidates and their distances from the voter: 
        [(Candidate1, dist1), (Candidate2, dist2), ..., (Candidate3, dist3)]
        """
        candidates = self.model.agents[Candidate]
        distances = [np.linalg.norm(self.opinion - candidate.opinion) for candidate in candidates]
        candidate_distances = list(zip(candidates, distances))
        candidate_distances.sort(key = lambda x: x[1]) 
        return candidate_distances


    def candidates_move(self):
        """
        This method controls how Candidates move around in opinion space.
        Mesa requires it be implemented for all agents - for Voter agents it does nothing.
        """
        return



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
        self.interaction_function = self.model.average
        self.get_interaction_agents = self.model.get_random_agent_set


    def candidates_move(self):
        """
        Right now Candidates move by:
            - interacting with voters in some random order and taking the average opinion of each interaction
        """
        interaction_agents = self.get_interaction_agents()

        # Voter-Voter interaction
        for voter in interaction_agents['voters']:
            new_opinion = self.interaction_function(self, voter)
            self.opinion = new_opinion
            self.pos = self.opinion


    def vote(self):
        """
        This method controls how Voters cast their vote.
        Mesa requires it be implemented for all agents - for Candidate agents it does nothing.
        """
        return

    
    def voters_move(self):
        """
        This method controls how Voters move around in opinion space.
        Mesa requires it be implemented for all agents - for Candidate agents it does nothing.
        """
        return



class ElectionSystem(mesa.Model):
    """
    The model representing the plurality election system.

    num_agents: total number of agents
    num_voters: number of voting agents
    num_candidates: number of candidate agents
    winner: the winning candidate of the election
    agents: all agents in model stored as a dict of {"AgentType":[agent1, agent2, ..., agentN]}
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
        self.schedule = mesa.time.StagedActivation(
            self,
            stage_list=['candidates_move', 'voters_move', 'vote'],
            shuffle_between_stages=True
        )
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
            model_reporters = {
                'winner': 'winner',
            },
            agent_reporters = {
                'voted_for': lambda a: a.voted_for if a.type == 'voter' else None,
                'num_votes': lambda a: a.num_votes if a.type == 'candidate' else None,
                'type': 'type',
                'opinion': 'opinion'
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
        
        self.winner = self.plurality_count().unique_id
        print(self.winner)


    def plurality_count(self):
        """
        Returns the winner of the election for plurality count.
        """
        candidates = self.agents[Candidate]
        most_votes = 0
        winner = None
        for candidate in candidates:
            if candidate.num_votes > most_votes:
                winner = candidate
                most_votes = candidate.num_votes

        return winner


    def average(self, a1, a2):
        """
        Helpfer function to compute the average opinion of two agents.
        """
        return (a1.opinion + a2.opinion) / 2
    

    def get_random_agent_set(self):
        """
        Returns a random subset of agents to interact with as {"AgentType": [Agent1, Agent2, ..., AgentN]}
        """
        candidates = self.agents[Candidate]
        voters = self.agents[Voter]
        interaction_agents = {
            'candidates': candidates,
            'voters': self.random.sample(voters, len(voters) // 4)
        }
        return interaction_agents
