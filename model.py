import mesa
import numpy as np


class Voter(mesa.Agent):
    """
    An agent representing the behavior of a voter.

    uid: the unique ID of the voter
    opinion: the array representing the voter's opinions as well as their position
    voted_for: who the candidate voted for
    """

    def __init__(self, uid, model, opinion, pos, interaction_function):
        super().__init__(uid, model)
        self.type = 'voter'
        self.unique_id = uid
        self.voted_for = None
        self.opinion = opinion
        self.pos = pos
        self.update_opinion = interaction_function
        self.get_interaction_agents = self.model.get_random_agent_set


    def voters_move(self):
        """
        How voters move between elections. Determined by:
            - who they interact with (self.get_interaction_agents)
            - how they interact with other agents (self.update_opinion)
        """
        if self.unique_id in self.model.voters_to_activate:
            interaction_agents = self.get_interaction_agents(num_voters=1, num_candidates=1)

            # Voter-Candidate interaction
            for candidate in interaction_agents['candidates']:
                self.update_opinion(self, candidate)

            # Voter-Voter interaction
            for voter in interaction_agents['voters']:
                self.update_opinion(self, voter)


    def vote(self):
        candidates = self.get_candidates()
        if candidates[0]:
            self.voted_for = candidates[0][0].unique_id
            candidates[0][0].num_votes += 1

    
    def get_candidates(self):
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
    """


    def __init__(self, uid, model, opinion, pos, interaction_function):
        super().__init__(uid, model)
        self.type = 'candidate'
        self.unique_id = uid
        self.opinion = opinion
        self.pos = pos
        self.voted_for = uid
        self.num_votes = 0
        self.update_opinion = interaction_function
        self.get_interaction_agents = self.model.get_random_agent_set


    def candidates_move(self):
        """
        How candidates move between elections. Determined by:
            - who they interact with (self.get_interaction_agents)
            - how they interact with other agents (self.update_opinion)
        """
        if self.unique_id in self.model.candidates_to_activate:
            interaction_agents = self.get_interaction_agents(num_voters=1, num_candidates=0)

            # Candidate-Voter interaction
            for voter in interaction_agents['candidates']:
                self.update_opinion(self, voter)
            
            # Candidate-Voter interaction
            for voter in interaction_agents['voters']:
                self.update_opinion(self, voter)


    def vote(self):
        """
        Assume candidate votes for themselves.
        """
        self.num_votes += 1
        self.voted_for = self.unique_id

    
    def voters_move(self):
        """
        This method controls how Voters move around in opinion space.
        Mesa requires it be implemented for all agents - for Candidate agents it does nothing.
        """
        return



class ElectionSystem(mesa.Model):
    """
    The model representing the election system.
    """


    def __init__(self, num_voters, num_candidates, width, height, num_opinions, voter_interaction_fn, candidate_interaction_fn, num_voters_to_activate, num_candidates_to_activate, d, mu):
        super().__init__()

        # GENERAL MODEL ATTRIBUTES
        self.num_agents = num_voters + num_candidates
        self.num_voters = num_voters
        self.num_voters_to_activate = num_voters_to_activate
        self.num_candidates = num_candidates
        self.num_candidates_to_activate = num_candidates_to_activate
        self.num_opinions = num_opinions
        self.winner = None
        self.voters_to_activate = []
        self.canddiates_to_activate = []
        self.agents = {}
        self.space = mesa.space.ContinuousSpace(x_max=width, y_max=height, torus=False)
        self.schedule = mesa.time.StagedActivation(
            self,
            stage_list=['candidates_move', 'voters_move', 'vote'],
            shuffle_between_stages=True
        )

        # bounded confidence parameters
        self.d = d
        self.mu = mu

        # available agent interaction functions
        interaction_functions = {
            'avg': self.average_opinion,
            'bc': self.bounded_confidence
        }

        # initialize voters
        self.agents[Voter] = []
        for i in range(self.num_voters):
            # get initial voter opinion and location
            opinion = np.array([self.random.uniform(0, min([width, height])) for _ in range(self.num_opinions)])
            if self.num_opinions == 2: 
                pos = (opinion[0], opinion[1])
            else:
                pos = (0, 0)

            # initialize voter
            voter = Voter(i, self, opinion, pos, interaction_functions[voter_interaction_fn])
            self.space.place_agent(voter, pos)
            self.schedule.add(voter)
            self.agents[Voter].append(voter)

        # initialize candidates
        self.agents[Candidate] = []
        for i in range(self.num_voters, self.num_agents):
            # get initial candidate opinion and location
            opinion = np.array([self.random.uniform(0, min([width, height])) for _ in range(self.num_opinions)])
            if self.num_opinions == 2: 
                pos = (opinion[0], opinion[1])
            else:
                pos = (0, 0)

            # initialize candidate
            candidate = Candidate(i, self, opinion, pos, interaction_functions[candidate_interaction_fn])
            self.space.place_agent(candidate, pos)
            self.schedule.add(candidate)
            self.agents[Candidate].append(candidate)


        # data collector
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters = {
                'winner': 'winner',
            },
            agent_reporters = self.get_agent_reporter_dict()
        )

        self.running = True
        self.datacollector.collect(self)


    def get_agent_reporter_dict(self):
        agent_reporters = {
            'voted_for': 'voted_for',
            'num_votes': lambda a: a.num_votes if a.type == 'candidate' else None,
            'type': 'type',
        }
        for d in range(self.num_opinions):
            agent_reporters['opinion'+str(d+1)] = lambda a, d=d: a.opinion[d]
        return agent_reporters
        

    def step(self):
        """
        This function gets called at every step in the model.
        """

        # randomly select agents to interact
        self.voters_to_activate = self.random.sample([a.unique_id for a in self.agents[Voter]], self.num_voters_to_activate)
        self.candidates_to_activate = self.random.sample([a.unique_id for a in self.agents[Candidate]], self.num_candidates_to_activate)

        self.schedule.step()
        self.datacollector.collect(self)
        self.winner = self.plurality_count().unique_id


    def plurality_count(self):
        """
        Returns the winner of the election for plurality count, and resets all candidate vote counts.
        """
        candidates = self.agents[Candidate]
        most_votes = 0
        winner = None
        for candidate in candidates:
            if candidate.num_votes > most_votes:
                winner = candidate
                most_votes = candidate.num_votes

        for candidate in candidates:
            candidate.num_votes = 0

        return winner


    def average_opinion(self, a1, a2):
        """
        Update agent a1's opinion as the average of their position and a2's opinion.
        """
        a1.opinion = (a1.opinion + a2.opinion) / self.num_opinions
    

    def bounded_confidence(self, a1, a2):
        """
        Update agent a1 and a2's opinions according to the bounded confidence model.
        """
        if np.linalg.norm(a1.opinion - a2.opinion) < self.d:
            a1.opinion = a1.opinion + (self.mu * (a2.opinion - a1.opinion))   
            a2.opinion = a2.opinion + (self.mu * (a1.opinion - a2.opinion))   
    

    def get_random_agent_set(self, num_voters, num_candidates):
        """
        Returns a random subset of agents to interact with as a dict.

        returns: {"AgentType": [Agent1, Agent2, ..., AgentN]}
        """
        candidates = self.agents[Candidate]
        voters = self.agents[Voter]
        interaction_agents = {
            'candidates': self.random.sample(candidates, num_candidates),
            'voters': self.random.sample(voters, num_voters)
        }
        return interaction_agents
