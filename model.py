import mesa
import numpy as np


class Voter(mesa.Agent):
    """
    An agent representing the behavior of a voter.

    uid: the unique ID of the voter
    opinion: the array representing the voter's opinions as well as their position
    voted_for: who the candidate voted for
    """

    def __init__(self, uid, model, opinion, pos, voter_interaction_fn, candidate_interaction_fn):
        super().__init__(uid, model)
        self.type = 'voter'
        self.unique_id = uid
        self.voted_for = None
        self.opinion = opinion
        self.pos = pos
        self.voter_interaction_fn = voter_interaction_fn
        self.candidate_interaction_fn = candidate_interaction_fn
        self.get_interaction_agents = self.model.get_random_agent_set


    def voters_move(self):
        """
        How voters move between elections. Determined by:
            - who they interact with (self.get_interaction_agents)
            - how they interact with other agents (self.candidate_interaction_fn and self.voter_interaction_fn)
        """
        if self.unique_id in self.model.voters_to_activate:
            interaction_agents = self.get_interaction_agents(num_voters=1, num_candidates=1)

            # Voter-Candidate interaction
            if self.candidate_interaction_fn:
                for candidate in interaction_agents['candidates']:
                    self.candidate_interaction_fn(a1=self, a2=candidate)

            # Voter-Voter interaction
            if self.voter_interaction_fn:
                for voter in interaction_agents['voters']:
                    self.voter_interaction_fn(a1=self, a2=voter)


    def vote(self):
        """
        Increase the vote count for the closest candidate by Euclidean distance.
        """
        candidates = self.model.agents[Candidate]
        distances = [np.linalg.norm(self.opinion - candidate.opinion) for candidate in candidates]
        candidate_distances = list(zip(candidates, distances))
        candidate_distances.sort(key = lambda x: x[1]) 
        if candidate_distances[0]:
            self.voted_for = candidate_distances[0][0].unique_id
            candidate_distances[0][0].num_votes += 1        


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

    def __init__(self, uid, model, opinion, pos, voter_interaction_fn, candidate_interaction_fn):
        super().__init__(uid, model)
        self.type = 'candidate'
        self.unique_id = uid
        self.opinion = opinion
        self.pos = pos
        self.voted_for = uid
        self.num_votes = 0
        self.is_winner = False
        self.voter_interaction_fn = voter_interaction_fn
        self.candidate_interaction_fn = candidate_interaction_fn
        self.get_interaction_agents = self.model.get_random_agent_set


    def candidates_move(self):
        """
        How candidates move between elections. Determined by:
            - who they interact with (self.get_interaction_agents)
            - how they interact with other agents (self.update_opinion)
        """
        if self.unique_id in self.model.candidates_to_activate:
            interaction_agents = self.get_interaction_agents(num_voters=1, num_candidates=1)

            # Candidate-Candidate interaction
            if self.candidate_interaction_fn:
                for voter in interaction_agents['candidates']:
                    self.candidate_interaction_fn(a1=self, a2=voter)
            
            # Candidate-Voter interaction
            if self.voter_interaction_fn:
                for voter in interaction_agents['voters']:
                    self.voter_interaction_fn(a1=self, a2=voter)


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

    def __init__(self, **kwargs):
        super().__init__()

        num_voters = kwargs.get('num_voters', 100)
        num_candidates = kwargs.get('num_candidates', 5)
        width = kwargs.get('width', 1)
        height = kwargs.get('height', 1)
        num_opinions = kwargs.get('num_opinions', 2)
        voter_voter_interaction_fn = kwargs.get('voter_voter_interaction_fn', None)
        voter_candidate_interaction_fn = kwargs.get('voter_candidate_interaction_fn', None)
        candidate_voter_interaction_fn = kwargs.get('candidate_voter_interaction_fn', None)
        candidate_candidate_interaction_fn = kwargs.get('candidate_candidate_interaction_fn', None)
        num_voters_to_activate = kwargs.get('num_voters_to_activate', 1)
        num_candidates_to_activate = kwargs.get('num_candidates_to_activate', 1)
        threshold = kwargs.get('threshold', 0.2)
        mu = kwargs.get('mu', 0.5)

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
        self.threshold = threshold
        self.mu = mu

        # agent interaction functions
        interaction_functions = {
            'avg': self.average_opinion,
            'bc': self.bounded_confidence,
            'kmean': self.k_mean
        }

        # initialize voters
        self.agents[Voter] = []
        for i in range(self.num_voters):
            # get initial voter opinion and location
            opinion = np.array([self.random.uniform(0, min([width, height])) for _ in range(self.num_opinions)])
            if self.num_opinions == 2:
                pos = (opinion[0], opinion[1])
            else:
                pos = (0, 0) # if opinion space dimensions != 2 we cant visualize anyways - just to make Mesa happy

            # initialize voter
            voter = Voter(i, self, opinion, pos, interaction_functions.get(voter_voter_interaction_fn, None), interaction_functions.get(voter_candidate_interaction_fn, None))
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
                pos = (0, 0) # if opinion space dimensions != 2 we cant visualize anyways - just to make Mesa happy

            # initialize candidate
            candidate = Candidate(i, self, opinion, pos, interaction_functions.get(candidate_voter_interaction_fn, None), interaction_functions.get(candidate_candidate_interaction_fn, None))
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
            'is_winner': lambda a: a.is_winner if a.type == 'candidate' else None,
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

        # move agents
        self.schedule.step()

        # find winner
        winner = self.plurality_count()
        winner.is_winner = True
        self.winner = winner.unique_id

        # collect data for round
        self.datacollector.collect(self)

        # clear winner for next round
        if self.winner: next((candidate for candidate in self.agents[Candidate] if candidate.unique_id == self.winner), None).is_winner = False


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


    def average_opinion(self, **kwargs):
        """
        Update agent a1's opinion to be the average of a1's position and a2's opinion.
        """
        a1 = kwargs.get('a1', None)
        a2 = kwargs.get('a2', None)
        if isinstance(a1, mesa.Agent) and isinstance(a2, mesa.Agent) and a1 and a2:
            a1.opinion = (a1.opinion + a2.opinion) / 2
        else:
            raise TypeError('Invalid parameter type. The average opinion update function accepts exactly two agents as parameters.')
    

    def bounded_confidence(self, **kwargs):
        """
        Update agent a1 and a2's opinions according to the bounded confidence model.
        """
        a1 = kwargs.get('a1', None)
        a2 = kwargs.get('a2', None)
        if isinstance(a1, mesa.Agent) and isinstance(a2, mesa.Agent) and a1 and a2:
            if np.linalg.norm(a1.opinion - a2.opinion) < self.threshold:
                a1.opinion = a1.opinion + (self.mu * (a2.opinion - a1.opinion))   
                a2.opinion = a2.opinion + (self.mu * (a1.opinion - a2.opinion))
        else:
            raise TypeError('Invalid parameter type. The average opinion update function accepts exactly two agents as parameters.')

    
    def k_mean(self, **kwargs):
        """
        Update a1 (candidate) opinion to be the mean of their voting bloc
        """
        candidate = kwargs.get('a1', None)
        if isinstance(candidate, Candidate):
            voters = self.agents[Voter]
            voting_bloc = [voter for voter in voters if voter.voted_for == candidate.unique_id]
            if len(voting_bloc) > 0:
                opinion_sum = sum([voter.opinion for voter in voting_bloc])
                candidate.opinion =  np.array(opinion_sum / len(voting_bloc))
            else:
                return
        else:
            raise TypeError('Invalid parameter type. The k_mean opinion update function is only implemented for Candidates.')
    

    def get_random_agent_set(self, num_voters, num_candidates):
        """
        Returns a random subset of agents to interact with as a dict.

        returns: {"AgentType": [Agent1, Agent2, ..., AgentN]}
        """
        candidates = self.agents[Candidate]
        voters = self.agents[Voter]

        if num_candidates == self.num_candidates:
            interaction_candidates = candidates
        else:
            interaction_candidates = self.random.sample(candidates, num_candidates)
        
        if num_voters == self.num_voters:
            interaction_voters = candidates
        else:
            interaction_voters = self.random.sample(voters, num_voters)

        interaction_agents = {
            'candidates': interaction_candidates,
            'voters': interaction_voters
        }
        return interaction_agents
