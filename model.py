import mesa
import numpy as np
import math
import torch


class Voter(mesa.Agent):
    """
    An agent representing the behavior of a voter.

    uid: the unique ID of the voter
    opinion: the array representing the voter's opinions as well as their position
    voted_for: who the candidate voted for
    """

    def __init__(self, uid, model, opinion, threshold, voter_interaction_fn, candidate_interaction_fn):
        super().__init__(uid, model)
        self.type = 'voter'
        self.unique_id = uid
        self.voted_for = None
        self.opinion = opinion
        self.threshold = threshold
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
                    self.candidate_interaction_fn(a1=self, a2=candidate, threshold=candidate.threshold)

            # Voter-Voter interaction
            if self.voter_interaction_fn:
                for voter in interaction_agents['voters']:
                    self.voter_interaction_fn(a1=self, a2=voter)     


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

    def __init__(self, uid, model, opinion, threshold, voter_interaction_fn, candidate_interaction_fn, exit_probability):
        super().__init__(uid, model)
        self.type = 'candidate'
        self.unique_id = uid
        self.opinion = opinion
        self.threshold = threshold
        self.voted_for = uid
        self.num_votes = 0
        self.is_winner = False
        self.num_wins = 0
        self.exit_probability = exit_probability
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
            interaction_agents = self.get_interaction_agents(num_voters=1, num_candidates=0)

            # Candidate-Candidate interaction
            if self.candidate_interaction_fn:
                for candidate in interaction_agents['candidates']:
                    self.candidate_interaction_fn(a1=self, a2=candidate)
            
            # Candidate-Voter interaction
            if self.voter_interaction_fn:
                for voter in interaction_agents['voters']:
                    self.voter_interaction_fn(a1=self, a2=voter, threshold=self.threshold)

    
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

    def __init__(self, seed=None, **kwargs):
        super().__init__()

        num_voters = kwargs.get('num_voters', 100)
        initial_num_candidates = kwargs.get('initial_num_candidates', 5)
        term_limit = kwargs.get('term_limit', 2)
        num_opinions = kwargs.get('num_opinions', 2)
        election_system = kwargs.get('election_system', 'plurality')
        voter_voter_interaction_fn = kwargs.get('voter_voter_interaction_fn', None)
        voter_candidate_interaction_fn = kwargs.get('voter_candidate_interaction_fn', None)
        candidate_voter_interaction_fn = kwargs.get('candidate_voter_interaction_fn', None)
        candidate_candidate_interaction_fn = kwargs.get('candidate_candidate_interaction_fn', None)
        num_voters_to_activate = kwargs.get('num_voters_to_activate', 1)
        num_candidates_to_activate = kwargs.get('num_candidates_to_activate', 1)
        initial_exit_probability = kwargs.get('initial_exit_probability', 0.5)
        exit_probability_decrease_factor = kwargs.get('exit_probability_decrease_factor', 0.5)
        min_candidates = kwargs.get('min_candidates', 2)
        max_candidates = kwargs.get('max_candidates', 5)
        initial_threshold = kwargs.get('initial_threshold', 0.2)
        threshold_increase_factor = kwargs.get('threshold_increase_factor', 2)
        num_candidates_to_benefit = kwargs.get('num_candidates_to_benefit', 2)
        mu = kwargs.get('mu', 0.5)
        num_rounds_before_election = kwargs.get('num_rounds_before_election', 1)
        beta = kwargs.get('beta', 0.2)
        C = kwargs.get('C', 5)
        gamma = kwargs.get('gamma', 0.2)
        radius = kwargs.get('radius', 1)
        learning_rate = kwargs.get('learning_rate', 0.1)

        # GENERAL MODEL ATTRIBUTES
        self.num_agents = num_voters + initial_num_candidates
        self.num_voters = num_voters
        self.num_voters_to_activate = num_voters_to_activate
        self.initial_num_candidates = initial_num_candidates
        self.num_candidates_to_activate = num_candidates_to_activate
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.term_limit = term_limit
        self.num_opinions = num_opinions
        self.num_rounds_before_election = num_rounds_before_election
        self.winner = None
        self.voters_to_activate = []
        self.candidates_to_activate = []
        self.agents = {}
        self.schedule = mesa.time.StagedActivation(
            self,
            stage_list=['candidates_move', 'voters_move'],
            shuffle_between_stages=True
        )

        # for adding candidates in later on
        self.candidate_index = num_voters
        self.initial_exit_probability = initial_exit_probability
        self.candidate_voter_interaction_fn = candidate_voter_interaction_fn
        self.candidate_candidate_interaction_fn = candidate_candidate_interaction_fn
        self.exit_probability_decrease_factor = exit_probability_decrease_factor

        # bounded confidence parameters
        self.initial_threshold = initial_threshold
        self.threshold_increase_factor = threshold_increase_factor
        self.num_candidates_to_benefit = num_candidates_to_benefit
        self.mu = mu

        # candidate ascent parameters
        self.C = C
        self.beta = beta
        self.radius = radius
        self.gamma = gamma
        self.learning_rate = learning_rate

        # agent interaction functions
        self.interaction_functions = {
            'avg': self.average_opinion,
            'bc': self.bounded_confidence,
            'kmean': self.k_mean,
            'ascent': self.candidate_ascent
        }

        # election systems
        election_systems = {
            'plurality': self.plurality_count,
            'rc': self.ranked_choice,
            'score': self.score_voting
        }
        self.election_system = election_systems[election_system]

        # initialize voters
        self.agents[Voter] = []
        for i in range(self.num_voters):
            # get initial voter opinion and location
            opinion = np.array([self.random.uniform(0, 1) for _ in range(self.num_opinions)])

            # initialize voter
            voter = Voter(
                i,
                self,
                opinion,
                self.initial_threshold,
                self.interaction_functions.get(voter_voter_interaction_fn, None),
                self.interaction_functions.get(voter_candidate_interaction_fn, None)
            )
            self.schedule.add(voter)
            self.agents[Voter].append(voter)

        # initialize candidates
        self.agents[Candidate] = []
        for i in range(self.num_voters, self.num_agents):
            # get initial candidate opinion and location
            opinion = np.array([self.random.uniform(0, 1) for _ in range(self.num_opinions)])

            # initialize candidate
            candidate = Candidate(
                i,
                self,
                opinion,
                self.initial_threshold,
                self.interaction_functions.get(candidate_voter_interaction_fn, None),
                self.interaction_functions.get(candidate_candidate_interaction_fn, None),
                initial_exit_probability
            )
            self.schedule.add(candidate)
            self.agents[Candidate].append(candidate)
            self.candidate_index += 1


        # data collector
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters = {
                'winner': 'winner.unique_id',
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
            'num_wins': lambda a: a.num_wins if a.type == 'candidate' else None,
            'threshold': lambda a: a.threshold if a.type == 'candidate' else None,
            'exit_probability': lambda a: a.exit_probability if a.type == 'candidate' else None,
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
        self.candidates_to_activate = self.random.sample([a.unique_id for a in self.agents[Candidate]], min(len(self.agents[Candidate]), self.num_candidates_to_activate))

        # move agents
        self.schedule.step()

        # find winner only on election round
        if self.schedule.steps % self.num_rounds_before_election == 0:

            # clear old winner
            if self.winner: next((candidate for candidate in self.agents[Candidate] if candidate is self.winner), None).is_winner = False
            self.winner = None

            # remove term limit candidates:
            self.remove_term_limit_candidates()

            # find new winner
            winner = self.election_system()
            winner.is_winner = True
            winner.num_wins += 1
            self.winner = winner

            self.candidate_ascent()

            # add and remove candidates
            self.change_candidates(winner)

        # collect data for round
        self.datacollector.collect(self)


    def remove_term_limit_candidates(self):
        """
        Remove all candidates who have hit their term limit
        """
        for candidate in self.agents[Candidate]:
            if candidate.num_wins >= self.term_limit:
                self.agents[Candidate].remove(candidate)
                self.schedule.remove(candidate)


    def change_candidates(self, winner):
        """
        Stochastically remove candidates based on their exit probabilities, winner candidate is excluded.
        Stochastically add candidates into the race, satisfying min and max candidate requirements.
        """
        candidates = self.agents[Candidate].copy()
        self.random.shuffle(candidates)

        # randomly remove candidates based on their exit probability, but not fewer than min num of candidates
        for candidate in candidates:
            if candidate is not winner:
                if len(candidates) >= self.min_candidates:
                    if self.random.random() < candidate.exit_probability:
                        candidates.remove(candidate)
                        self.agents[Candidate].remove(candidate)
                        self.schedule.remove(candidate)

        # randomly add candidates to satisfy min max restraints
        current_num_candidates = len(candidates)
        if len(candidates) < self.min_candidates:
            num_candidates_to_add = self.random.randint(self.min_candidates - current_num_candidates, self.max_candidates - current_num_candidates)
        elif len(candidates) < self.max_candidates and len(candidates) >= self.min_candidates:
            num_candidates_to_add = self.random.randint(0, self.max_candidates - current_num_candidates)
        else:
            num_candidates_to_add = 0

        for _ in range(num_candidates_to_add):
            opinion = np.array([self.random.uniform(0, 1) for _ in range(self.num_opinions)])
            candidate = Candidate(
                self.candidate_index,
                self,
                opinion,
                self.initial_threshold,
                self.interaction_functions.get(self.candidate_voter_interaction_fn, None),
                self.interaction_functions.get(self.candidate_candidate_interaction_fn, None),
                self.initial_exit_probability
            )
            self.schedule.add(candidate)
            self.agents[Candidate].append(candidate)
            self.candidate_index += 1


    def plurality_count(self):
        """
        Returns the winner of the election for plurality count, and resets all candidate vote counts.
        """
        candidates = self.agents[Candidate]
        voters = self.agents[Voter]

        # tensor of each voters probaility of voting for each candidate
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in voters])
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in candidates])

        # sample voter probabilities
        voter_probabilities = self.vote_probabilities(voter_opinions, candidate_opinions)
        ballots = []
        for voter, prob in zip(voters, voter_probabilities):
            vote = np.random.choice(candidates, size=1, p=prob)[0]
            ballots.append(vote)
            voter.voted_for = vote.unique_id
        
        # get vote counts
        counts = {candidate:0 for candidate in candidates}
        for ballot in ballots:
            counts[ballot] += 1

        # set vote counts for candidates
        for candidate, num_votes in counts.items():
            candidate.num_votes = num_votes

        # change exit probabilities and thresholds
        results = [(candidate, count) for candidate, count in counts.items()]       
        results.sort(key=lambda x: x[1], reverse=True)
        for place, candidate in enumerate(results):
            candidate[0].exit_probability = self.sigmoid_offset(self.exit_probability_decrease_factor * (place - self.num_candidates_to_benefit), candidate[0].exit_probability)
            candidate[0].threshold = self.sigmoid_offset(self.threshold_increase_factor * (self.num_candidates_to_benefit - place), candidate[0].threshold)

        winner = max(counts, key=counts.get)
        return winner


    def ranked_choice(self):
        """
        Returns the winner of the election according to instant runoff ranked choice.
        """
        candidates = self.agents[Candidate]
        voters = self.agents[Voter]

        # tensor of each voters probaility of voting for each candidate
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in voters])
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in candidates])

        # sample voter probabilities
        voter_probabilities = self.vote_probabilities(voter_opinions, candidate_opinions)
        ballots = []
        for voter, prob in zip(voters, voter_probabilities):
            votes = np.random.choice(candidates, size=len(candidates), p=prob, replace=False)
            ballots.append(list(votes))
        
        for candidate in candidates:
            candidate.num_votes = 0

        # list ordered in candidate places
        results = []

        winner = None
        while winner is None:
            counts = {candidate:0 for candidate in candidates}
            for ballot in ballots:
                selected_candidate = ballot[0]
                counts[selected_candidate] += 1

            max_vote_candidate = max(counts, key=counts.get)
            if counts[max_vote_candidate] > ((len(voters) // 2) + 1):
                winner = max_vote_candidate
                for candidate, num_votes in counts.items():
                    candidate.num_votes = num_votes
                for voter, preference in zip(voters, ballots):
                    voter.voted_for = preference[0].unique_id
                winning_round_results = [(candidate, count) for candidate, count in counts.items()]       
                winning_round_results.sort(key=lambda x: x[1])
                for candidate in winning_round_results:
                    results.insert(0, candidate[0])
            else:
                least_vote_candidate = min(counts, key=counts.get)
                results.insert(0, least_vote_candidate)
                candidates.remove(least_vote_candidate)
                for ballot in ballots:
                    ballot.remove(least_vote_candidate)

        # change exit probabilities and thresholds
        for place, candidate in enumerate(results):
            candidate.exit_probability = self.sigmoid_offset(self.exit_probability_decrease_factor * (place - self.num_candidates_to_benefit), candidate.exit_probability)
            candidate.threshold = self.sigmoid_offset(self.threshold_increase_factor * (self.num_candidates_to_benefit - place), candidate.threshold)

        return winner


    def score_voting(self):
        """
        Returns the winner of the election according to score voting.
        """
        candidates = self.agents[Candidate]
        voters = self.agents[Voter]

        # tensor of each voters probaility of voting for each candidate
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in voters])
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in candidates])

        # sample voter probabilities
        voter_probabilities = self.vote_probabilities(voter_opinions, candidate_opinions)
        ballots = []
        for voter, prob in zip(voters, voter_probabilities):
            candidate_probs = list(zip(candidates, prob))
            candidate_probs.sort(key=lambda x: x[1])
            ballots.append(prob)
            voter.voted_for = prob[0][0].unique_id
        
        # ballots = []
        # for voter in voters:
        #     voter_probabilities = self.vote_probabilities(voter)
        #     voter_probabilities.sort(key=lambda x: x[1])
        #     ballots.append(voter_probabilities)
        #     voter.voted_for = voter_probabilities[0][0].unique_id
        
        # get vote scores
        scores = {candidate:0 for candidate in candidates}
        for ballot in ballots:
            for candidate, score in ballot:
                scores[candidate] += score

        # set vote scores for candidates
        for candidate, score in scores.items():
            candidate.num_votes = score

        # normalize scores and change exit probabilities and thresholds
        results = [(candidate, score) for candidate, score in scores.items()]
        for candidate, score in results:
            normalized_score = len(self.agents[Candidate]) - ((score - min(scores.values())) / (max(scores.values()) - min(scores.values())) * len(self.agents[Candidate]))
            candidate.exit_probability = self.sigmoid_offset(self.exit_probability_decrease_factor * (normalized_score - self.num_candidates_to_benefit), candidate.exit_probability)
            candidate.threshold = self.sigmoid_offset(self.threshold_increase_factor * (self.num_candidates_to_benefit - normalized_score), candidate.threshold)


        winner = max(scores, key=scores.get)
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
        threshold = kwargs.get('threshold', self.initial_threshold)
        if isinstance(a1, mesa.Agent) and isinstance(a2, mesa.Agent) and a1 and a2:
            if np.linalg.norm(a1.opinion - a2.opinion) < threshold:
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


    def candidate_ascent(self):
        """
        Move the candidate in the direction of the gradient calculated from the candidates expected number of votes.
        """

        # get tensors of voters and candidate opinions
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in self.agents[Voter]])
        voter_opinions.requires_grad_(True)
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in self.agents[Candidate]])
        candidates = [c for c in self.agents[Candidate]] # list of candidates in same order as gradients
        candidate_opinions.requires_grad_(True)
        
        expected_votes = self.plurality_objective_fn(voter_opinions, candidate_opinions)

        # update candidates positions with gradients
        for candidate_index, candidate in zip(range(len(self.agents[Candidate])), candidates):
            candidate_opinions.grad = None
            expected_votes[candidate_index].backward(retain_graph=True)
            gain = candidate_opinions.grad
            candidate.opinion += self.learning_rate * np.array(gain[candidate_index])


    def plurality_objective_fn(self, X, y):
        """
        Returns the sum of probabilities that for all voters for voting for each candidate.
        """

        # calculate euclidean distance
        D = X.unsqueeze(1) - y.unsqueeze(0)
        D_squared = D ** 2
        D_summed = torch.sum(D_squared, dim=2)
        D = torch.sqrt(D_summed)

        # apply "radius of support" function to each distance
        g = lambda z: (1 / (1 + torch.exp(self.gamma * (z - self.radius))))
        R = g(D)
        R.requires_grad_(True)

        # apply softmax
        s = lambda z: torch.exp(self.beta * z)
        S = s(R)
        S.requires_grad_(True)

        # calculate the expected votes for each candidate
        expected_votes = S / S.sum(dim=1).unsqueeze(1)
        expected_votes = expected_votes.sum(dim=0)
        return expected_votes
    

    def vote_probabilities(self, voter_opinions, candidate_opinions):
        """
        Returns probabilities of each voter voting for each candidate as a 2d tensor.
        """

        # calculate euclidean distance
        D = voter_opinions.unsqueeze(1) - candidate_opinions.unsqueeze(0)
        D_squared = D ** 2
        D_summed = torch.sum(D_squared, dim=2)
        D = torch.sqrt(D_summed)

        # apply "radius of support" function to each distance
        g = lambda z: (1 / (1 + torch.exp(self.gamma * (z - self.radius))))
        R = g(D)

        # apply softmax
        s = lambda z: torch.exp(self.beta * z)
        S = s(R)

        # normalize
        probs = S / S.sum(dim=1).unsqueeze(1)
        return probs


    def get_random_agent_set(self, num_voters, num_candidates):
        """
        Returns a random subset of agents to interact with as a dict.

        returns: {"AgentType": [Agent1, Agent2, ..., AgentN]}
        """
        candidates = self.agents[Candidate]
        voters = self.agents[Voter]

        if num_candidates == self.agents[Candidate]:
            interaction_candidates = candidates
        else:
            interaction_candidates = self.random.sample(candidates, num_candidates)
        
        if num_voters == self.agents[Voter]:
            interaction_voters = candidates
        else:
            interaction_voters = self.random.sample(voters, num_voters)

        interaction_agents = {
            'candidates': interaction_candidates,
            'voters': interaction_voters
        }
        return interaction_agents


    def sigmoid_offset(self, x, offset):
        """
        Return 'offset sigmoid' of x and offset.
        """
        c = -math.log((1 - offset) / offset)
        return 1 / (1 + math.exp(-(x + c)))
    

    def normalize_to_range(self, scores):
        """
        Helper function to normalize numbers to a range.
        Used for converting the scores in score voting to "place like positions" in election consequences.
        """
        min_score = min(scores.values())
        max_score = max(scores)
        normalized_scores = []
        for score in scores:
            normalized_value = ((score - min_score) / (max_score - min_score)) * (self.num_candidates_to_benefit - (-self.num_candidates_to_benefit)) + (-self.num_candidates_to_benefit)
            normalized_scores.append(normalized_value)
        return normalized_scores

