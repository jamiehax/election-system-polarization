import mesa
import numpy as np
import math
import torch


class Voter(mesa.Agent):
    """
    An agent representing the behavior of a voter.
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
        Move the voter according to their opinion update functions.
        """
        if self.unique_id in self.model.voters_to_activate:
            interaction_agents = self.get_interaction_agents(num_voters=1, num_candidates=1)

            # Voter-Voter interaction
            if self.voter_interaction_fn:
                for voter in interaction_agents['voters']:
                    self.voter_interaction_fn(self, voter)     

            # Voter-Candidate interaction
            if self.candidate_interaction_fn:
                for candidate in interaction_agents['candidates']:
                    self.candidate_interaction_fn(self, candidate, threshold=candidate.threshold)
          


class Candidate(mesa.Agent):
    """
    An agent representing the behavior of a cadndiate.
    """

    def __init__(self, uid, model, opinion, threshold, exit_probability):
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
        self.get_interaction_agents = self.model.get_random_agent_set

    
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
        num_voters_to_activate = kwargs.get('num_voters_to_activate', 1)
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
        gamma = kwargs.get('gamma', 0.2)
        radius = kwargs.get('radius', 1)
        learning_rate = kwargs.get('learning_rate', 0.1)
        second_choice_weight_factor = kwargs.get('second_choice_weight_factor', 0.1)
        voter_noise_factor = kwargs.get('voter_noise_factor', 0)
        exposure = kwargs.get('exposure', 0.5)
        device = kwargs.get('device', 'cpu')
        num_steps_before_ascent = kwargs.get('num_steps_before_ascent', 20)

        # GENERAL MODEL ATTRIBUTES
        self.num_agents = num_voters + initial_num_candidates
        self.num_voters = num_voters
        self.num_voters_to_activate = num_voters_to_activate
        self.voter_noise_factor = voter_noise_factor
        self.initial_num_candidates = initial_num_candidates
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.term_limit = term_limit
        self.num_opinions = num_opinions
        self.num_rounds_before_election = num_rounds_before_election
        self.winner = None
        self.voters_to_activate = []
        self.candidates_to_activate = []
        self.agent_set = {}
        self.schedule = mesa.time.StagedActivation(
            self,
            stage_list=['voters_move'],
            shuffle_between_stages=True
        )
        self.device = device
        self.num_steps_before_ascent = num_steps_before_ascent
        self.variance = 0

        # for adding candidates in later on
        self.candidate_index = num_voters
        self.initial_exit_probability = initial_exit_probability
        self.exit_probability_decrease_factor = exit_probability_decrease_factor

        # bounded confidence parameters
        self.initial_threshold = initial_threshold
        self.threshold_increase_factor = threshold_increase_factor
        self.num_candidates_to_benefit = num_candidates_to_benefit
        self.mu = mu

        # attraction-repulsion parameters
        self.exposure = exposure

        # candidate ascent parameters
        self.beta = beta
        self.radius = radius
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.second_choice_weight_factor = second_choice_weight_factor

        # agent interaction functions
        self.interaction_functions = {
            'avg': self.average_opinion,
            'bc': self.bounded_confidence,
            'bc1': self.bounded_confidence_one_sided,
            'kmean': self.k_mean,
            'ar': self.attraction_repulsion
        }

        # election systems
        election_systems = {
            'plurality': self.plurality_count,
            'rc': self.ranked_choice,
            'score': self.score_voting
        }
        self.election_system = election_systems[election_system]

        # candidate gradient ascent objective functions
        objective_functions = {
            'plurality': self.plurality_objective_fn,
            'rc': self.rc_objective_fn
        }
        self.candidate_objective_fn = objective_functions[election_system]

        # initialize voters
        self.agent_set[Voter] = []
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
            self.agent_set[Voter].append(voter)

        # initialize candidates
        self.agent_set[Candidate] = []
        for i in range(self.num_voters, self.num_agents):
            # get initial candidate opinion and location
            opinion = np.array([self.random.uniform(0, 1) for _ in range(self.num_opinions)])

            # initialize candidate
            candidate = Candidate(
                i,
                self,
                opinion,
                self.initial_threshold,
                initial_exit_probability
            )
            self.schedule.add(candidate)
            self.agent_set[Candidate].append(candidate)
            self.candidate_index += 1


        # data collector
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters = {
                'winner': 'winner.unique_id',
                'variance': 'variance'
            },
            agent_reporters = self.get_agent_reporter_dict()
        )

        self.running = True
        self.datacollector.collect(self)


    def get_agent_reporter_dict(self):
        agent_reporters = {
            'voted_for': 'voted_for',
            'num_votes': lambda a: a.num_votes if a.type == 'candidate' else None,
            'is_winner': lambda a: a.is_winner if a.type == 'candidate' else False,
            'num_wins': lambda a: a.num_wins if a.type == 'candidate' else None,
            'threshold': 'threshold',
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
        self.voters_to_activate = self.random.sample([a.unique_id for a in self.agent_set[Voter]], self.num_voters_to_activate)

        # move voters
        self.schedule.step()

        # candidates move strategically on specified rounds
        if self.schedule.steps % self.num_steps_before_ascent == 0:
            self.candidate_ascent()

        # election round processes
        if self.schedule.steps % self.num_rounds_before_election == 0:

            # clear old winner
            if self.winner: next((candidate for candidate in self.agent_set[Candidate] if candidate is self.winner), None).is_winner = False
            self.winner = None

            # remove term limit candidates:
            self.remove_term_limit_candidates()

            # find new winner
            winner = self.election_system()
            winner.is_winner = True
            winner.num_wins += 1
            self.winner = winner

            # add and remove candidates
            self.change_candidates(winner)

        # compute variance for round
        all_agents = self.agent_set[Voter] + self.agent_set[Candidate]
        all_opinions = [agent.opinion[0] for agent in all_agents]
        variance = np.array(all_opinions).var()
        self.variance = variance

        # collect data for round
        self.datacollector.collect(self)


    def remove_term_limit_candidates(self):
        """
        Remove all candidates who have hit their term limit
        """
        for candidate in self.agent_set[Candidate]:
            if candidate.num_wins >= self.term_limit:
                self.agent_set[Candidate].remove(candidate)
                self.schedule.remove(candidate)


    def change_candidates(self, winner):
        """
        Stochastically remove candidates based on their exit probabilities, winner candidate is excluded.
        Stochastically add candidates into the race, satisfying min and max candidate requirements.
        """
        candidates = self.agent_set[Candidate].copy()
        self.random.shuffle(candidates)

        # randomly remove candidates based on their exit probability, but not fewer than min num of candidates
        for candidate in candidates:
            if candidate is not winner:
                if len(candidates) >= self.min_candidates:
                    if self.random.random() < candidate.exit_probability:
                        candidates.remove(candidate)
                        self.agent_set[Candidate].remove(candidate)
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
                self.initial_exit_probability
            )
            self.schedule.add(candidate)
            self.agent_set[Candidate].append(candidate)
            self.candidate_index += 1


    def plurality_count(self):
        """
        Returns the winner of the election for plurality count, and resets all candidate vote counts.
        """
        candidates = self.agent_set[Candidate]
        voters = self.agent_set[Voter]

        # tensor of each voters probaility of voting for each candidate
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in voters]).to(self.device)
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in candidates]).to(self.device)

        # sample voter probabilities
        voter_probabilities = self.vote_probabilities(voter_opinions, candidate_opinions)
        ballots = []
        for voter, prob in zip(voters, voter_probabilities):
            # prob[torch.isnan(prob)] = 0 # replace NaN values with 0
            vote = np.random.choice(candidates, size=1, p=prob.cpu())[0]
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

        # create copy lists to remove candidates from during election rounds
        candidates = self.agent_set[Candidate].copy()
        voters = self.agent_set[Voter].copy()

        # tensor of each voters probaility of voting for each candidate
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in voters]).to(self.device)
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in candidates]).to(self.device)

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
        candidates = self.agent_set[Candidate]
        voters = self.agent_set[Voter]

        # tensor of each voters probaility of voting for each candidate
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in voters]).to(self.device)
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in candidates]).to(self.device)

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
            normalized_score = len(self.agent_set[Candidate]) - ((score - min(scores.values())) / (max(scores.values()) - min(scores.values())) * len(self.agent_set[Candidate]))
            candidate.exit_probability = self.sigmoid_offset(self.exit_probability_decrease_factor * (normalized_score - self.num_candidates_to_benefit), candidate.exit_probability)
            candidate.threshold = self.sigmoid_offset(self.threshold_increase_factor * (self.num_candidates_to_benefit - normalized_score), candidate.threshold)


        winner = max(scores, key=scores.get)
        return winner


    def average_opinion(self, a1, a2, threshold=None):
        """
        Update agent a1 and s2's opinions to be the average of their opinions.
        """
        average = ((a1.opinion + a2.opinion) / 2) + (self.voter_noise_factor * np.random.normal(0, 1, size=self.num_opinions))
        a1.opinion = self.bounded_update(average)
        a2.opinion = self.bounded_update(average)


    def bounded_confidence(self, a1, a2, threshold=None):
        """
        Update agent a1 and a2's opinions according to the bounded confidence model.
        """
        if not threshold:
            threshold = self.initial_threshold

        if np.linalg.norm(a1.opinion - a2.opinion) < threshold:
            a1.opinion = self.bounded_update((a1.opinion + (self.mu * (a2.opinion - a1.opinion))) + (self.voter_noise_factor * np.random.normal(0, 1, size=self.num_opinions)))
            a2.opinion = self.bounded_update((a2.opinion + (self.mu * (a1.opinion - a2.opinion))) + (self.voter_noise_factor * np.random.normal(0, 1, size=self.num_opinions)))


    def bounded_confidence_one_sided(self, a1, a2, threshold=None):
        """
        Update agent a1's according to a one sided bounded confidence opinion update.
        """
        if not threshold:
            threshold = self.initial_threshold

        if np.linalg.norm(a1.opinion - a2.opinion) < threshold:
            a1.opinion = self.bounded_update((a1.opinion + (self.mu * (a2.opinion - a1.opinion))) + (self.voter_noise_factor * np.random.normal(0, 1, size=self.num_opinions)))

    
    def attraction_repulsion(self, a1, a2, threshold=None):
        """
        Update agent a1 and a2's opinions according to the attraction-repulsion model.
        """
        if not threshold:
            threshold = self.initial_threshold

        d = np.linalg.norm(a1.opinion - a2.opinion)
        interaction_probability = (1/2)**(d / self.exposure)
        if self.random.random() < interaction_probability:
            if d <= threshold:
                a1.opinion = self.bounded_update((a1.opinion + ((a2.opinion - a1.opinion) * self.mu)) + (self.voter_noise_factor * np.random.normal(0, 1, size=self.num_opinions)))
            else:
                a1.opinion = self.bounded_update((a1.opinion - ((a2.opinion - a1.opinion) * self.mu)) + (self.voter_noise_factor * np.random.normal(0, 1, size=self.num_opinions)))



    def k_mean(self, **kwargs):
        """
        Update a1 (candidate) opinion to be the mean of their voting bloc
        """
        candidate = kwargs.get('a1', None)
        if isinstance(candidate, Candidate):
            voters = self.agent_set[Voter]
            voting_bloc = [voter for voter in voters if voter.voted_for == candidate.unique_id]
            if len(voting_bloc) > 0:
                opinion_sum = sum([voter.opinion for voter in voting_bloc])
                candidate.opinion =  self.bounded_update(np.array(opinion_sum / len(voting_bloc)))
            else:
                return
        else:
            raise TypeError('Invalid parameter type. The k_mean opinion update function is only implemented for Candidates.')


    def candidate_ascent(self):
        """
        Move the candidate in the direction of the gradient calculated from the candidates expected number of votes.
        """

        # get tensors of voters and candidate opinions
        voter_opinions = torch.stack([torch.tensor(v.opinion) for v in self.agent_set[Voter]]).to(self.device)
        voter_opinions.requires_grad_(True)
        candidate_opinions = torch.stack([torch.tensor(c.opinion) for c in self.agent_set[Candidate]]).to(self.device)
        candidates = [c for c in self.agent_set[Candidate]] # list of candidates in same order as gradients
        candidate_opinions.requires_grad_(True)
        
        expected_votes = self.candidate_objective_fn(voter_opinions, candidate_opinions)

        # update candidates positions with gradients
        for candidate_index, candidate in zip(range(len(self.agent_set[Candidate])), candidates):
            expected_votes[candidate_index].backward(retain_graph=True)
            gains = candidate_opinions.grad.cpu()
            gain = np.array(gains[candidate_index])

            # check if gradient is NaN
            if not np.isnan(gain).any():
                candidate.opinion = self.bounded_update(candidate.opinion + (self.learning_rate * gain))
            
            # zero out gradient for next candidate
            candidate_opinions.grad.zero_()


    def plurality_objective_fn(self, X, y):
        """
        Returns the expected votes of each candidate based on first choice probabilities.
        """
        vote_probabilities = self.vote_probabilities(X, y)
        expected_votes = vote_probabilities.sum(dim=0)
        return expected_votes
    

    def rc_objective_fn(self, X, y):
        """
        Returns the expected votes of each candidate based on first and second choice probabilities.
        """

        # calculate the first choice expected votes for each candidate
        first_choice_probs = self.vote_probabilities(X, y)
        first_expected_votes = first_choice_probs.sum(dim=0)

        # calculate the second choice expected votes for each candidate
        second_choice_probs = torch.zeros_like(first_choice_probs)
        for j in range(len(y)):

            # vote probabilities when candidate j is not in race
            if j == y.shape[0] - 1: # edge case of last row not slicing correctly
                y_no_j = y[:j] 
            else:
                y_no_j = torch.cat((y[:j], y[j+1:]), dim=0)
            second_probs = self.vote_probabilities(X, y_no_j)

            # insert column of zero probs at candidate j index
            zeros_column = torch.zeros(second_probs.shape[0], 1)
            second_probs = torch.cat((second_probs[:, :j], zeros_column, second_probs[:, j:]), dim=1)

            # multiply first choice probs by second choice probs
            second_probs = second_probs * first_choice_probs

            # add second choice probs with respect to candidate j to probs sum
            second_choice_probs = second_choice_probs + second_probs


        second_expected_votes = second_choice_probs.sum(dim=0)

        expected_votes = first_expected_votes + (self.second_choice_weight_factor * second_expected_votes)
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
        candidates = self.agent_set[Candidate]
        voters = self.agent_set[Voter]

        if num_candidates == self.agent_set[Candidate]:
            interaction_candidates = candidates
        else:
            interaction_candidates = self.random.sample(candidates, num_candidates)
        
        if num_voters == self.agent_set[Voter]:
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
    

    def bounded_update(self, update):
        """
        Return the the opinion update subject to the opinion space bounds.
        """
        max_update = np.minimum(update, np.ones(self.num_opinions))
        min_update = np.maximum(max_update, np.zeros(self.num_opinions))
        return min_update

