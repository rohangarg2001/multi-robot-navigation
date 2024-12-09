from environment_discrete import DiscreteEnvironment
from collections import defaultdict
import random
import math 
from typing import Any, Callable
import numpy as np

class MCTS_DPW():
    def __init__(self, 
                env: DiscreteEnvironment,
                N: dict[tuple[Any, Any], int],
                Q: dict[tuple[Any, Any], float],
                d: int,
                m: int,
                c: float,
                U: Callable[[Any], float],
                state_action_pairs = defaultdict[tuple[Any, Any], list[Any]],      # the key to the dict is the 
                # state-action pair and it outputs the states that got visited after transitioning from state and action. 
                available_actions = defaultdict[Any, list[Any]],
                # availiable_actions contains the actions that were sampled from state
                # IMP : DOESN'T contain actions as tuples. It's a list 
                k_a : float = 40,               # k_a : action
                k_o : float = 20,               # k_o : observation
                alpha : float = 0.5,                        # for action widening
                beta : float = 0.5,
                ):                        # for state widening
        self.env = env
        self.N = N
        self.Q = Q          
        self.d = d          # depth
        self.m = m          # no of simulations
        self.c = c
        self.U = U
        self.A = env.all_action_combinations
        self.alpha = alpha
        self.beta = beta
        self.state_action_pairs = state_action_pairs
        self.available_actions = available_actions
        self.k_a = k_a
        self.k_o = k_o
    
    def __call__(self, s:Any) -> Any:
        for _ in range(self.m):
            self.simulate(s, self.d)
        tuple_s = tuple(np.concatenate(s))
        actions = self.available_actions[tuple_s]                 # list of available actions     
        ## so now we select an action from the actions
        index_action = np.argmax([self.Q[(tuple_s, tuple(a))] for a in actions])
        return actions[index_action]          
    
    def simulate(self, s: Any, d:int) -> Any:
        if d <= 0:
            return self.U(s)
        tuple_s = tuple(np.concatenate(s))
        tuple_a = tuple(self.A[0])                          
        if (tuple_s, tuple_a) not in self.N:
            for a in self.A:
                self.Q[(tuple_s, tuple(a))] = 0.0
                self.N[(tuple_s, tuple(a))] = 0
                self.available_actions[tuple_s] = list()
                # self.state_action_pairs[(tuple_s, tuple(a))] = list()               ## contains the states visited after (s,a)
            return self.U(s)
        self.env.set_positon(s)       
        a = self.explore(s)
        # ## state widening
        # number_of_states_visited_from_s_a = len(self.state_action_pairs[(tuple_s, tuple(a))])                 
        # max_states_from_s_a = int(np.ceil(self.k_o * (self.N[(tuple_s, tuple(a))] ** self.beta))) + 1

        # if(number_of_states_visited_from_s_a < max_states_from_s_a):
        #     ## it means we can sample a new state
        #     s_prime = self.env.step(a)
        #     self.state_action_pairs[(tuple_s, tuple(a))].append(s_prime)
        # else:
        #     ## we need to sample from existing next states arrays.
        s_prime = self.env.step(a)
        r = self.env.calculate_reward()
        q = r + self.env.gamma * self.simulate(s_prime, d-1)
        self.N[(tuple_s, tuple(a))] += 1
        self.Q[(tuple_s, tuple(a))] += (q - self.Q[(tuple_s,tuple(a))]) / self.N[(tuple_s,tuple(a))]
        return q
    
    def explore(self, s: Any) -> Any:
        ## Adding action widening here
        A, N = self.A, self.N
        tuple_s = tuple(np.concatenate(s))
        N_s = np.sum([N[(tuple_s,tuple(a))] for a in A])

        number_actions = len(self.available_actions[tuple_s])           ## no of childen of node s
        max_actions = int(np.ceil(self.k_a * (N_s ** self.alpha))) + 1                  ## ISN'T ADDING ONE NEEDED HERE

        if(number_actions < max_actions):
            unused_actions = [a for a in self.A if not any(np.array_equal(a, used_a) for used_a in self.available_actions[tuple_s])]            # discrete case, otherwise in continous we can sample a new action easily
            if (len(unused_actions) != 0):
                new_action = unused_actions[random.randint(0, len(unused_actions)-1)]
                self.available_actions[tuple_s].append(new_action)
        index_action = np.argmax([self.ucb1(s,a,N_s) for a in self.available_actions[tuple_s]])
        # print(index_action, len(self.available_actions[tuple_s]))
        return self.available_actions[tuple_s][index_action]

    def ucb1(self, s:Any, a:Any, Ns: int) -> float:
        N,Q,c = self.N, self.Q, self.c
        tuple_s = tuple(np.concatenate(s))
        return Q[(tuple_s,tuple(a))] + c * self.exploration(N[(tuple_s,tuple(a))], Ns)

    def exploration(self, Nsa: int, Ns: int) -> float:
        if(Nsa == 0):
            return np.inf
        else:
            return np.sqrt(np.log(Ns/Nsa))