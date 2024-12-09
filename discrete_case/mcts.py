from environment_discrete import DiscreteEnvironment
from collections import defaultdict
import random
import math 
from typing import Any, Callable
import numpy as np

class MCTS():
    def __init__(self, 
                 env: DiscreteEnvironment,
                 N: dict[tuple[Any, Any], int],         # N matrix
                 Q: dict[tuple[Any, Any], float],       # Q matrix
                 d: int,
                 m: int,
                 c: float,
                 U: Callable[[Any], float]):
        self.env = env
        self.N = N
        self.Q = Q          
        self.d = d          # depth
        self.m = m          # no of simulations
        self.c = c
        self.U = U
        self.A = env.all_action_combinations

    def __call__(self, s:Any) -> Any:
        for _ in range(self.m):
            self.simulate(s, d=self.d)
        tuple_s = tuple(np.concatenate(s))
        index_action = np.argmax([self.Q[(tuple_s, tuple(a))] for a in self.env.all_action_combinations])
        return self.env.all_action_combinations[index_action]

    def simulate(self, s:Any, d:int):
        if (d<=0):
            return self.U(s)
            ## can try using a random roll out policy here
        tuple_s = tuple(np.concatenate(s))
        tuple_a = tuple(self.A[0])
        if (tuple_s, tuple_a) not in self.N:
            for a in self.A:
                a_tuple = tuple(a.tolist())
                self.Q[(tuple_s, a_tuple)] = 0
                self.N[(tuple_s, a_tuple)] = 0.0
            return self.U(s)
        self.env.set_positon(s)
        a = self.explore(s)
        s_prime = self.env.step(a)
        r = self.env.calculate_reward()
        q = r + self.env.gamma * self.simulate(s_prime, d-1)
        self.N[(tuple_s,tuple(a))] += 1
        self.Q[(tuple_s,tuple(a))] += (q - self.Q[(tuple_s,tuple(a))]) / self.N[(tuple_s,tuple(a))]
        return q

    def explore(self, s:Any) -> Any:
        A, N = self.A, self.N
        tuple_s = tuple(np.concatenate(s))
        N_s = np.sum([N[(tuple_s,tuple(a))] for a in A])
        index_action = np.argmax([self.ucb1(s,a,N_s) for a in A])
        return A[index_action]
    
    def ucb1(self, s:Any, a:Any, Ns: int) -> float:
        N,Q,c = self.N, self.Q, self.c
        tuple_s = tuple(np.concatenate(s))
        return Q[(tuple_s,tuple(a))] + c * self.exploration(N[(tuple_s,tuple(a))], Ns)

    def exploration(self, Nsa: int, Ns: int) -> float:
        if(Nsa == 0):
            return np.inf
        else:
            return np.sqrt(np.log(Ns/Nsa))