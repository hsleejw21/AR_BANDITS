from autoregressive_bandits.algorithms.base import Agent, ObservationContext
from autoregressive_bandits.algorithms.ar2 import AR2Agent
from autoregressive_bandits.algorithms.baselines import EpsilonGreedyAgent, RandomAgent, UCB1Agent
from autoregressive_bandits.algorithms.var_oracle_ar2 import VAROracleAR2Agent

AGENT_REGISTRY = {
    "ar2": AR2Agent,
    "epsilon_greedy": EpsilonGreedyAgent,
    "random": RandomAgent,
    "ucb1": UCB1Agent,
    "var_oracle_ar2": VAROracleAR2Agent,
    "var_estimated_ar2": VAROracleAR2Agent,
}

__all__ = [
    "AGENT_REGISTRY",
    "AR2Agent",
    "Agent",
    "EpsilonGreedyAgent",
    "ObservationContext",
    "RandomAgent",
    "UCB1Agent",
    "VAROracleAR2Agent",
]
