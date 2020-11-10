"""
Registry of custom implemented algorithms names
Please refer to the following examples to add your custom algorithms : 
- AlphaZero : https://github.com/ray-project/ray/tree/master/rllib/contrib/alpha_zero
- bandits : https://github.com/ray-project/ray/tree/master/rllib/contrib/bandits
- maddpg : https://github.com/ray-project/ray/tree/master/rllib/contrib/maddpg
- random_agent: https://github.com/ray-project/ray/tree/master/rllib/contrib/random_agent
An example integration of the random agent is shown here : 
- https://github.com/AIcrowd/neurips2020-procgen-starter-kit/tree/master/algorithms/custom_random_agent
"""

def _import_ucb_drac_agent():
    from .ucb_drac_agent.trainer import UcbDracTrainer
    return UcbDracTrainer


CUSTOM_ALGORITHMS = {
    "UcbDracAgent": _import_ucb_drac_agent
}
