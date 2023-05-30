from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from .comm_env import CommunicationEnv


class MARLCommEnv(AECEnv, CommunicationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
