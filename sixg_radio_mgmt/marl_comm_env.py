from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from typing import Callable, Optional, Tuple, Type, Union

from .comm_env import CommunicationEnv

import numpy as np

from .association import Association
from .basestations import Basestations
from .channel import Channel
from .metrics import Metrics
from .mobility import Mobility
from .slices import Slices
from .traffic import Traffic
from .ues import UEs


class MARLCommEnv(AECEnv, CommunicationEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        CommunicationEnv.__init__(self, *args, **kwargs)
