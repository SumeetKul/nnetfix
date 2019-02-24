"""
NNETFIX
=====

NNetfix: A Neural Network to 'fix' Gravitational Wave signals affected by short-duration noise transients.

With improved Advanced LIGO and Virgo detector sensitivity, it is increasingly likely that astrophysical Gravitational Wave (GW) signals overlap with short-duration noise transients. NNetfix is a scikit-Learn Multi-Layered Perceptron(MLP)-based neural network that `fixes` GW signals coincident with short-duration glitches. NNETFIX operates by gating the glitch and identifying the features of the GW signal to reconstruct it in the portion of the data affected by the glitch, improving upon the signal-to-noise ratio and recovering the signal parameters effectively, in low latency.

The code is currently hosted at https://git.ligo.org/sumeet.kulkarni/nnetfix.

"""


from __future__ import absolute_import

from . import core, templatebank

from .core import utils, 
#from .core.sampler import run_sampler
#from .core.likelihood import Likelihood

__version__ = utils.get_version_information()

