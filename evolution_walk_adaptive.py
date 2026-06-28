"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ESM2 evolution walk

Starts from random sequences, proposes mutations, 
Scores them with ESM2-based PRCS plus epistasis, 
Adaptively makes mutation steps smaller as PRCS improves

~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import torch
import torch.nn.functional as F
import esm

import os
import warnings
import sys
from pathlib import Path
import random
import math
import time
import pickle
from typing import List, Tuple
from itertools import combinations