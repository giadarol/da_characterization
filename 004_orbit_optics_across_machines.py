import itertools
import xtrack as xt
import xobjects as xo
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

delta_max = 27.e-5

collider = xt.Multiline.from_json('collider_04_tuned_and_leveled_bb_on.json')

line = collider.lhcb1

tw = line.twiss()

# M2
# x_co = -8.989189359942636e-07