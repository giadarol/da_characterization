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

# x_co
# Apple -8.989189359942636e-07
# AMD   -8.989189346524893e-07

# betx at IP3
# Apple 110.84097649513589
# AMD   110.84097648965272

p = line.build_particles(x_norm=1, nemitt_x=2.5e-6, nemitt_y=2.5e-6)
# matched x
# Apple 0.0001918249725735483
# AMD   0.00019182497257012318

# Example 6x6 matrix
MM = np. array([[80,  7, 29, 17, 93,  4],
                [51, 22, 85, 66, 34, 32],
                [91, 50,  0, 18, 60, 69],
                [29, 49, 60, 48, 45, 31],
                [75, 38, 32, 81, 89,  5],
                [14, 83, 18, 42, 82, 62]],
                dtype=float)

b = np.array([[13, 55, 91, 75, 86, 25, 34, 97, 29, 89],
              [81, 29, 84, 77, 16, 17,  9, 59, 86, 26],
              [24, 11,  1, 52, 34, 58, 35, 71, 39, 42],
              [79, 39, 41, 77, 23,  2, 49, 12, 16, 16],
              [ 0, 46, 19, 12, 11, 87, 57, 94, 29, 84],
              [70, 49, 15, 61, 86, 12, 64, 73,  1,  2]],
              dtype=float)

x = np.linalg.solve(MM, b)
x0 = np.linalg.solve(MM, b[:, 0])

# On Apple
# x0[4]
# 0.07444200479472769
# x[4, 0]
# 0.07444200479472773

# On AMD
# x0[4]
# 0.07444200479472772
# x[4, 0]
# 0.07444200479472766

WW = np.array([[ 1.05281041e+01,  1.28932090e-15, -6.17465349e-01,
        -3.12015770e-01,  2.10683856e-03, -8.67865572e-03],
       [-1.93499284e-01,  9.45658140e-02,  1.44902247e-02,
         1.87708058e-04, -4.67930491e-06, -4.61380648e-04],
       [ 8.77730042e-01, -4.39717523e-01,  1.49425638e+01,
         1.82993629e-15, -5.24509370e-03,  1.64937770e-03],
       [ 1.24433656e-02, -1.21837510e-03,  1.74389386e-01,
         6.66281202e-02, -4.73524295e-05,  3.35291675e-05],
       [ 1.72184162e-01, -2.13139058e-02, -1.64554593e-02,
        -9.38755362e-04,  2.63766223e+01,  3.23020460e-15],
       [ 1.32258731e-05, -6.85232512e-06,  6.92167821e-06,
         1.32941276e-05, -1.92160965e-04,  3.79124021e-02]])

WW_inv = np.linalg.inv(WW)


