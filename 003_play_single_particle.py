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


theta = np.deg2rad(30)
r_test = 8
x_norm_test = r_test * np.cos(theta)
y_norm_test = r_test * np.sin(theta)

p_test = line.build_particles(x_norm=x_norm_test*(1 + np.array([0, 1e-12])),
                                y_norm=y_norm_test,
                                delta=delta_max,
                                nemitt_x=2.5e-6, nemitt_y=2.5e-6)
p_test0 = p_test.copy()


line.track(p_test, num_turns=100000, turn_by_turn_monitor=True, with_progress=True)
mon_test = line.record_last_track
norm_test = tw.get_normalized_coordinates(mon_test.data, nemitt_x=2.5e-6, nemitt_y=2.5e-6)
rx_test = np.sqrt(norm_test.x_norm**2 + norm_test.px_norm**2)
ry_test = np.sqrt(norm_test.y_norm**2 + norm_test.py_norm**2)
r_test = np.sqrt(rx_test**2 + ry_test**2)

r_test = r_test.reshape(mon_test.x.shape).T


for nn in line.get_table().name[:-1]:
    if type(line[nn]) is xt.Drift:
        line[nn].length *= (1 +1e-14)

p_test2 = p_test0.copy()

line.track(p_test2, num_turns=100000, turn_by_turn_monitor=True, with_progress=True)
mon_test2 = line.record_last_track
norm_test2 = tw.get_normalized_coordinates(mon_test2.data, nemitt_x=2.5e-6, nemitt_y=2.5e-6)
rx_test2 = np.sqrt(norm_test2.x_norm**2 + norm_test2.px_norm**2)
ry_test2 = np.sqrt(norm_test2.y_norm**2 + norm_test2.py_norm**2)
r_test2 = np.sqrt(rx_test2**2 + ry_test2**2)

r_test2 = r_test2.reshape(mon_test2.x.shape).T

r_ref = r_test[:, 0]
r_shift_particle = r_test[:, 1]
r_change_circumference = r_test2[:, 0]

plt.figure(1)

plt.close('all')
plt.figure(1)
plt.plot(r_ref, label='simulation')
plt.plot(r_shift_particle, label='scale particle amplitude by 1.000000000001')
plt.plot(r_change_circumference, label='scale ring circumference by 1.00000000000001')
plt.legend()
plt.figure(2)
plt.plot(np.abs(r_shift_particle - r_ref), label='scale particle amplitude by 1.000000000001')
plt.plot(np.abs(r_change_circumference - r_ref), label='scale ring circumference by 1.00000000000001')

plt.legend()

plt.show()

