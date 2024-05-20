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
r_gen_half = np.linspace(6.5, 8, 100)[:30]
r_gen = np.zeros(2*len(r_gen_half))
r_gen[::2] = r_gen_half
r_gen[1::2] = r_gen_half + 1e-15

# r_gen = r_gen[17] + np.array([0, 1e-15, 1e-14, 1e-13, 1e-12])
# r_gen = r_gen[20] + np.array([0, 1e-15, 1e-14, 1e-13, 1e-12])
# r_gen = 6.75757575757575867925e+00 + np.array([0, 1e-15, 1e-14, 1e-13, 1e-12])

x_norm_gen = r_gen * np.cos(theta)
y_norm_gen = r_gen * np.sin(theta)

p_test = line.build_particles(x_norm=x_norm_gen,
                                y_norm=y_norm_gen,
                                delta=delta_max,
                                nemitt_x=2.5e-6, nemitt_y=2.5e-6)
p_test0 = p_test.copy()

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

line.track(p_test, num_turns=100000, turn_by_turn_monitor=True, with_progress=True)

p_test.sort(interleave_lost_particles=True)

mon_test = line.record_last_track
norm_test = tw.get_normalized_coordinates(mon_test.data, nemitt_x=2.5e-6, nemitt_y=2.5e-6)
rx_test = np.sqrt(norm_test.x_norm**2 + norm_test.px_norm**2)
ry_test = np.sqrt(norm_test.y_norm**2 + norm_test.py_norm**2)
r_test = np.sqrt(rx_test**2 + ry_test**2)
r_test = r_test.reshape(mon_test.x.shape).T

# Find the first particle where the small change has stabilized
i_stabilized = 2 * np.where(p_test.state[::2] != p_test.state[1::2])[0][0]
r_ref = r_test[:, i_stabilized]
r_shift_particle = r_test[:, i_stabilized + 1]

for nn in line.get_table().name[:-1]:
    if type(line[nn]) is xt.Drift:
        line[nn].length *= (1 - 1e-14)

p_test2 = p_test0.filter((p_test0.particle_id == i_stabilized)
                       | (p_test0.particle_id == i_stabilized + 1))

line.track(p_test2, num_turns=100000, turn_by_turn_monitor=True, with_progress=True)
mon_test2 = line.record_last_track
norm_test2 = tw.get_normalized_coordinates(mon_test2.data, nemitt_x=2.5e-6, nemitt_y=2.5e-6)
rx_test2 = np.sqrt(norm_test2.x_norm**2 + norm_test2.px_norm**2)
ry_test2 = np.sqrt(norm_test2.y_norm**2 + norm_test2.py_norm**2)
r_test2 = np.sqrt(rx_test2**2 + ry_test2**2)

r_test2 = r_test2.reshape(mon_test2.x.shape).T

r_change_circumference = r_test2[:, 0]

plt.figure(1)

plt.close('all')
plt.figure(1)
plt.plot(r_ref, label='simulation')
plt.plot(r_shift_particle, label=r'increase particle amplitude by $10^{-15}$ $\sigma$')
plt.plot(r_change_circumference, label=r'scale ring circumference by (1 + $10^{-14}$)')
# plt.plot(np.abs(r_change_circumference - r_ref), label='scale ring circumference by 1.00000000000001')

plt.legend()

plt.show()

