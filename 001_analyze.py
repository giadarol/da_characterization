import itertools
import xtrack as xt
import xobjects as xo
import pandas as pd
import numpy as np

n_repeat = 100
fname_data = 'out_100k.json'
bb_scale = 1.0

n_repeat = 20
fname_data = 'out_1M.json'
bb_scale = 1.0

n_repeat = 16
fname_data = 'out_1M_less_bb.json'
bb_scale = 0.8

n_repeat = 16
fname_data = 'out_1M_bb_0.9.json'
bb_scale = 0.9

collider = xt.Multiline.from_json('collider_04_tuned_and_leveled_bb_on.json')
collider.vars['beambeam_scale'] = bb_scale

line = collider.lhcb1

tw = line.twiss()

import json
with open(fname_data, 'r') as fid:
    dct = json.load(fid)

all_particles_init = xt.Particles.from_dict(dct['all_particles_init'])
all_particles = xt.Particles.from_dict(dct['all_particles'])

all_particles_init.sort(interleave_lost_particles=True)
all_particles.sort(interleave_lost_particles=True)

norm = tw.get_normalized_coordinates(all_particles_init,
                                     nemitt_x=2.5e-6, nemitt_y=2.5e-6)

mask_lost = all_particles.state <= 0


n_part_per_run = len(all_particles_init.x) // n_repeat

p_init_runs = []
p_runs = []
r_lost_min = []
norm_runs = []
r_runs = []
at_turn_runs = []
for ii in range(n_repeat):
    p_init_run = all_particles_init.filter(
        (all_particles_init.particle_id >= ii*n_part_per_run)
        & (all_particles_init.particle_id < (ii+1)*n_part_per_run))
    p_run = all_particles.filter(
        (all_particles.particle_id >= ii*n_part_per_run)
        & (all_particles.particle_id < (ii+1)*n_part_per_run))
    p_init_run.sort(interleave_lost_particles=True)
    p_run.sort(interleave_lost_particles=True)
    mask_lost_run = p_run.state <= 0
    norm_run = tw.get_normalized_coordinates(p_init_run,
                                            nemitt_x=2.5e-6, nemitt_y=2.5e-6)
    r_lost_min.append(
        np.min(np.sqrt(norm_run.x_norm**2 + norm_run.y_norm**2)[mask_lost_run]))
    p_init_runs.append(p_init_run)
    p_runs.append(p_run)
    norm_runs.append(norm_run)

    r_run = np.sqrt(norm_run.x_norm**2 + norm_run.y_norm**2)
    r_runs.append(r_run)

    at_turn_runs.append(p_run.at_turn)

# identify particle determining da
r_all = np.sqrt(norm.x_norm**2 + norm.y_norm**2)
at_turn_all = all_particles.at_turn

r_all_lost = r_all.copy()
r_all_lost[~mask_lost] = np.nan
idx_da = np.nanargmin(r_all_lost)

p_da_end = all_particles.filter(all_particles.particle_id == idx_da)
p_da = all_particles_init.filter(all_particles.particle_id == idx_da)
p_da_init = p_da.copy()

line.track(p_da, num_turns=1000, turn_by_turn_monitor=True, with_progress=True)
mon_da = line.record_last_track

norm_da = tw.get_normalized_coordinates(mon_da.data, nemitt_x=2.5e-6, nemitt_y=2.5e-6)

# theta = np.deg2rad(30)
# r_test = 8
# x_norm_test = r_test * np.cos(theta)
# y_norm_test = r_test * np.sin(theta)

# p_test = line.build_particles(x_norm=x_norm_test*(1 + np.array([0, 1e-14, 1e-12, 1e-10])),
#                                 y_norm=y_norm_test,
#                                 delta=p_da.delta[0],
#                                 nemitt_x=2.5e-6, nemitt_y=2.5e-6)


# line.track(p_test, num_turns=100000, turn_by_turn_monitor=True, with_progress=True)
# mon_test = line.record_last_track
# norm_test = tw.get_normalized_coordinates(mon_test.data, nemitt_x=2.5e-6, nemitt_y=2.5e-6)
# rx_test = np.sqrt(norm_test.x_norm**2 + norm_test.px_norm**2)
# ry_test = np.sqrt(norm_test.y_norm**2 + norm_test.py_norm**2)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(norm.x_norm, norm.y_norm, '.')
plt.plot(norm.x_norm[mask_lost], norm.y_norm[mask_lost], 'xr')

r_all = np.sqrt(norm.x_norm**2 + norm.y_norm**2)
at_turn_all = all_particles.at_turn

plt.figure(2)
i_run = 7
# plt.plot(r_all, at_turn_all, '.')
plt.plot(r_runs[i_run], at_turn_runs[i_run], '.')

plt.figure(3)
plt.plot(norm_da.x_norm, norm_da.px_norm, '.')
plt.xlabel('x norm')
plt.ylabel('px norm')

plt.figure(4)
plt.plot(norm_da.y_norm, norm_da.py_norm, '.')
plt.xlabel('y norm')
plt.ylabel('py norm')

rx_norm = np.sqrt(norm_da.x_norm**2 + norm_da.px_norm**2)
ry_norm = np.sqrt(norm_da.y_norm**2 + norm_da.py_norm**2)
plt.figure(5)
plt.plot(rx_norm, label='rx')
plt.plot(ry_norm, label='ry')
plt.plot(np.sqrt(rx_norm**2 + ry_norm**2), label='r')
plt.legend()
plt.xlabel('turn')


plt.show()
