import itertools
import xtrack as xt
import xobjects as xo
import pandas as pd
import numpy as np

collider = xt.Multiline.from_json('collider_04_tuned_and_leveled_bb_on.json')

line = collider.lhcb1

tw = line.twiss()

import json
with open('out.json', 'r') as fid:
    dct = json.load(fid)

all_particles_init = xt.Particles.from_dict(dct['all_particles_init'])
all_particles = xt.Particles.from_dict(dct['all_particles'])

all_particles_init.sort(interleave_lost_particles=True)
all_particles.sort(interleave_lost_particles=True)

norm = tw.get_normalized_coordinates(all_particles_init,
                                     nemitt_x=2.5e-6, nemitt_y=2.5e-6)

mask_lost = all_particles.state <= 0

n_repeat = 20
n_part_per_run = len(all_particles_init.x) // n_repeat

p_init_runs = []
p_runs = []
r_lost_min = []
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

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(norm.x_norm, norm.y_norm, '.')
plt.plot(norm.x_norm[mask_lost], norm.y_norm[mask_lost], 'xr')

plt.show()
