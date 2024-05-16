import itertools
import xtrack as xt
import xobjects as xo
import pandas as pd
import numpy as np

collider = xt.Multiline.from_json('collider_04_tuned_and_leveled_bb_on.json')

collider.vars['beambeam_scale'] = 0.8

line = collider.lhcb1

tw = line.twiss()

shift_r_vector = np.linspace(-0.01, 0.01, 20)[:10]

r_min = 2
r_max = 10
n_r = 256
n_angles = 5
delta_max = 27.e-5
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

particles_objects = []

for shift in shift_r_vector:
    radial_list = np.linspace(r_min + shift, r_max + shift, n_r,
                               endpoint=False)
    theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]

    # Define particle distribution as a cartesian product of the above
    particle_list = [(particle_id, ii[1], ii[0]) for particle_id, ii
             in enumerate(itertools.product(theta_list, radial_list))]

    particle_df = pd.DataFrame(particle_list,
                columns=["particle_id", "normalized amplitude in xy-plane",
                         "angle in xy-plane [deg]"])

    r_vect = particle_df["normalized amplitude in xy-plane"].values
    theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # type: ignore # [rad]

    A1_in_sigma = r_vect * np.cos(theta_vect)
    A2_in_sigma = r_vect * np.sin(theta_vect)

    particles = line.build_particles(
            x_norm=A1_in_sigma,
            y_norm=A2_in_sigma,
            delta=delta_max,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
    )
    particles_objects.append(particles)

all_particles = xt.Particles.merge(particles_objects)
all_particles_init = all_particles.copy(_context=xo.context_default)

context = xo.ContextCupy()
line.discard_tracker()
line.build_tracker(_context=context)
all_particles.move(_context=context)

num_turns = 1000000
line.track(all_particles, num_turns=num_turns, with_progress=10)

all_particles.move(_context=xo.context_default)
all_particles.sort(interleave_lost_particles=True)


dct_out = {
    'all_particles_init': all_particles_init.to_dict(),
    'all_particles': all_particles.to_dict()}

import json
with open('out_1M_less_bb.json', 'w') as fid:
    json.dump(dct_out, fid, cls=xo.JEncoder)
