import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json('collider_04_tuned_and_leveled_bb_on.json')

line_name = 'lhcb1'
bb_scale = 0.

line = collider[line_name]
line.vars['beambeam_scale'] = bb_scale

tw = line.twiss()
tt = line.get_table()

det = line.get_amplitude_detuning_coefficients(
    a0_sigmas=0.1, a1_sigmas=0.2, a2_sigmas=0.3)

det_table = xt.Table({
    'name': np.array(list(det.keys())),
    'value': np.array([v for v in det.values()]),
})

nl_chrom = line.get_non_linear_chromaticity(delta0_range=(-2e-4, 2e-4), num_delta=5, fit_order=3)

out = ''

out += f'Line: {line_name}\n'
out += f'Beam beam scale: {bb_scale}\n'
out += '\n'

out += f'Installed element types:\n'
out += repr([nn for nn in sorted(list(set(tt.element_type))) if len(nn)>0]) + '\n'
out += '\n'

out += f'Tunes:        Qx  = {tw["qx"]:.5f}       Qy = {tw["qy"]:.5f}\n'
out += f"Chromaticity: Q'x = " + f'{tw["dqx"]:.2f}     ' + "Q'y = " + f'{tw["dqy"]:.2f}\n'
out += f'c_minus:      {tw["c_minus"]:.5e}\n'
out += '\n'

out += f'Synchrotron tune: {tw["qs"]:5e}\n'
out += f'Slip factor:      {tw["slip_factor"]:.5e}\n'
out += '\n'


out += f'Twiss parameters and phases at IPs:\n'
out += tw.rows['ip.*'].cols[
    'name s betx bety alfx alfy mux muy'].show(
        output=str, max_col_width=int(1e6), digits=8)
out += '\n\n'

out += f'Dispersion at IPs:\n'
out += tw.rows['ip.*'].cols[
    'name s dx dy dpx dpy'].show(
        output=str, max_col_width=int(1e6), digits=8)
out += '\n\n'

out += 'Crab dispersion at IPs:\n'
out += tw.rows['ip.*'].cols[
    'name s dx_zeta dy_zeta dpx_zeta dpy_zeta'].show(
        output=str, max_col_width=int(1e6), digits=8)
out += '\n\n'

out += 'Amplitude detuning coefficients:\n'
out += det_table.show(output=str, max_col_width=int(1e6), digits=6)
out += '\n\n'

out += 'Non-linear chromaticity:\n'
out += f'dnqx = {list(nl_chrom["dnqx"])}\n'
out += f'dnqy = {list(nl_chrom["dnqy"])}\n'
out += '\n\n'

out += 'Tunes and momentum compaction vs delta:\n'
out += nl_chrom.show(output=str, max_col_width=int(1e6), digits=6)
out += '\n\n'

