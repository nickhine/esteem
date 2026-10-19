#!/usr/bin/env python
# coding: utf-8

"""Defines a task to use a Machine Learning calculator to generate
Molecular Dynamics trajectories, and also to process the results
to generate infrared spectra from trajectory data"""


# # Main Routine
# Load essential modules
import sys
import os
import string
from ase.io.trajectory import Trajectory
from esteem.trajectories import generate_md_trajectory, find_initial_geometry, get_trajectory_list, targstr
from esteem.trajectories import committee_stde_per_atom, md_output_names, finish_md_output


def driving_target(traj_target):
    """The target that drives the MD: the first, if several are evaluated at each snapshot (eg [1,0]).
    Output trajectories are named by target, so this one gives the trajectory's main output file"""
    return traj_target[0] if isinstance(traj_target, list) else traj_target


class MLTrajTask:

    """
    Defines a task to use a Machine Learning calculator to generate
    Molecular Dynamics trajectories

    The main routine is :meth:`esteem.tasks.ml_trajectories.run`
    """

    def __init__(self, **kwargs):
        self.wrapper = None
        self.snap_wrapper = None
        self.snap_calc_params = None
        self.snap_target = None
        self.which_targets = None
        self.script_settings = None
        self.task_command = 'mltraj'
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

    # Main routine
    def run(self):
        """Main routine for the ML_Trajectories task"""

        # Check input args are valid
        # validate_args(self)

        # Make sure trajectory choices are valid
        all_trajs = get_trajectory_list(self.ntraj)
        if self.which_trajs is None:
            which_trajs = all_trajs
            self.which_targets = [self.target] * len(which_trajs)
        else:
            which_trajs = self.which_trajs
            assert len(self.which_targets) == len(self.which_trajs)
            for traj_label in which_trajs:
                if traj_label not in all_trajs:
                    raise Exception(f"Invalid trajectory name: {traj_label}")

        # Set up calculator parameters dict
        if self.calc_seed is None:
            self.calc_seed = self.seed
        calc_params = {'calc_seed': self.calc_seed,
                       'calc_suffix': self.calc_suffix,
                       'calc_dir_suffix': self.calc_dir_suffix,
                       'calc_prefix': f'../../{self.calc_prefix}',  # MD will be run from subdirectory
                       'target': self.target}
        if self.calc_seed is not None:
            calc_params['calc_seed'] = self.calc_seed

        model = {}
        continuation_len = {}
        enough_data = {traj_label: False for traj_label in which_trajs}
        offsets = {traj_label: '' for traj_label in which_trajs}
        while not all(enough_data.values()):
            for traj_label, traj_target in zip(which_trajs, self.which_targets):

                # Find (or relax) initial geometry
                calc_params['calc_prefix'] = f'../{self.calc_prefix}'
                if isinstance(self.target, dict):
                    calc_params['calc_head'] = self.target[traj_target]

                # Check if we have a continuation to load in
                model[traj_label] = None
                if self.continuation:
                    # an unfinished run is under the incomplete name, a finished one under the final name
                    final, incomplete = md_output_names(self.seed, driving_target(traj_target), traj_label, self.traj_suffix)
                    continuation_trajfile = incomplete if os.path.exists(incomplete) else final
                    if os.path.exists(continuation_trajfile):
                        continuation_traj = Trajectory(continuation_trajfile)
                        continuation_len[traj_label] = len(continuation_traj)
                        model[traj_label] = continuation_traj[-1]
                        stde_last = committee_stde_per_atom(model[traj_label].get_potential_energy(),
                                                            len(model[traj_label]))
                        # This trajectory was stopped on committee disagreement, so it is finished:
                        # continuing would carry on from the poorly-described region
                        if (self.md_stop_stde_thresh is not None and continuation_len[traj_label] < self.nsnap
                                and stde_last > self.md_stop_stde_thresh):
                            print(f'# Continuation file {continuation_trajfile} found, containing {continuation_len[traj_label]} snapshots. '
                                  f'It was stopped on committee disagreement ({stde_last:.4f} > {self.md_stop_stde_thresh} eV/atom), so is finished.')
                            enough_data[traj_label] = True
                        # This model already blew up, so increase offset and load a new one
                        elif model[traj_label].get_temperature() > 10 * self.temp:
                            offsets[traj_label] += "A"
                            model[traj_label] = None
                            print(f'# Continuation file {continuation_trajfile} found, containing {continuation_len[traj_label]} snapshots. Abort was flagged, restarting.')
                        else:
                            print(f'# Continuation file {continuation_trajfile} found, containing {continuation_len[traj_label]} snapshots. Continuing.')
                        continuation_traj.close()
                    else:
                        self.continuation = False
                        continuation_len[traj_label] = 0
                        print(f'# No continuation file {continuation_trajfile} found, initialising from scratch')
                else:
                    continuation_len[traj_label] = 0

                # If no continuation, find initial geometry
                if model[traj_label] is None:
                    offset_traj_label = traj_label + offsets[traj_label]
                    model[traj_label] = find_initial_geometry(self.seed, self.wrapper.geom_opt,
                                                              calc_params, offset_traj_label)
                # Set up any constraints
                if self.constraints is not None:
                    from ase.constraints import FixBondLengths, Hookean, FixInternals
                    set_constraints = []
                    for c in self.constraints:
                        if isinstance(c, FixBondLengths):
                            bondlength = c.bondlengths[0]
                            atoms = c.pairs[0]
                            model[traj_label].set_distance(atoms[0], atoms[1], bondlength, fix=0)
                            model[traj_label].set_constraint(c)
                        if isinstance(c, Hookean):
                            del model[traj_label].constraints
                            set_constraints.append(c)
                        if isinstance(c, FixInternals):
                            set_constraints.append(c)
                    model[traj_label].set_constraint(set_constraints)

            # Evaluate initial energies
            for traj_label in which_trajs:
                if continuation_len[traj_label] >= self.nsnap or enough_data[traj_label]:
                    continue
                ep = self.wrapper.singlepoint(model[traj_label], "test", calc_params)[0]
                ek = model[traj_label].get_kinetic_energy()
                print(f'# Trajectory {traj_label} initial potential energy = {ep - self.wrapper.atom_e}')
                print(f'# Trajectory {traj_label} initial   kinetic energy = {ek}')

            # Current path will be two levels down from starting path in MD run, so adjust prefix
            calc_params['calc_prefix'] = f'../../{self.calc_prefix}'
            for traj_label, traj_target in zip(which_trajs, self.which_targets):
                if enough_data[traj_label]:
                    continue
                # Pass in routine to actually run MD into generic Snapshot MD driver
                status = generate_md_trajectory(model[traj_label], self.seed, traj_target, traj_label, self.traj_suffix,
                                                traj_wrapper=self.wrapper, count_snaps=self.nsnap, count_equil=self.nequil,
                                                md_steps=self.md_steps, md_timestep=self.md_timestep,
                                                md_friction=self.md_friction, store_full_traj=self.store_full_traj,
                                                temp=self.temp, traj_calc_params=calc_params, dynamics=self.dynamics,
                                                snap_target=self.snap_target if not self.recalculate_carved_traj else None,
                                                snap_wrapper=self.snap_wrapper if not self.recalculate_carved_traj else None,
                                                snap_calc_params=self.snap_calc_params if not self.recalculate_carved_traj else None,
                                                continuation=self.continuation, debugger=self.debugger,
                                                stop_stde_thresh=self.md_stop_stde_thresh)
                self.continuation = True
                final, incomplete = md_output_names(self.seed, driving_target(traj_target), traj_label, self.traj_suffix)
                output_trajfile = final if os.path.exists(final) else incomplete
                if os.path.exists(output_trajfile):
                    output_traj = Trajectory(output_trajfile)
                    continuation_len[traj_label] = len(output_traj)
                    output_traj.close()
                    print(f'# Output file {output_trajfile} now contains {continuation_len[traj_label]} of {self.nsnap} snapshots.')
                else:
                    # aborted during equilibration, before any snapshots were written
                    continuation_len[traj_label] = 0
                if continuation_len[traj_label] >= self.nsnap:
                    enough_data[traj_label] = True
                elif status == 'stopped':
                    # Stopped on committee disagreement: keep what we have, do not restart
                    print(f'# Trajectory {traj_label} stopped on committee disagreement with '
                          f'{continuation_len[traj_label]} of {self.nsnap} snapshots: treating it as finished')
                    enough_data[traj_label] = True
                else:
                    # The run was aborted (eg temperature blew up): restart from a different
                    # initial snapshot, giving up after max_restarts attempts
                    offsets[traj_label] += 'A'
                    if len(offsets[traj_label]) > self.max_restarts:
                        print(f'# Trajectory {traj_label} aborted {len(offsets[traj_label])} times: giving up '
                              f'with {continuation_len[traj_label]} of {self.nsnap} snapshots')
                        enough_data[traj_label] = True
                        # what there is counts as the finished trajectory (for every target evaluated)
                        for t in (traj_target if isinstance(traj_target, list) else [traj_target]):
                            finish_md_output(self.seed, t, traj_label, self.traj_suffix)
                    else:
                        print(f'# Trajectory {traj_label} aborted: restart {len(offsets[traj_label])} of '
                              f'{self.max_restarts} from a new initial snapshot')

    def make_parser(self):
        "Makes a parser to setup input variables"

        import argparse
        from ase.units import AUT

        main_help = ('ML_Trajectories.py: Generate trajectory files using a pre-trained ML-based calculator.\n')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help, epilog=epi_help)
        parser.add_argument('--seed', '-s', type=str, help="Base name of the system to simulate, usually '{solu}_{solv}' (or '{solu}' with no solvent). Used to name the trajectory directory and files and to find the initial geometries.")
        parser.add_argument('--traj_suffix', '-S', default="mldyn", type=str, help="Suffix of the trajectory files generated, which are named '{seed}_{targ}_{label}_{traj_suffix}.traj' (where {targ} is eg 'gs' or 'es1' and {label} is the trajectory label A, B, ...). Also names the '{traj_suffix}_equil', '{traj_suffix}_md' and '{traj_suffix}_snaps' working directories.")
        parser.add_argument('--calc_seed', '-Z', default=None, type=str, help="Base name of the calculator to use (eg 'all_{solv}'), if different from seed; '{solu}' and '{solv}' are replaced by the solute and solvent names. If None, seed is used.")
        parser.add_argument('--calc_suffix', '-C', default='', type=str, help="Suffix naming the calculator that drives the MD (eg 'MACEac0u'), often encoding the ML method, generation and committee member.")
        parser.add_argument('--calc_dir_suffix', '-D', default=None, type=str, help="Suffix of the directory containing the calculator, named '{calc_seed}_{targ}_{calc_dir_suffix}'. Unlike the training and testing tasks, it is not defaulted to calc_suffix, so it should be set.")
        parser.add_argument('--traj_dir_suffix', '-Y', default=None, type=str, help="Suffix of the directory in which the trajectories are run and written, named '{seed}_{targ}_{traj_dir_suffix}'. If None, '{calc_dir_suffix}_mldyn' is used (or '{calc_suffix}_mldyn' if calc_dir_suffix is None).")
        parser.add_argument('--calc_prefix', '-P', default='', type=str, help="Prefix for the path to the calculator, relative to the base directory (the task adds '../' or '../../' as it runs in subdirectories).")
        parser.add_argument('--target', '-t', default=0, type=int, help="Electronic state on which the MD is run: 0 for the ground state, n for excited state n. For a multi-head calculator, a dict mapping state indices to state names (eg {0: 'gs', 1: 'es1'}), the head being chosen from which_targets.")
        parser.add_argument('--md_timestep', '-q', default=10 * AUT, type=float, help="MD timestep in ASE time units (eg 0.5*ase.units.fs). May be a dict with keys 'EQ' and 'MD' giving separate values for the equilibration and production (snapshot) runs.")
        parser.add_argument('--md_friction', '-r', default=0.002, type=float, help="Friction coefficient for Langevin dynamics, in inverse ASE time units. May be a dict with keys 'EQ' and 'MD' giving separate values for the equilibration and production runs; 0.0 turns the thermostat off.")
        parser.add_argument('--md_steps', '-Q', default=100, type=int, help='Number of MD steps in each run between snapshots (and in each equilibration run).')
        parser.add_argument('--md_init_traj_link', '-I', default=None, type=str, help="Path, relative to the base directory, of a trajectory of starting geometries, linked as '{seed}_{targ}_md_init.traj' in the trajectory directory; each trajectory starts from a randomly chosen frame of it. May contain '{solu}' and '{solv}'. May be a list of paths, whose trajectories are merged; if None, the MD starts from the optimised geometry.")
        parser.add_argument('--geom_prefix', '-G', default='', type=str, help="Directory, relative to the base directory, containing the optimised initial geometries: the driver appends '/is_opt_{solv}' (or '/opt' with no solvent) and copies '{seed}_{targ}.xyz', '{solu}.xyz' and '{solv}.xyz' from there if present. May contain '{solu}' and '{solv}'.")
        parser.add_argument('--continuation', '-o', default=False, type=bool, help='If True, continue any existing trajectories from their last frame, skipping equilibration and appending to their output files; trajectories with no output file yet are started afresh. If False, existing output is overwritten.')
        parser.add_argument('--debugger', '-g', default=None, help='Optional object whose check(atoms) method is called after each MD run (equilibration and production), as a sanity check. None for no checks.')
        parser.add_argument('--store_full_traj', '-f', default=False, type=bool, help='If True, store the full step-by-step trajectory of every MD run, not just the snapshots taken every md_steps steps.')
        parser.add_argument('--freq', '-F', default=False, type=bool, help='If True, post-process the trajectory into an IR spectrum. Currently not used.')
        parser.add_argument('--temp', '-T', default=300.0, type=float, help='Thermostat temperature in K. A trajectory is treated as having blown up, and restarted, if its temperature exceeds ten times this.')
        parser.add_argument('--ntraj', '-n', default=1, type=int, help='Total number of trajectories (labelled A, B, C, ...) in the ensemble. If which_trajs is None, all of them are run.')
        parser.add_argument('--nsnap', '-N', default=200, type=int, help='Number of snapshots to record in each trajectory, one every md_steps steps after equilibration.')
        parser.add_argument('--nequil', '-e', default=10, type=int, help='Number of equilibration runs, each of md_steps steps, performed and discarded before snapshots are recorded.')
        parser.add_argument('--max_restarts', default=10, type=int, help='Number of times an aborted (eg blown-up) trajectory is restarted from a new initial geometry before giving up and keeping the snapshots recorded so far.')
        parser.add_argument('--md_stop_stde_thresh', default=None, type=float, help='Stop a trajectory once the committee energy standard deviation per atom (eV) exceeds this. Checked during equilibration as well as at snapshots; frames up to and including the one that crossed it are kept, and the trajectory is then treated as finished. None (default) disables the check.')
        parser.add_argument('--which_trajs', '-w', default=None, type=str, help="List of the trajectory labels (eg ['A', 'B']) to run in this task. If None, all ntraj trajectories are run on target; otherwise which_targets must give the target of each trajectory.")
        parser.add_argument('--carve_trajectory_radius', '-R', default=None, type=float, help="If set, after the MD, carve clusters from each snapshot containing the solute and all solvent molecules within this radius (Angstrom) of it, using the Clusters task. May be a dict keyed by '{solu}_{solv}' or by solvent name. None for no carving.")
        parser.add_argument('--carve_trajectory_max_atoms', '-m', default=None, type=float, help='Maximum number of atoms in each carved cluster. None for no limit.')
        parser.add_argument('--recalculate_carved_traj', '-J', default=False, type=bool, help="If True, recalculate the energies and forces of the carved clusters with snap_wrapper and snap_calc_params, writing '{traj_suffix}_recalc_{carved_suffix}' trajectories, instead of evaluating snapshots with snap_wrapper during the MD. Requires carve_trajectory_radius.")
        parser.add_argument('--constraints', '-c', default=None, type=str, help='ASE constraint objects to apply during the MD, as a list (FixBondLengths, Hookean and FixInternals are supported), or a dict mapping solute names to such lists. None for no constraints.')
        parser.add_argument('--dynamics', '-d', default=None, type=str, help="Object specifying the type of dynamics, passed to the wrapper's run_md (eg with type 'LANG' for Langevin or 'NPT'). If None, Langevin dynamics with md_friction is used.")
        parser.add_argument('--carved_suffix', default=None, type=str, help="Suffix added to traj_suffix in the names of the carved-cluster trajectories, '{traj_suffix}_{carved_suffix}'. If None, 'carved' is used.")
        parser.add_argument('--corr_traj', default=False, type=bool, help="If True, when recalculating carved clusters, also calculate each cluster with the solute removed ('_nosolu' trajectories), for use in correction calculations.")

        return parser

    def validate_args(self, args):
        default_args = self.make_parser().parse_args(['--seed', 'a'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


def load_trajectory_dipole(seed_state_str, traj_suffix, ntraj, nsnaps, mdsteps, extension='.traj'):
    '''
    Loads a set of saved trajectory files and extracts the dipole moment as a 
    function of time
    '''

    from ase.io import read, Trajectory
    from esteem.trajectories import get_trajectory_list
    import numpy as np

    # Storage for result
    mu_t = np.zeros((ntraj, nsnaps * mdsteps, 3))
    # Names of trajectories
    chars = get_trajectory_list(ntraj)
    trajname = chars[0]
    # Loop over trajectories, reading them in and storing dipole moments in an array
    for i, trajname in enumerate(chars):
        k = 0
        for j in range(0, nsnaps):
            if nsnaps == 0:
                file = f'{seed_state_str}_{trajname}_{traj_suffix}{extension}'
            else:
                file = f'{seed_state_str}_{trajname}_{traj_suffix}{j:04}{extension}'
            traj = Trajectory(file)
            print(file)  # progress update
            for f in traj[1:]:
                mu_t[i, k] = f.get_dipole_moment()
                k = k + 1

    return mu_t


def calculate_ir_spectrum(mu_t, dt, freq_scale_fac, sigma):
    '''
    Processes the dipole moment as a function of time for a collection of trajectories,
    to calculate IR absorption spectrum
    '''

    import numpy as np

    # Take gradient of mu(t) to get dmu/dt
    mu_dot = np.gradient(mu_t, dt, axis=(1,))

    # Take FFT of dmu/dt and get corresponding frequencies (scaled, eg to cm^-1)
    mu_dot_tilde = np.fft.fftn(mu_dot, axes=(1,))
    omega = np.fft.fftfreq(len(mu_dot[0]), dt) * freq_scale_fac

    # Average over snapshots and take dot product with self to get autocorrelation
    mu_dot_tilde_av = np.average(mu_dot_tilde, axis=0)
    mu_dot_tilde_av_mag = (np.sum(mu_dot_tilde_av * np.conj(mu_dot_tilde_av), axis=1))

    # Convolve with Gaussian of width sigma (in cm^-1)
    gaussian = np.exp(-(omega / sigma)**2 / 2)
    mu_dot_tilde_av_mag_conv = np.convolve(mu_dot_tilde_av_mag, gaussian, mode="full")

    return mu_dot_tilde_av_mag_conv, omega


# # Command-line driver
def get_parser():
    mltraj = MLTrajTask()
    return mltraj.make_parser()


# Not much need for this now as wrapper will need defining for this to be useful
if __name__ == '__main__':

    mltraj = MLTrajTask()

    # Parse command line values
    args = mltraj.make_parser.parse_args()
    for arg in vars(args):
        setattr(mltraj, arg, getattr(args, arg))
    print('#', args)

    # Run main routine
    mltraj.run()
