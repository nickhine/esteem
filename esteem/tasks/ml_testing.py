#!/usr/bin/env python
# coding: utf-8

"""Defines a task to test a Machine Learning calculator by comparing its results
to those of an existing trajectory or set of trajectories"""

# Load essential modules
import sys
import os
import string
from ase.io.trajectory import Trajectory

from esteem.trajectories import merge_traj, get_trajectory_list
import numpy as np
from esteem.trajectories import compare_traj_to_traj, atom_energy


class MLTestingTask:

    def __init__(self, **kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'mltest'
        self.train_params = {}
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

    def get_trajnames(self):
        all_trajs = get_trajectory_list(self.ntraj)
        if self.which_trajs is None:
            which_trajs = all_trajs
        else:
            which_trajs = self.which_trajs
            if isinstance(which_trajs, dict):
                which_trajs_dict = which_trajs
                which_trajs = []
                which_trajs = list(which_trajs_dict)
            for trajname in which_trajs:
                if trajname not in all_trajs:
                    raise Exception("Invalid trajectory name: ", trajname)
        trajstem = self.wrapper.calc_filename(self.seed, self.target, prefix=self.calc_prefix, suffix="")
        trajnames = [trajstem + s + '_' + self.traj_suffix + '.traj'
                     for s in which_trajs]

        return which_trajs, trajnames

    # Main routine
    def run(self):
        """Main routine for the ML_Testing task.

        Merges the trajectories to test against, evaluates each frame with the calculator
        (see :func:`compare_wrapper_to_traj`), optionally subtracts reference energies, and
        compares and plots the energies and forces with
        :func:`esteem.trajectories.compare_traj_to_traj`."""

        from os.path import commonprefix

        # Check input args are valid
        # validate_args(self)

        # Get strings for trajectory names
        if self.calc_seed is None:
            self.calc_seed = self.seed
        traj_prefix = self.traj_prefix
        # Assume path is relative to base directory, unless blank
        if traj_prefix != "":
            traj_prefix = "../" + traj_prefix

        trajfn = self.wrapper.calc_filename(self.calc_seed, self.target, prefix="", suffix=self.traj_suffix)
        trajstem = self.wrapper.calc_filename(self.calc_seed, self.target, prefix=traj_prefix, suffix="")

        all_trajs = get_trajectory_list(self.ntraj)
        traj_suffix = self.traj_suffix
        if self.which_trajs is None:
            which_trajs = all_trajs
        else:
            which_trajs = self.which_trajs
            if "_" in which_trajs:
                traj_suffix = which_trajs.split("_", 1)[1]
                which_trajs = which_trajs.split("_", 1)[0]
            traj_target = None
            for traj in which_trajs:
                if self.which_targets[traj] != traj_target and traj_target is not None:
                    raise Exception("Non-matching targets in test trajectories")
                traj_target = self.which_targets[traj]
                print('# Target set to: ', traj_target)
            # for trajname in which_trajs:
            #    if trajname not in all_trajs:
            #        raise Exception("Invalid trajectory name:",trajname)

        # If all trajectories exist, test against them
        trajnames = [trajstem + s + '_' + traj_suffix + '.traj'
                     for s in which_trajs]
        print('# Merging trajectories: ', trajnames)
        if not all([os.path.isfile(f) for f in trajnames]):
            raise Exception('# Missing Trajectory file(s): ',
                            [f for f in trajnames if not os.path.isfile(f)])
        if not all([os.path.getsize(f) > 0 for f in trajnames]):
            raise Exception('# Empty Trajectory file(s) found: ',
                            [f for f in trajnames if os.path.getsize(f) == 0])

        # Test the ML calculator against the results in the merged trajectory
        which_traj_str = ''.join(which_trajs)
        if len(which_traj_str) > 20:
            which_traj_str = which_traj_str[0:20]

        if isinstance(self.calc_suffix, dict):
            calc_suffix = commonprefix(list(self.calc_suffix.keys()))
        else:
            calc_suffix = self.calc_suffix
        # test_dir = f'{seed_state_str}_{calc_suffix}_test'
        intrajfile = trajfn + "_" + calc_suffix + "_" + which_traj_str + '_merged.traj'
        if os.path.isfile(intrajfile):
            print(f'# Merged input trajectory file {intrajfile} already exists. Overwriting!')
        else:
            print(f'# Writing merged input trajectory file {intrajfile}')
        merge_traj(trajnames, intrajfile)
        intraj = Trajectory(intrajfile)
        # Set up the calculator parameters
        print(f"# Loading Calculator with params: {self.calc_seed} {self.target} {self.calc_prefix} {self.calc_dir_suffix} {self.calc_suffix}")
        # self.wrapper.calc = None
        calc_params = {'calc_seed': self.calc_seed,
                       'calc_suffix': self.calc_suffix,
                       'calc_dir_suffix': self.calc_dir_suffix,
                       'calc_prefix': f'../{self.calc_prefix}',  # Testing will be run from subdirectory
                       'calc_head': traj_target,
                       'target': self.target}  # Run on specific target (calculator for all targets may be loaded if using multihead)
        if hasattr(self.wrapper, 'update_atom_e'):
            self.wrapper.update_atom_e = True
        # Load the calculator
        # self.wrapper.load(self.calc_seed,self.target,'../'+self.calc_prefix,self.calc_suffix,self.calc_dir_suffix)
        output_traj = self.output_traj
        if output_traj is None:
            output_traj = calc_suffix + "_" + which_traj_str + "_test"
        outtrajfile = trajfn + "_" + output_traj + '.traj'
        if os.path.isfile(outtrajfile):
            print(f'# Warning: output trajectory file {outtrajfile} already exists. Overwriting!')
        else:
            print(f'# Writing output trajectory file {outtrajfile}')
        compare_wrapper_to_traj(self.wrapper, calc_params, intraj, outtrajfile)

        # Open temporary trajectory for reading
        outtraj = Trajectory(outtrajfile)
        if self.ref_mol_dir is not None:
            intrajfile_refsub = trajfn + "_" + calc_suffix + "_" + which_traj_str + '_refsub.traj'
            self.subtract_reference_energies(intraj, intrajfile_refsub, traj_target)
            intraj.close()
            intraj = Trajectory(intrajfile_refsub)
            outtrajfile_refsub = trajfn + "_" + output_traj + '_refsub.traj'
            self.subtract_reference_energies(outtraj, outtrajfile_refsub, traj_target)
            outtraj.close()
            outtraj = Trajectory(outtrajfile_refsub)

        # Finally, plot comparison
        clabel = 'RMS Force component deviation (eV/Ang)'
        xlabel = 'Trajectory Energy (eV)'
        ylabel = 'Calculator Energy (eV)'
        compare_traj_to_traj(intraj, outtraj, self.plotfunc, self.plotfile, xlabel, ylabel, clabel)

        if self.cleanup:  # optional cleanup
            os.remove(intrajfile)
            os.remove(outtrajfile)
            if self.ref_mol_dir is not None:
                os.remove(intrajfile_refsub)
                os.remove(outtrajfile_refsub)

    def subtract_reference_energies(self, trajin, trajout_file, traj_target):
        """Subtract reference energies from a trajectory to just get energy above reference zero.

        The calculator's energies for the isolated solute and solvent molecules (from ref_mol_dir)
        are subtracted from each frame of trajin, as the solute energy plus one solvent energy per
        solvent molecule, and the frames are written to the trajectory file trajout_file."""
        from esteem.drivers import get_solu_solv_names
        from esteem.trajectories import targstr
        from esteem.tasks.clusters import get_ref_mol_energy

        ref_solu, ref_solv = get_solu_solv_names(self.seed)
        if ref_solv == "NO_SOLVENT_FOUND":
            ref_solv = None
        trajout = Trajectory(trajout_file, 'w')
        targ = 0
        if targ == 0:
            ref_solu_t = ref_solu
        else:
            ref_solu_t = f'{ref_solu}_{targstr(targ)}'
        ref_mol_dir = self.ref_mol_dir.replace("{targ}", targstr(targ))
        ref_solu_dir = f'../{ref_mol_dir}'
        ref_mol_dir = self.ref_mol_dir.replace("{targ}", targstr(0))
        ref_solv_dir = f'../{ref_mol_dir}'
        calc_params = {'calc_seed': self.calc_seed,
                       'calc_suffix': self.calc_suffix,
                       'calc_dir_suffix': self.calc_dir_suffix,
                       'calc_prefix': f'../{self.calc_prefix}',
                       'calc_head': traj_target,
                       'target': self.target}
        # Read in Reference E, f, p
        if ref_solv is not None:
            ref_mol_xyz = f'{ref_solv_dir}/is_opt_{ref_solv}/{ref_solv}.xyz'
            solv_energy, solv_model = get_ref_mol_energy(self.wrapper, ref_solv, ref_solv, calc_params, ref_mol_xyz, ref_solv_dir)
            if isinstance(solv_energy, np.ndarray):
                solv_energy = np.mean(solv_energy)
            print('# Solvent reference energy: ', solv_energy)
            ref_mol_xyz = f'{ref_solu_dir}/is_opt_{ref_solv}/{ref_solu}.xyz'
        else:
            ref_mol_xyz = f'{ref_solu_dir}/opt/{ref_solu}.xyz'
            solv_energy = 0.0
        solu_energy, solu_model = get_ref_mol_energy(self.wrapper, ref_solu, ref_solv, calc_params, ref_mol_xyz, ref_solu_dir)
        if isinstance(solu_energy, np.ndarray):
            solu_energy = np.mean(solu_energy)
        print('# Solute reference energy: ', solu_energy)

        for i, frame in enumerate(trajin):
            n = len(frame) - len(solu_model)
            if n > 0:
                n = int(n / len(solv_model))
            e = frame.get_potential_energy()
            e = e - solu_energy - n * solv_energy
            frame.calc.results["energy"] = e
            trajout.write(frame, **frame.calc.results)
        trajout.close()

    # Generate default arguments and return as parser
    def make_parser(self):

        import argparse

        main_help = ('ML_Testing.py: Test an ML-based Calculator by comparing it to trajectories.')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help, epilog=epi_help)
        parser.add_argument('--seed', '-s', type=str, help="Base name of the system whose trajectories are tested, usually '{solu}_{solv}'. Used to find the solute and solvent names for reference energies, and replaced per trajectory set by ref_mol_seed_dict if that has an entry.")
        parser.add_argument('--calc_seed', '-Z', default=None, type=str, help="Base name of the calculator to test (eg 'all_{solv}'), if different from seed; '{solu}' and '{solv}' are replaced by the solute and solvent names. If None, seed is used.")
        parser.add_argument('--calc_suffix', '-S', default="", type=str, help="Suffix naming the calculator to test (eg 'MACEac0u'). May be a dict mapping several calculator suffixes to their random seeds, to test a committee, in which case the common prefix of the keys is used in output file names.")
        parser.add_argument('--calc_dir_suffix', '-D', default=None, type=str, help="Suffix of the directory containing the calculator, named '{calc_seed}_{targ}_{calc_dir_suffix}' where {targ} is eg 'gs' or 'es1'. Also names the directory '{calc_seed}_{targ}_{calc_dir_suffix}_test' in which the test is run. If None, calc_suffix is used.")
        parser.add_argument('--calc_prefix', '-P', default="", type=str, help="Prefix for the path to the calculator, relative to the base directory (the task adds '../' as it runs in a subdirectory). Also prefixed to the names of the links created from traj_links.")
        parser.add_argument('--target', '-t', default=0, type=int, help="Electronic state the calculator is tested for: 0 for the ground state, n for excited state n. For a multi-head calculator, a dict mapping state indices to state names (eg {0: 'gs', 1: 'es1'}), the head being chosen from which_targets.")
        parser.add_argument('--output_traj', '-o', default=None, type=str, help="Suffix for the trajectory file of calculator results, written as '{calc_seed}_{targ}_{traj_suffix}_{output_traj}.traj'. If None, '{calc_suffix}_{labels}_test' is used, where {labels} are the trajectory labels tested.")
        parser.add_argument('--plotfile', '-p', default=None, nargs='?', const="TkAgg", type=str, help="Name of the image file (eg a .png) to which the plot comparing calculator and trajectory energies is written; plot data and statistics are also written alongside it. 'show' displays the plot instead, and None makes no plot. May contain '{solu}' and '{solv}'; it is set automatically for each set when which_trajs is a dict.")
        parser.add_argument('--plotfunc', '-F', default=None, help='Function used to make the comparison plot, called as plotfunc(e_traj, e_calc, rms_force_dev, xlabel, ylabel, clabel, stats, plotfile). If None, esteem.trajectories.plot_diff is used.')
        parser.add_argument('--cleanup', '-C', default=True, type=bool, help='If True, delete the merged input trajectory, the calculator-results trajectory and any reference-corrected copies of them once the test has finished.')
        parser.add_argument('--traj_prefix', '-Q', default="", type=str, help="Directory, relative to the base directory, containing the trajectory files to test against (links from traj_links are made there). May contain '{solu}' and '{solv}'. If empty, the test directory is used.")
        parser.add_argument('--traj_suffix', '-T', default="training", type=str, help="Suffix of the trajectory files tested against, which are named '{calc_seed}_{targ}_{label}_{traj_suffix}.traj'. Overridden if an entry of which_trajs has the form 'labels_suffix'.")
        parser.add_argument('--ntraj', '-n', default=1, type=int, help='Total number of trajectories (labelled A, B, C, ...) available. If which_trajs is None, all of them are tested against.')
        parser.add_argument('--which_trajs', '-w', default=None, type=str, help="Trajectory labels to test against, as a string or list (eg 'ABC'), optionally followed by '_' and a suffix that replaces traj_suffix. May be a dict mapping a name to each such entry, in which case each is tested separately with its own plot file. If None, all ntraj trajectories are used; otherwise which_targets must give the state of each label.")
        parser.add_argument('--which_trajs_valid', '-v', default=None, type=str, help='Trajectory labels of a validation set. Currently only used when assembling tasks with esteem.active_learning; the testing task itself does not use it.')
        parser.add_argument('--which_trajs_test', '-u', default=None, type=str, help='Trajectory labels of a test set. Currently only used when assembling tasks with esteem.active_learning; the testing task itself does not use it.')
        parser.add_argument('--traj_links', '-L', default=None, type=dict, help="Dict mapping each trajectory label to the path of the trajectory file to use for it (or a list of paths), relative to the base directory. Symbolic links to these files are created in the traj_prefix directory; paths may contain '{solu}' and '{solv}'.")
        parser.add_argument('--traj_links_valid', '-V', default=None, type=dict, help='Dict of links for validation trajectories, as for traj_links. Not used by the testing task itself, but if it is not None, esteem.active_learning also adds validation trajectories when assembling tasks.')
        parser.add_argument('--traj_links_test', '-U', default=None, type=dict, help='Dict of links for test trajectories, as for traj_links. Not used by the testing task itself, but if it is not None, esteem.active_learning also adds test trajectories when assembling tasks.')
        parser.add_argument('--ref_mol_seed_dict', '-z', default={}, type=dict, help="Dict mapping the names used as keys in which_trajs to the seed (eg '{solu}_{solv}') to use for that trajectory set, for sets drawn from systems with different names. The seed determines which reference energies are subtracted.")
        parser.add_argument('--ref_mol_dir', '-r', default=None, type=str, help="Directory, relative to the base directory, of the Solutes task outputs for the isolated solute and solvent molecules; '{targ}' is replaced by a state string (eg 'gs'). Their optimised geometries are evaluated with the calculator, and the solute energy plus one solvent energy per solvent molecule is subtracted from all energies before plotting. If None, no reference energies are subtracted.")

        return parser

    def validate_args(self, args):
        default_args = self.make_parser().parse_args(['--seed', 'a'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Comparison of results from trajectory to calculator


def compare_wrapper_to_traj(wrapper, calc_params, trajin, trajout_file):
    """Compare the energy and force predictions of a calculator to results in an existing trajectory.

    Each frame of trajin is evaluated with ``wrapper.singlepoint`` using calc_params, the deviations
    are printed, and the calculator's energy, forces and dipole are written to the trajectory
    file trajout_file."""

    trajout = Trajectory(trajout_file, 'w')

    # Loop over the frames in the input trajectory
    for i, frame in enumerate(trajin):

        # Read in total energy and forces from trajectory
        e_traj = frame.get_potential_energy()
        f_traj = frame.get_forces()
        d_traj = frame.get_dipole_moment()

        err = False
        seed = f"{calc_params['calc_seed']}{i:4d}"

        e_calc, f_calc, d_calc = wrapper.singlepoint(frame, seed, calc_params,
                                                     forces=True, dipole=True)

        # Calculate RMS and Max force errors
        rms_fd = np.sqrt(np.mean((f_traj - f_calc)**2))
        max_fd = np.max(np.sqrt((f_traj - f_calc)**2))
        rms_dd = np.sqrt(np.mean((d_traj - d_calc)**2))

        # Print header for columns
        if (i == 0):
            print('#Idx    E_traj (eV)  E_calc (eV)      E_diff (eV)       RMS_fd    MAX_fd   RMS_dd')

        if isinstance(calc_params["calc_suffix"], dict):
            print(f'{i:4d} {e_traj:12.5f} {np.mean(e_calc):12.5f} {np.std(e_calc):8.5f} {np.mean(e_traj - e_calc):8.5f} {np.mean(np.abs(e_traj - e_calc)):8.5f} {rms_fd:8.5f} {max_fd:8.5f}')
        else:
            print('%4d   %14.5f %14.5f     %12.8f     %8.5f  %8.5f %8.5f' %
                  (i, e_traj, e_calc, e_traj - e_calc, rms_fd, max_fd, rms_dd))

        # Assemble dictionary of properties to write to the output trajectory
        kw = {'dipole': d_calc,
              # 'charges': q_calc,
              'energy': e_calc,
              'forces': f_calc}

        # Write to trajectory
        trajout.write(frame, **kw)

    trajout.close()


# # Command-line driver


def get_parser():
    mltest = MLTestingTask()
    return mltest.make_parser()


if __name__ == '__main__':

    from esteem import wrappers
    mltest = MLTestingTask()

    # Parse command line values
    args = get_parser().parse_args()
    for arg in vars(args):
        setattr(mltest, arg, getattr(args, arg))
    print('#', args)
    mltest.wrapper = wrappers.amp.AMPWrapper()

    # Run main program
    mltest.run()
