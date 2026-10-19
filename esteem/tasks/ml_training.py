#!/usr/bin/env python
# coding: utf-8
"""Defines a task to train a Machine Learning calculator on trajectories of snapshots,
by merging them and calling the train() method of the ML wrapper"""

from esteem.trajectories import merge_traj, diff_traj, get_trajectory_list, targstr
import sys
import os
import string
from shutil import copyfile


def make_diff_trajs(target_dict, which_targets, trajnames):
    which_diff = target_dict['diff']
    # Split the strings either side of the '_m_' separator (short for "minus")
    if '_m_' in which_diff:
        diff_targets = which_diff.split('_m_')
    else:
        raise Exception(f"Invalid difference target string: {which_diff}. Expected format: 'target1_m_target2'")
    trajnames_diff = []
    itarget = diff_targets[0]
    jtarget = diff_targets[1]
    # Create lists of the trajectories for each of these targets
    itrajs = [traj for traj, targ in zip(trajnames, which_targets) if targ == itarget]
    jtrajs = [traj for traj, targ in zip(trajnames, which_targets) if targ == jtarget]
    print(f'# Calling diff_traj with {itarget} {jtarget} {trajnames}')
    assert (len(itrajs) == len(jtrajs))
    # iterate over both lists simultaneously
    for itraj, jtraj in zip(itrajs, jtrajs):
        # Create new name for output trajectory by splitting on concatenation
        # of target names and substituting the last instance with "diff"
        ktraj = 'diff'.join(itraj.rsplit(f'{jtarget}{itarget}', 1))
        print(f'# Calling diff_traj with {itraj} {jtraj} {ktraj}')
        # Process trajectories to write difference trajectory
        diff_traj(itraj, jtraj, ktraj)
        # Store the name of the difference trajectory for later use
        trajnames_diff.append(ktraj)
    return trajnames_diff


class MLTrainingTask:

    def __init__(self, **kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'mltrain'
        self.train_params = {}
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

    def get_which_trajs(self, prefix=""):
        """Returns the trajectory labels and targets used for training (or validation, if 'valid' is in prefix)"""
        all_trajs = get_trajectory_list(self.ntraj)
        if 'valid' in prefix:
            which_trajs = self.which_trajs_valid
            which_targets = self.which_targets_valid
        else:
            which_trajs = self.which_trajs
            which_targets = self.which_targets
        if which_trajs is None:
            which_trajs = all_trajs
            which_targets = [self.target] * len(which_trajs)
        else:
            for trajname in which_trajs:
                if trajname not in all_trajs:
                    raise Exception("Invalid trajectory name: ", trajname)
        return which_trajs, which_targets

    def get_trajnames(self, prefix=""):
        which_trajs, which_targets = self.get_which_trajs(prefix)
        trajstem = self.wrapper.calc_filename(self.seed, self.target, prefix=self.calc_prefix, suffix="")
        trajnames = [f'{trajstem}{s}_{self.traj_suffix}.traj' for s in which_trajs]

        return which_trajs, which_targets, trajnames

    # Main routine
    def run(self):
        """Main routine for the ML_Training task.

        Merges the training trajectories for each target into
        '{calc_prefix}{seed}_{targ}_{traj_suffix}_merged_{calc_suffix}.traj' (and the validation
        data, split off at random or from which_trajs_valid, into the corresponding '_validmerged_'
        file), then calls the wrapper's train() method and returns the calculator it gives."""

        # Check input args are valid
        # validate_args(args)

        # Get name of folder where calculator files live
        dirname = self.wrapper.calc_filename(self.seed, self.target, prefix="", suffix=self.calc_dir_suffix)

        # If we need an atom trajectory, copy it from traj_suffix to calc_suffix:
        if hasattr(self.wrapper, 'atom_energies'):
            atom_traj_file = f'{self.seed}_atoms_{self.traj_suffix}.traj'
            if os.path.isfile(atom_traj_file):
                atom_calc_file = f'{dirname}/{self.seed}_atoms_{self.calc_suffix}.traj'
                if not os.path.isfile(atom_calc_file):
                    print(f'# Copying from {atom_traj_file} to {atom_calc_file} for atom energies')
                    copyfile(atom_traj_file, atom_calc_file)
                else:
                    print(f'# Using {atom_calc_file} for atom energies')
            else:
                raise Exception(f'# Trajectory file {atom_traj_file} not found for atom energies')

        # If we are training on energy differences, calculate these now
        prefs = [""]
        if self.which_trajs_valid is not None:
            prefs = ["", "valid"]
            separate_valid = True
        else:
            separate_valid = False
            valid_fraction = self.valid_fraction
            if hasattr(self, 'rand_seed'):
                rand_seed = self.rand_seed
            else:
                rand_seed = 123
        trajfile_dict = {}
        validfile_dict = {}
        split_seed_dict = {}
        for prefix in prefs:
            # If all trajectories exist, merge them
            which_trajs, which_targets, trajnames = self.get_trajnames(prefix)
            target_dict = self.target
            if not isinstance(self.target, dict):
                target_dict = {self.target: str(self.target)}

            # Special handling for difference trajectories
            if 'diff' in target_dict:
                trajnames_diff = make_diff_trajs(target_dict, which_targets, trajnames)
            # Iterate over all targets
            for target in target_dict:
                targetstr = target_dict[target]
                # Special handling for difference trajectories
                if target == 'diff':
                    trajnames_target = trajnames_diff
                else:
                    if isinstance(self.target, dict):
                        # Create a list of the trajectories for this target
                        trajnames_target = [traj for traj, targ in zip(trajnames, which_targets) if targ == targetstr]
                    else:
                        # If only one target, use all trajectories
                        trajnames_target = trajnames
                # Construct the base of the filename for the merged trajectory
                trajfn = self.wrapper.calc_filename(self.seed, target, prefix=self.calc_prefix, suffix=self.traj_suffix)
                print(f'# Trajectories to merge: {trajnames_target} for target {targetstr}', flush=True)
                # Check that all the trajectories exist and are non-empty
                if all([os.path.isfile(f) for f in trajnames]):
                    if all([os.path.getsize(f) > 0 for f in trajnames]):
                        if separate_valid:
                            if prefix == "":
                                # Merge the trajectories for training
                                trajfile = f'{trajfn}_{prefix}merged_{self.calc_suffix}.traj'
                                merge_traj(trajnames_target, trajfile)
                            if prefix == "valid":
                                # Merge the trajectories for validation
                                validfile = f'{trajfn}_{prefix}merged_{self.calc_suffix}.traj'
                                merge_traj(trajnames_target, validfile)
                            else:
                                validfile = None
                        else:
                            # Merge the trajectories for training and validation (splitting as required)
                            pref = prefix
                            trajfile = f'{trajfn}_{pref}merged_{self.calc_suffix}.traj'
                            pref = 'valid'
                            validfile = f'{trajfn}_{pref}merged_{self.calc_suffix}.traj'
                            merge_traj(trajnames_target, trajfile, validfile, valid_fraction,
                                       split_seed=rand_seed, split_seed_dict=split_seed_dict)

                    else:
                        raise Exception('# Empty Trajectory file(s) found: ',
                                        [f for f in trajnames if os.path.getsize(f) == 0])
                else:
                    raise Exception('# Missing Trajectory files: ',
                                    [f for f in trajnames if not os.path.isfile(f)])
                # Store the filename lists for later use in constructing heads dictionaries
                if prefix == "":
                    trajfile_dict[targetstr] = trajfile
                if prefix == 'valid' or not separate_valid:
                    validfile_dict[targetstr] = validfile

        if self.reset_loss:
            if hasattr(self.wrapper, "reset_loss"):
                self.wrapper.reset_loss(seed=self.seed, target=self.target,
                                        prefix=self.calc_prefix, suffix=self.calc_suffix,)
            else:
                raise Exception("# Error: reset_loss == True, yet wrapper has no reset_loss function")
        # Train the ML calculator using this training data
        calc = self.wrapper.train(seed=self.seed, trajfile=trajfile_dict, validfile=validfile_dict, targets_in=self.target,
                                  prefix=self.calc_prefix, suffix=self.calc_suffix, dir_suffix=self.calc_dir_suffix,
                                  restart=self.restart, **self.train_params)
        return calc

    def make_parser(self):

        import argparse

        # Parse command line values
        main_help = ('ML_Training.py: Train a ML-based Calculator from QMD trajectory files.')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help, epilog=epi_help)
        parser.add_argument('--seed', '-s', type=str, help="Base name of the system the calculator is trained for, usually '{solu}_{solv}' (or 'all_{solv}' etc for a calculator covering several solutes). Used to name the training directory, the trajectory files and the calculator files.")
        parser.add_argument('--calc_suffix', '-S', default="", type=str, help="Suffix naming this calculator (eg 'MACEac0u'), often encoding the ML method, generation and committee member. Used in the names of the merged training trajectories and of the calculator files.")
        parser.add_argument('--calc_dir_suffix', '-D', default=None, type=str, help="Suffix for the directory in which the calculator is trained, named '{seed}_{targ}_{calc_dir_suffix}' where {targ} is eg 'gs' or 'es1'. If None, calc_suffix is used; giving several calculators (eg committee members) the same calc_dir_suffix makes them share a directory.")
        parser.add_argument('--calc_prefix', '-P', default="", type=str, help="Prefix prepended to the names of the training trajectory files and passed to the wrapper. If empty, the driver sets it to the training directory name plus '/'.")
        parser.add_argument('--target', '-t', default=0, type=int, help="Electronic state the calculator is trained for: 0 for the ground state, n for excited state n. For a multi-head calculator, a dict mapping state indices to state names (eg {0: 'gs', 1: 'es1'}), optionally with a 'diff' entry such as 'es1_m_gs' to also train on the energy difference between two states.")
        parser.add_argument('--traj_prefix', '-Q', default='training', type=str, help='Prefix for the trajectory on which to train the calculator. Currently not used by the training task.')
        parser.add_argument('--traj_suffix', '-T', default='training', type=str, help="Suffix of the training trajectory files, which are named '{seed}_{targ}_{label}_{traj_suffix}.traj' (one per trajectory label A, B, ...) in the training directory. The atom-energies trajectory '{seed}_atoms_{traj_suffix}.traj' is also looked for if the wrapper needs one.")
        parser.add_argument('--geom_prefix', default='gs_PBE0/is_opt_{solv}', nargs='?', type=str, help="Directory, relative to the base directory, from which the geometry '{solu}_{solv}.xyz' (or '{solu}.xyz' with no solvent) is copied into the training directory. May contain '{solu}' and '{solv}', which are replaced by the solute and solvent names. If None, the base directory is used.")
        parser.add_argument('--ntraj', '-n', default=1, type=int, help='Total number of trajectories (labelled A, B, C, ...) available. If which_trajs is None, all of them are used for training.')
        parser.add_argument('--restart', '-r', default=False, nargs='?', const=True, type=bool, help='If True, ask the wrapper to load an existing calculator and continue training it rather than starting afresh.')
        parser.add_argument('--reset_loss', '-R', default=False, nargs='?', const=True, type=bool, help='If True, reset the loss function of an existing calculator before training (eg because new training data has been added). Requires a wrapper with a reset_loss method.')
        parser.add_argument('--which_trajs', '-w', default=None, type=str, help="List of trajectory labels (eg ['A', 'B', 'C']) to train on. If None, all ntraj trajectories are used with the state in target; otherwise which_targets must give the state of each trajectory.")
        parser.add_argument('--which_trajs_valid', '-v', default=None, type=str, help='List of trajectory labels to use as a separate validation set. If None, validation data is instead drawn at random from the training trajectories (see valid_fraction).')
        parser.add_argument('--which_trajs_test', '-u', default=None, type=str, help='List of trajectory labels to use as a test set. Currently only used when assembling tasks with esteem.active_learning; the training task itself does not use it.')
        parser.add_argument('--traj_links', '-L', default=None, type=dict, help="Dict mapping each training trajectory label to the path of the trajectory file to use for it (or a list of paths), relative to the base directory. Symbolic links to these files are created in the training directory; paths may contain '{solu}' and '{solv}'.")
        parser.add_argument('--traj_links_valid', '-V', default=None, type=dict, help='Dict mapping each validation trajectory label (from which_trajs_valid) to the path of the trajectory file to link to it, relative to the base directory. None if there is no separate validation set.')
        parser.add_argument('--traj_links_test', '-U', default=None, type=dict, help='Dict mapping each test trajectory label to the path of the trajectory file to link to it, relative to the base directory. None if there is no test set.')
        parser.add_argument('--cutoff', '-d', default=6.5, type=float, help="Descriptor cutoff radius in Angstrom. Currently not used: set cutoffs through the wrapper's training arguments (eg via train_params) instead.")
        parser.add_argument('--valid_fraction', '-W', default=0.1, type=float, help='Fraction of the training frames held back at random as validation data, when which_trajs_valid is None.')

        return parser

    def validate_args(self, args):
        default_args = self.make_parser().parse_args(['--seed', 'a'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Command-line driver
def get_parser():
    mltrain = MLTrainingTask()
    return mltrain.make_parser()


if __name__ == '__main__':

    from esteem import wrappers
    mltrain = MLTrainingTask()

    # Parse command line values
    args = mltrain.make_parser().parse_args()
    for arg in vars(args):
        setattr(mltrain, arg, getattr(args, arg))
    print('#', args)
    mltrain.wrapper = wrappers.amp.AMPWrapper()

    # Run main program
    mltrain.run()
