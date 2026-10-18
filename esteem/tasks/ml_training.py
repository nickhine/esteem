#!/usr/bin/env python
# coding: utf-8
"""Defines a task to train a Machine Learning calculator on a trajectory of snapshots
by calling the train() function of the MLWrapper"""

from esteem.trajectories import merge_traj, diff_traj, get_trajectory_list, targstr
import sys
import os
import string
from shutil import copyfile

def make_diff_trajs(target_dict,which_targets,trajnames):
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
     itrajs = [traj for traj,targ in zip(trajnames,which_targets) if targ==itarget]
     jtrajs = [traj for traj,targ in zip(trajnames,which_targets) if targ==jtarget]
     print(f'# Calling diff_traj with {itarget} {jtarget} {trajnames}')
     assert(len(itrajs)==len(jtrajs))
     # iterate over both lists simultaneously
     for itraj,jtraj in zip(itrajs,jtrajs):
         # Create new name for output trajectory by splitting on concatenation
         # of target names and substituting the last instance with "diff"
         ktraj = 'diff'.join(itraj.rsplit(f'{jtarget}{itarget}', 1))
         print(f'# Calling diff_traj with {itraj} {jtraj} {ktraj}')
         # Process trajectories to write difference trajectory
         diff_traj(itraj,jtraj,ktraj)
         # Store the name of the difference trajectory for later use
         trajnames_diff.append(ktraj)
     return trajnames_diff

class MLTrainingTask:

    def __init__(self,**kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'mltrain'
        self.train_params = {}
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self,arg,getattr(args,arg))
            
    def get_trajnames(self,prefix=""):
        all_trajs = get_trajectory_list(self.ntraj)
        if 'valid' in prefix:
            which_trajs = self.which_trajs_valid
            which_targets = self.which_targets_valid
        else:
            which_trajs = self.which_trajs
            which_targets = self.which_targets
        if which_trajs is None:
            which_trajs = all_trajs
            which_targets = [self.target]*len(which_trajs)
        else:
            for trajname in which_trajs:
                if trajname not in all_trajs:
                    raise Exception("Invalid trajectory name: ",trajname)
        trajstem = self.wrapper.calc_filename(self.seed,self.target,prefix=self.calc_prefix,suffix="")
        trajnames = [f'{trajstem}{s}_{self.traj_suffix}.traj' for s in which_trajs]

        return which_trajs,which_targets,trajnames

    # Main routine
    def run(self):
        """Main routine for the ML_Training task"""

        # Check input args are valid
        #validate_args(args)

        # Get name of folder where calculator files live
        dirname = self.wrapper.calc_filename(self.seed,self.target,prefix="",suffix=self.calc_dir_suffix)

        # If we need an atom trajectory, copy it from traj_suffix to calc_suffix:
        if hasattr(self.wrapper,'atom_energies'):
            atom_traj_file = f'{self.seed}_atoms_{self.traj_suffix}.traj'
            if os.path.isfile(atom_traj_file):
                atom_calc_file = f'{dirname}/{self.seed}_atoms_{self.calc_suffix}.traj'
                if not os.path.isfile(atom_calc_file):
                    print(f'# Copying from {atom_traj_file} to {atom_calc_file} for atom energies')
                    copyfile(atom_traj_file,atom_calc_file)
                else:
                    print(f'# Using {atom_calc_file} for atom energies')
            else:
                raise Exception(f'# Trajectory file {atom_traj_file} not found for atom energies')

        # If we are training on energy differences, calculate these now
        prefs = [""]
        if self.which_trajs_valid is not None:
            prefs = ["","valid"]
            separate_valid = True
        else:
            separate_valid = False
            valid_fraction = self.valid_fraction
            if hasattr(self,'rand_seed'):
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
            if not isinstance(self.target,dict):
                target_dict = {self.target:str(self.target)}

            # Special handling for difference trajectories
            if 'diff' in target_dict:
                trajnames_diff = make_diff_trajs(target_dict,which_targets,trajnames)
            # Iterate over all targets
            for target in target_dict:
                targetstr = target_dict[target]
                # Special handling for difference trajectories
                if target == 'diff':
                    trajnames_target = trajnames_diff
                else:
                    if isinstance(self.target,dict):
                        # Create a list of the trajectories for this target
                        trajnames_target = [traj for traj,targ in zip(trajnames,which_targets) if targ==targetstr]
                    else:
                        # If only one target, use all trajectories
                        trajnames_target = trajnames
                # Construct the base of the filename for the merged trajectory
                trajfn = self.wrapper.calc_filename(self.seed,target,prefix=self.calc_prefix,suffix=self.traj_suffix)
                print(f'# Trajectories to merge: {trajnames_target} for target {targetstr}',flush=True)
                # Check that all the trajectories exist and are non-empty
                if all([os.path.isfile(f) for f in trajnames]):
                    if all([os.path.getsize(f) > 0 for f in trajnames]):
                        if separate_valid:
                            if prefix=="":
                                # Merge the trajectories for training
                                trajfile = f'{trajfn}_{prefix}merged_{self.calc_suffix}.traj'
                                merge_traj(trajnames_target,trajfile)
                            if prefix=="valid":
                                # Merge the trajectories for validation
                                validfile = f'{trajfn}_{prefix}merged_{self.calc_suffix}.traj'
                                merge_traj(trajnames_target,validfile)
                            else:
                                validfile=None
                        else:
                            # Merge the trajectories for training and validation (splitting as required)
                            pref = prefix
                            trajfile = f'{trajfn}_{pref}merged_{self.calc_suffix}.traj'
                            pref = 'valid'
                            validfile = f'{trajfn}_{pref}merged_{self.calc_suffix}.traj'
                            merge_traj(trajnames_target,trajfile,validfile,valid_fraction,
                                       split_seed=rand_seed,split_seed_dict=split_seed_dict)

                    else:
                        raise Exception('# Empty Trajectory file(s) found: ',
                                        [f for f in trajnames if os.path.getsize(f)==0])
                else:
                    raise Exception('# Missing Trajectory files: ',
                                    [f for f in trajnames if not os.path.isfile(f)])
                # Store the filename lists for later use in constructing heads dictionaries
                if prefix=="":
                    trajfile_dict[targetstr] = trajfile
                if prefix=='valid' or not separate_valid:
                    validfile_dict[targetstr] = validfile
            
        if self.reset_loss:
            if hasattr(self.wrapper,"reset_loss"):
                self.wrapper.reset_loss(seed=self.seed,target=self.target,
                                        prefix=self.calc_prefix,suffix=self.calc_suffix,)
            else:
                raise Exception("# Error: reset_loss == True, yet wrapper has no reset_loss function")
        # Train the ML calculator using this training data
        calc = self.wrapper.train(seed=self.seed,trajfile=trajfile_dict,validfile=validfile_dict,targets_in=self.target,
                                  prefix=self.calc_prefix,suffix=self.calc_suffix,dir_suffix=self.calc_dir_suffix,
                                  restart=self.restart,**self.train_params)
        return calc

    def make_parser(self):

        import argparse

        # Parse command line values
        main_help = ('ML_Training.py: Train a ML-based Calculator from QMD trajectory files.')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--seed','-s',type=str,help='Base name stem for the calculation (often the name of the molecule)')
        parser.add_argument('--calc_suffix','-S',default="",type=str,help='Suffix for the calculator')
        parser.add_argument('--calc_dir_suffix','-D',default=None,type=str,help='Suffix for the calculator directory ')
        parser.add_argument('--calc_prefix','-P',default="",type=str,help='Prefix for the calculator (often specifies directory)')
        parser.add_argument('--target','-t',default=0,type=int,help='Excited state index, zero for ground state')
        parser.add_argument('--traj_prefix','-Q',default='training',type=str,help='Prefix for the trajectory on which to train the calculator')
        parser.add_argument('--traj_suffix','-T',default='training',type=str,help='Suffix for the trajectory on which to train the calculator')
        parser.add_argument('--geom_prefix',default='gs_PBE0/is_opt_{solv}',nargs='?',type=str,help='Prefix for the path at which to find the input geometry')
        parser.add_argument('--ntraj','-n',default=1,type=int,help='How many total trajectories (A,B,C...) with this naming are present for training')
        parser.add_argument('--restart','-r',default=False,nargs='?',const=True,type=bool,help='Whether to load a pre-existing calculator and resume training')
        parser.add_argument('--reset_loss','-R',default=False,nargs='?',const=True,type=bool,help='Whether to reset the loss function due to new training data being added')
        parser.add_argument('--which_trajs','-w',default=None,type=str,help='Which trajectories (A,B,C...) with this naming are to be trained against')
        parser.add_argument('--which_trajs_valid','-v',default=None,type=str,help='Which trajectories (A,B,C...) with this naming are to be validated against')
        parser.add_argument('--which_trajs_test','-u',default=None,type=str,help='Which trajectories (A,B,C...) with this naming are to be tested against')
        parser.add_argument('--traj_links','-L',default=None,type=dict,help='Targets for links to create for training trajectories')
        parser.add_argument('--traj_links_valid','-V',default=None,type=dict,help='Targets for links to create for validation trajectories')
        parser.add_argument('--traj_links_test','-U',default=None,type=dict,help='Targets for links to create for testing trajectories')
        parser.add_argument('--cutoff','-d',default=6.5,type=float,help='Gaussian descriptor cutoff')
        parser.add_argument('--valid_fraction','-W',default=0.1,type=float,help='Fraction of trajectory to use as validation data')
        
        return parser

    def validate_args(args):
        default_args = make_parser().parse_args(['--seed','a'])
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
        setattr(mltrain,arg,getattr(args,arg))
    print('#',args)
    mltrain.wrapper = wrappers.amp.AMPWrapper()

    # Run main program
    mltrain.run()

