#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines a task to train a Machine Learning calculator on a trajectory of snapshots
by calling the train() function of the MLWrapper"""


# # Main Routine

# In[ ]:


from esteem.trajectories import merge_traj, diff_traj, get_trajectory_list, targstr

import sys
import os
import string
from shutil import copyfile

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
        else:
            which_trajs = self.which_trajs
        if which_trajs is None:
            which_trajs = all_trajs
        else:
            for trajname in which_trajs:
                if trajname not in all_trajs:
                    raise Exception("Invalid trajectory name: ",trajname)
        trajstem = self.wrapper.calc_filename(self.seed,self.target,prefix=self.calc_prefix,suffix="")
        trajnames = [f'{trajstem}{s}_{self.traj_suffix}.traj' for s in which_trajs]

        return which_trajs,trajnames

    # Main routine
    def run(self):
        """Main routine for the ML_Training task"""

        # Check input args are valid
        #validate_args(args)

        trajfn = self.wrapper.calc_filename(self.seed,self.target,prefix=self.calc_prefix,suffix=self.traj_suffix)

        # If we need an atom trajectory, copy it from traj_suffix to calc_suffix:
        if hasattr(self.wrapper,'atom_energies'):
            atom_traj_file = f'{self.seed}_atoms_{self.traj_suffix}.traj'
            if os.path.isfile(atom_traj_file):
                atom_calc_file = f'{self.seed}_atoms_{self.calc_suffix}.traj'
                print(f'# Copying from {atom_traj_file} to {atom_calc_file} for atom energies')
                copyfile(atom_traj_file,atom_calc_file)
            else:
                raise Exception(f'# Trajectory file {atom_traj_file} not found for atom energies')

        # If we are training on energy differences, calculate these now
        prefs = [""]
        if self.which_trajs_valid is not None:
            prefs = ["","valid"]
        for prefix in prefs:
            if False: #'diff' in self.target:
                which_trajs, trajnames = self.get_trajnames(prefix)
                itarget = 0
                jtarget = 1
                for traj in trajnames:
                    itraj = traj.replace("diff",targstr(itarget))
                    jtraj = traj.replace("diff",targstr(jtarget))
                    print('# Calling diff_traj with {itraj} {jtraj} {traj}')
                    diff_traj(itraj,jtraj,traj)

            # If all trajectories exist, merge them
            which_trajs, trajnames = self.get_trajnames(prefix)
            print(f'# Trajectories to merge: {trajnames}',flush=True)
            if all([os.path.isfile(f) for f in trajnames]):
                if all([os.path.getsize(f) > 0 for f in trajnames]):
                    if prefix=="":
                        trajfile = f'{trajfn}_{prefix}merged_{self.calc_suffix}.traj'
                        merge_traj(trajnames,trajfile)
                    if prefix=="valid":
                        validfile = f'{trajfn}_{prefix}merged_{self.calc_suffix}.traj'
                        merge_traj(trajnames,validfile)
                    else:
                        validfile=None

                else:
                    raise Exception('# Empty Trajectory file(s) found: ',
                                    [f for f in trajnames if os.path.getsize(f)==0])
            else:
                raise Exception('# Missing Trajectory files: ',
                                [f for f in trajnames if not os.path.isfile(f)])
            
        if self.reset_loss:
            if hasattr(self.wrapper,"reset_loss"):
                self.wrapper.reset_loss(seed=self.seed,target=self.target,
                                        prefix=self.calc_prefix,suffix=self.calc_suffix,)
            else:
                raise Exception("# Error: reset_loss == True, yet wrapper has no reset_loss function")
        # Train the ML calculator using this training data
        calc = self.wrapper.train(seed=self.seed,trajfile=trajfile,validfile=validfile,target=self.target,
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
        '''
        parser.add_argument('--cores','-c',default=1,type=int,help='Number of parallel cores on which to run the training')
        parser.add_argument('--steps','-A',default=None,type=int,help='Annealer steps')
        parser.add_argument('--Tmax','-u',default=800.0,type=float,help='Annealer starting temperature')
        parser.add_argument('--Tmin','-v',default=0.01,type=float,help='Annealer final temperature')
        parser.add_argument('--energy_rmse','-E',default=0.02,type=float,help='RMS Energy deviation for convergence')
        parser.add_argument('--force_rmse','-F',default=0.02,type=float,help='RMS Force deviation for convergence')
        parser.add_argument('--energy_maxresid','-G',default=None,type=float,help='Maximum energy deviation for convergence')
        parser.add_argument('--force_maxresid','-H',default=None,type=float,help='Maximum force deviation for convergence')
        parser.add_argument('--hiddenlayers','-L',default=(10,10,10),nargs='*',type=int,help='Hidden Layer structure ')
        parser.add_argument('--force_coefficient','-f',default=0.04,type=float,help='Weighting of forces (compared to energy) in training')
        parser.add_argument('--overfit','-o',default=0.00,type=float,help='Weighting of forces (compared to energy) in training')
        '''
        
        return parser

    def validate_args(args):
        default_args = make_parser().parse_args(['--seed','a'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Command-line driver

# In[ ]:


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

