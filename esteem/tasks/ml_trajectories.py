#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines a task to use a Machine Learning calculator to generate
Molecular Dynamics trajectories, and also to process the results
to generate infrared spectra from trajectory data"""


# # Main Routine

# In[ ]:


# Load essential modules
import sys
import os
import string
from ase.io.trajectory import Trajectory
from esteem.trajectories import generate_md_trajectory, find_initial_geometry, get_trajectory_list, targstr

class MLTrajTask:

    """
    Defines a task to use a Machine Learning calculator to generate
    Molecular Dynamics trajectories
    
    The main routine is :meth:`esteem.tasks.ml_trajectories.run`
    """

    def __init__(self,**kwargs):
        self.wrapper = None
        self.snap_wrapper = None
        self.snap_calc_params = None
        self.script_settings = None
        self.task_command = 'mltraj'
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self,arg,getattr(args,arg))

    # Main routine
    def run(self):
        """Main routine for the ML_Trajectories task"""

        # Check input args are valid
        #validate_args(self)

        # Make sure trajectory choices are valid
        all_trajs = get_trajectory_list(self.ntraj)
        if self.which_trajs is None:
            which_trajs = all_trajs
        else:
            which_trajs = self.which_trajs
            for traj_label in which_trajs:
                if traj_label not in all_trajs:
                    raise Exception(f"Invalid trajectory name: {traj_label}")

        # Set up calculator parameters dict
        if self.calc_seed is None:
            self.calc_seed = self.seed
        calc_params = {'calc_seed': self.calc_seed,
                       'calc_suffix': self.calc_suffix,
                       'calc_dir_suffix': self.calc_dir_suffix,
                       'calc_prefix': f'../../{self.calc_prefix}', # MD will be run from subdirectory
                       'target': self.target}
        if self.calc_seed is not None:
            calc_params['calc_seed'] = self.calc_seed

        model = {}
        continuation_len = {}
        for traj_label in which_trajs:

            # Find (or relax) initial geometry
            calc_params['calc_prefix'] = f'../{self.calc_prefix}'

            model[traj_label] = None
            if self.continuation:
                continuation_trajfile = f"{self.seed}_{targstr(self.target)}_{traj_label}_{self.traj_suffix}.traj"
                if os.path.exists(continuation_trajfile):
                    continuation_traj = Trajectory(continuation_trajfile)
                    continuation_len[traj_label] = len(continuation_traj)
                    model[traj_label] = continuation_traj[-1]
                    print(f'# Continuation file {continuation_trajfile} found, containing {continuation_len[traj_label]} snapshots. Continuing.')
                else:
                    self.continuation = False
                    continuation_len[traj_label] = 0
                    print('# No continuation file found, initialising from scratch')
            else:
                continuation_len[traj_label] = 0

            if model[traj_label] is None:
                model[traj_label] = find_initial_geometry(self.seed,self.wrapper.geom_opt,
                                                      calc_params,traj_label)                
            if self.constraints is not None:
                from ase.constraints import FixBondLengths, Hookean, FixInternals
                set_constraints = []
                for c in self.constraints:
                    if isinstance(c, FixBondLengths):
                        bondlength = c.bondlengths[0]
                        atoms = c.pairs[0]
                        model[traj_label].set_distance(atoms[0],atoms[1],bondlength,fix=0)
                        model[traj_label].set_constraint(c)
                    if isinstance(c, Hookean):
                        del model[traj_label].constraints
                        set_constraints.append(c)
                    if isinstance(c, FixInternals):
                        set_constraints.append(c)
                model[traj_label].set_constraint(set_constraints)

        # Evaluate initial energies
        for i in which_trajs:
            if continuation_len[traj_label]>=self.nsnap:
                continue
            ep = self.wrapper.singlepoint(model[i],"test",calc_params)[0]
            ek = model[i].get_kinetic_energy()
            print(f'# Trajectory {i} initial potential energy = {ep-self.wrapper.atom_e}')
            print(f'# Trajectory {i} initial   kinetic energy = {ek}')

        # Current path will be two levels down from starting path in MD run, so adjust prefix
        calc_params['calc_prefix'] = f'../../{self.calc_prefix}'
        for traj_label in which_trajs:
            # Pass in routine to actually run MD into generic Snapshot MD driver
            generate_md_trajectory(model[traj_label],self.seed,self.target,traj_label,self.traj_suffix,
                                   wrapper=self.wrapper,count_snaps=self.nsnap,count_equil=self.nequil,
                                   md_steps=self.md_steps,md_timestep=self.md_timestep,
                                   md_friction=self.md_friction,store_full_traj=self.store_full_traj,
                                   temp=self.temp,calc_params=calc_params,dynamics=self.dynamics,
                                   snap_wrapper=self.snap_wrapper if not self.recalculate_carved_traj else None,
                                   snap_calc_params=self.snap_calc_params if not self.recalculate_carved_traj else None,
                                   continuation=self.continuation,debugger=self.debugger)


    def make_parser(self):
        
        "Makes a parser to setup input variables"

        import argparse
        from ase.units import AUT

        main_help = ('ML_Trajectories.py: Generate trajectory files using a pre-trained ML-based calculator.\n')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--seed','-s',type=str,help='Base name stem for the calculation (often the name of the molecule)')
        parser.add_argument('--traj_suffix','-S',default="mldyn",type=str,help='Suffix for the trajectory files to be generated')
        parser.add_argument('--calc_seed','-Z',default=None,type=str,help='Seed for the calculator')
        parser.add_argument('--calc_suffix','-C',default='',type=str,help='Suffix for the calculator (often specifies ML hyperparameters)')
        parser.add_argument('--calc_dir_suffix','-D',default=None,type=str,help='Prefix for the calculator (often specifies directory)')
        parser.add_argument('--calc_prefix','-P',default='',type=str,help='Prefix for the calculator (often specifies directory)')
        parser.add_argument('--target','-t',default=0,type=int,help='Excited state index, zero for ground state')
        parser.add_argument('--md_timestep','-q',default=10*AUT,type=float,help='Timestep in ASE units')
        parser.add_argument('--md_friction','-r',default=0.002,type=float,help='Langevin friction coefficient')
        parser.add_argument('--md_steps','-Q',default=100,type=int,help='Number of MLMD steps between each snapshot')
        parser.add_argument('--md_init_traj_link','-I',default=None,type=str,help='Path to initial MD trajectory file, relative to base directory. May contain {{solu}} and {{solv}} for substitution by solute and solvent names respectively.')
        parser.add_argument('--geom_prefix','-G',default='',type=str,help='String to append to filenames for initial geometries. May contain {{solu}} and {{solv}} for substitution by solute and solvent names respectively.')
        parser.add_argument('--continuation','-o',default=False,type=bool,help='Whether to continue an existing run')
        parser.add_argument('--debugger','-g',default=None,help='Debugger class to run on trajectory after each snapshot, as a sanity check')
        parser.add_argument('--store_full_traj','-f',default=False,type=bool,help='Store full step-by-step trajectory rather than just snapshots every md_steps')
        parser.add_argument('--freq','-F',default=False,type=bool,help='Post-process trajectory into IR spectrum')
        parser.add_argument('--temp','-T',default=300.0,type=float,help='Temperature for thermostat')
        parser.add_argument('--ntraj','-n',default=1,type=int,help='Number of separate trajectories in full ensemble')
        parser.add_argument('--nsnap','-N',default=200,type=int,help='Number of snapshots to record in trajectory')
        parser.add_argument('--nequil','-e',default=10,type=int,help='Number of discarded equilibration snapshots before data is recorded')
        parser.add_argument('--which_trajs','-w',default=None,type=str,help='Which of the separate trajectories are to be run in this task')
        parser.add_argument('--carve_trajectory_radius','-R',default=None,type=float,help='Radius around solute to carve trajectory at')
        parser.add_argument('--recalculate_carved_traj','-J',default=False,type=bool,help='Use snap_wrapper (if present, wrapper if not) to recalculate energies and forces after carving')
        parser.add_argument('--constraints','-c',default=None,type=str,help='Constraints (ASE constraints class)')
        parser.add_argument('--dynamics','-d',default=None,type=str,help='Dynamics (ASE Dynamics class)')

        return parser

    def validate_args(args):
        default_args = make_parser().parse_args(['--seed','a'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# In[ ]:


def load_trajectory_dipole(seed_state_str,traj_suffix,ntraj,nsnaps,mdsteps,extension='.traj'):
    '''
    Loads a set of saved trajectory files and extracts the dipole moment as a 
    function of time
    '''

    from ase.io import read, Trajectory
    from esteem.trajectories import get_trajectory_list
    import numpy as np
    
    # Storage for result
    mu_t = np.zeros((ntraj,nsnaps*mdsteps,3))
    # Names of trajectories
    chars = get_trajectory_list(ntraj)
    trajname = chars[0]
    # Loop over trajectories, reading them in and storing dipole moments in an array
    for i,trajname in enumerate(chars):
        k=0
        for j in range(0,nsnaps):
            if nsnaps==0:
                file = f'{seed_state_str}_{trajname}_{traj_suffix}{extension}'
            else:
                file = f'{seed_state_str}_{trajname}_{traj_suffix}{j:04}{extension}'
            traj = Trajectory(file)
            print(file) # progress update
            for f in traj[1:]:
                mu_t[i,k] = f.get_dipole_moment()
                k = k + 1

    return mu_t

def calculate_ir_spectrum(mu_t,dt,freq_scale_fac,sigma):
    '''
    Processes the dipole moment as a function of time for a collection of trajectories,
    to calculate IR absorption spectrum
    '''

    import numpy as np
    
    # Take gradient of mu(t) to get dmu/dt
    mu_dot = np.gradient(mu_t,dt,axis=(1,))

    # Take FFT of dmu/dt and get corresponding frequencies (scaled, eg to cm^-1)
    mu_dot_tilde = np.fft.fftn(mu_dot,axes=(1,))
    omega = np.fft.fftfreq(len(mu_dot[0]),dt)*freq_scale_fac

    # Average over snapshots and take dot product with self to get autocorrelation
    mu_dot_tilde_av = np.average(mu_dot_tilde,axis=0)
    mu_dot_tilde_av_mag = (np.sum(mu_dot_tilde_av*np.conj(mu_dot_tilde_av),axis=1))

    # Convolve with Gaussian of width sigma (in cm^-1)
    gaussian = np.exp(-(omega/sigma)**2/2)
    mu_dot_tilde_av_mag_conv = np.convolve(mu_dot_tilde_av_mag, gaussian, mode="full")
    
    return mu_dot_tilde_av_mag_conv, omega


# # Command-line driver

# In[ ]:


def get_parser():
    mltraj = MLTrajTask()
    return mltraj.make_parser()

# Not much need for this now as wrapper will need defining for this to be useful
if __name__ == '__main__':

    mltraj = MLTrajTask()

    # Parse command line values
    args = mltraj.make_parser.parse_args()
    for arg in vars(args):
        setattr(mltraj,arg,getattr(args,arg))
    print('#',args)

    # Run main routine
    mltraj.run()

