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
    return traj_target[0] if isinstance(traj_target,list) else traj_target

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
        self.snap_target = None
        self.which_targets = None
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
            self.which_targets = [self.target]*len(which_trajs)
        else:
            which_trajs = self.which_trajs
            assert len(self.which_targets)==len(self.which_trajs)
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
        enough_data = {traj_label:False for traj_label in which_trajs}
        offsets = {traj_label:'' for traj_label in which_trajs}
        while not all(enough_data.values()):
            for traj_label,traj_target in zip(which_trajs,self.which_targets):

                # Find (or relax) initial geometry
                calc_params['calc_prefix'] = f'../{self.calc_prefix}'
                if isinstance(self.target,dict):
                    calc_params['calc_head'] = self.target[traj_target]

                # Check if we have a continuation to load in
                model[traj_label] = None
                if self.continuation:
                    # an unfinished run is under the incomplete name, a finished one under the final name
                    final, incomplete = md_output_names(self.seed,driving_target(traj_target),traj_label,self.traj_suffix)
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
                        elif model[traj_label].get_temperature() > 10*self.temp:
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
                    model[traj_label] = find_initial_geometry(self.seed,self.wrapper.geom_opt,
                                                          calc_params,offset_traj_label)
                # Set up any constraints
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
            for traj_label in which_trajs:
                if continuation_len[traj_label]>=self.nsnap or enough_data[traj_label]:
                    continue
                ep = self.wrapper.singlepoint(model[traj_label],"test",calc_params)[0]
                ek = model[traj_label].get_kinetic_energy()
                print(f'# Trajectory {traj_label} initial potential energy = {ep-self.wrapper.atom_e}')
                print(f'# Trajectory {traj_label} initial   kinetic energy = {ek}')

            # Current path will be two levels down from starting path in MD run, so adjust prefix
            calc_params['calc_prefix'] = f'../../{self.calc_prefix}'
            for traj_label,traj_target in zip(which_trajs,self.which_targets):
                if enough_data[traj_label]:
                    continue
                # Pass in routine to actually run MD into generic Snapshot MD driver
                status = generate_md_trajectory(model[traj_label],self.seed,traj_target,traj_label,self.traj_suffix,
                                       traj_wrapper=self.wrapper,count_snaps=self.nsnap,count_equil=self.nequil,
                                       md_steps=self.md_steps,md_timestep=self.md_timestep,
                                       md_friction=self.md_friction,store_full_traj=self.store_full_traj,
                                       temp=self.temp,traj_calc_params=calc_params,dynamics=self.dynamics,
                                       snap_target=self.snap_target if not self.recalculate_carved_traj else None,
                                       snap_wrapper=self.snap_wrapper if not self.recalculate_carved_traj else None,
                                       snap_calc_params=self.snap_calc_params if not self.recalculate_carved_traj else None,
                                       continuation=self.continuation,debugger=self.debugger,
                                       stop_stde_thresh=self.md_stop_stde_thresh)
                self.continuation = True
                final, incomplete = md_output_names(self.seed,driving_target(traj_target),traj_label,self.traj_suffix)
                output_trajfile = final if os.path.exists(final) else incomplete
                if os.path.exists(output_trajfile):
                    output_traj = Trajectory(output_trajfile)
                    continuation_len[traj_label] = len(output_traj)
                    output_traj.close()
                    print(f'# Output file {output_trajfile} now contains {continuation_len[traj_label]} of {self.nsnap} snapshots.')
                else:
                    # aborted during equilibration, before any snapshots were written
                    continuation_len[traj_label] = 0
                if continuation_len[traj_label]>=self.nsnap:
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
                        for t in (traj_target if isinstance(traj_target,list) else [traj_target]):
                            finish_md_output(self.seed,t,traj_label,self.traj_suffix)
                    else:
                        print(f'# Trajectory {traj_label} aborted: restart {len(offsets[traj_label])} of '
                              f'{self.max_restarts} from a new initial snapshot')



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
        parser.add_argument('--traj_dir_suffix','-Y',default=None,type=str,help='Prefix for the trajectory (often specifies directory)')
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
        parser.add_argument('--max_restarts',default=10,type=int,help='Number of times an aborted (eg blown-up) trajectory is restarted from a new initial snapshot before giving up')
        parser.add_argument('--md_stop_stde_thresh',default=None,type=float,help='Stop a trajectory once the committee energy standard deviation per atom (eV) exceeds this. Checked during equilibration as well as at snapshots; frames up to and including the one that crossed it are kept, and the trajectory is then treated as finished. None (default) disables the check')
        parser.add_argument('--which_trajs','-w',default=None,type=str,help='Which of the separate trajectories are to be run in this task')
        parser.add_argument('--carve_trajectory_radius','-R',default=None,type=float,help='Radius around solute to carve trajectory at')
        parser.add_argument('--carve_trajectory_max_atoms','-m',default=None,type=float,help='Max atoms when carving trajectory')
        parser.add_argument('--recalculate_carved_traj','-J',default=False,type=bool,help='Use snap_wrapper (if present, wrapper if not) to recalculate energies and forces after carving')
        parser.add_argument('--constraints','-c',default=None,type=str,help='Constraints (ASE constraints class)')
        parser.add_argument('--dynamics','-d',default=None,type=str,help='Dynamics (ASE Dynamics class)')
        parser.add_argument('--carved_suffix',default=None,type=str,help='Suffix of mltraj output files corresponding to the size of carved clusters')
        parser.add_argument('--corr_traj', default=False, type=bool, help='Boolean specifying whether to carve and recalculate correction trajectories')

        return parser

    def validate_args(args):
        default_args = make_parser().parse_args(['--seed','a'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")

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

