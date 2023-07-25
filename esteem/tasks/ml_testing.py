#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines a task to test a Machine Learning calculator by comparing its results
to those of an existing trajectory or set of trajectories"""


# # Main Routine

# In[ ]:


# Load essential modules
import sys
import os
import string
from ase.io.trajectory import Trajectory

from esteem.trajectories import merge_traj, get_trajectory_list

class MLTestingTask:

    def __init__(self,**kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'mltest'
        self.train_params = {}
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self,arg,getattr(args,arg))
            
    def get_trajnames(self):
        all_trajs = get_trajectory_list(self.ntraj)
        if self.which_trajs is None:
            which_trajs = all_trajs
        else:
            which_trajs = self.which_trajs
            if isinstance(which_trajs,dict):
                which_trajs_dict = which_trajs
                which_trajs = []
                for w in which_trajs_dict:
                    for v in which_trajs_dict[w]:
                        which_trajs.append(v)
            for trajname in which_trajs:
                if trajname not in all_trajs:
                    raise Exception("Invalid trajectory name: ",trajname)
        trajstem = self.wrapper.calc_filename(self.seed,self.target,prefix=self.calc_prefix,suffix="")
        trajnames = [trajstem + s + '_'+self.traj_suffix+'.traj' 
             for s in which_trajs]

        return which_trajs,trajnames

    # Main routine
    def run(self): 
        
        """Main routine for the ML_Testing task"""
        
        from os.path import commonprefix

        # Check input args are valid
        #validate_args(self)

        # Get strings for trajectory names
        if self.calc_seed is None:
            self.calc_seed = self.seed
        traj_prefix = self.traj_prefix
        # Assume path is relative to base directory, unless blank
        if traj_prefix!="":
            traj_prefix = "../"+traj_prefix
        
        trajfn = self.wrapper.calc_filename(self.calc_seed,self.target,prefix="",suffix=self.traj_suffix)
        trajstem = self.wrapper.calc_filename(self.calc_seed,self.target,prefix=traj_prefix,suffix="")

        all_trajs = get_trajectory_list(self.ntraj)
        traj_suffix = self.traj_suffix
        if self.which_trajs is None:
            which_trajs = all_trajs
        else:
            which_trajs = self.which_trajs
            if "_" in which_trajs:
                traj_suffix = which_trajs.split("_",1)[1]
                which_trajs = which_trajs.split("_",1)[0]
            #for trajname in which_trajs:
            #    if trajname not in all_trajs:
            #        raise Exception("Invalid trajectory name:",trajname)

        # If all trajectories exist, test against them
        trajnames = [trajstem + s + '_' + traj_suffix + '.traj'
                     for s in which_trajs]
        print('# Merging trajectories: ',trajnames)
        if not all([os.path.isfile(f) for f in trajnames]):
            raise Exception('# Missing Trajectory file(s): ',
                            [f for f in trajnames if not os.path.isfile(f)])
        if not all([os.path.getsize(f) > 0 for f in trajnames]):
            raise Exception('# Empty Trajectory file(s) found: ',
                            [f for f in trajnames if os.path.getsize(f)==0])

        # Test the ML calculator against the results in the merged trajectory
        which_traj_str = ''.join(which_trajs)
        if len(which_traj_str) > 20:
            which_traj_str = which_traj_str[0:20]

        if isinstance(self.calc_suffix,dict):
            calc_suffix = commonprefix(list(self.calc_suffix.keys()))
        else:
            calc_suffix = self.calc_suffix
        #test_dir = f'{seed_state_str}_{calc_suffix}_test'
        intrajfile = trajfn+"_"+calc_suffix+"_"+which_traj_str+'_merged.traj'
        if os.path.isfile(intrajfile):
            print(f'# Merged input trajectory file {intrajfile} already exists. Overwriting!')
        else:
            print(f'# Writing merged input trajectory file {intrajfile}')
        merge_traj(trajnames,intrajfile)
        intraj = Trajectory(intrajfile)
        # Load the calculator
        print("# Loading Calculator")
        calc_params = {'calc_seed': self.calc_seed,
                       'calc_suffix': self.calc_suffix,
                       'calc_dir_suffix': self.calc_dir_suffix,
                       'calc_prefix': f'../{self.calc_prefix}', # Testing will be run from subdirectory
                       'target': self.target}
        if hasattr(self.wrapper,'update_atom_e'):
            self.wrapper.update_atom_e = True
        output_traj = self.output_traj
        if output_traj is None:
            output_traj = calc_suffix + "_"+which_traj_str+"_test"
        outtrajfile = trajfn+"_"+output_traj+'.traj'
        if os.path.isfile(outtrajfile):
            print(f'# Warning: output trajectory file {outtrajfile} already exists. Overwriting!')
        else:
            print(f'# Writing output trajectory file {outtrajfile}')
        compare_wrapper_to_traj(self.wrapper,calc_params,intraj,outtrajfile)
        
        # Open temporary trajectory for reading
        outtraj = Trajectory(outtrajfile)
        if self.ref_mol_dir is not None:
            intrajfile = trajfn+"_"+calc_suffix+"_"+which_traj_str+'_refsub.traj'
            self.subtract_reference_energies(intraj,intrajfile)
            intraj.close()
            intraj = Trajectory(intrajfile)
            outtrajfile = trajfn+"_"+output_traj+'_refsub.traj'
            self.subtract_reference_energies(outtraj,outtrajfile)
            outtraj.close()
            outtraj = Trajectory(outtrajfile)
            
        # Finally, plot comparison
        clabel = 'RMS Force component deviation (eV/Ang)'
        xlabel = 'Trajectory Energy (eV)'
        ylabel = 'Calculator Energy (eV)'
        compare_traj_to_traj(intraj,outtraj,self.plotfunc,self.plotfile,xlabel,ylabel,clabel)

        if self.cleanup: # optional cleanup
            os.remove(intrajfile)
            os.remove(outtrajfile)

    def subtract_reference_energies(self,trajin,trajout_file):
        """Subtract reference energies from a trajectory to just get energy above reference zero"""
        from esteem.drivers import get_solu_solv_names
        from esteem.trajectories import targstr
        from esteem.tasks.clusters import get_ref_mol_energy

        ref_solu, ref_solv = get_solu_solv_names(self.seed)
        if ref_solv=="NO_SOLVENT_FOUND":
            ref_solv = None
        trajout = Trajectory(trajout_file,'w')
        targ = 0
        if targ==0:
                ref_solu_t = ref_solu
        else:
            ref_solu_t = f'{ref_solu}_{targstr(targ)}'
        ref_mol_dir = self.ref_mol_dir.replace("{targ}",targstr(targ))
        ref_solu_dir = f'../{ref_mol_dir}'
        ref_mol_dir = self.ref_mol_dir.replace("{targ}",targstr(0))
        ref_solv_dir = f'../{ref_mol_dir}'
        calc_params = {'calc_seed': self.calc_seed,
                       'calc_suffix': self.calc_suffix,
                       'calc_dir_suffix': self.calc_dir_suffix,
                       'calc_prefix': f'../{self.calc_prefix}',
                       'target': self.target}
        # Read in Reference E, f, p
        if ref_solv is not None:
            ref_mol_xyz = f'{ref_solv_dir}/is_opt_{ref_solv}/{ref_solv}.xyz'
            solv_energy,solv_model = get_ref_mol_energy(self.wrapper,ref_solv,ref_solv,calc_params,ref_mol_xyz,ref_solv_dir)
            if isinstance(solv_energy,np.ndarray):
                solv_energy = np.mean(solv_energy)
            print('# Solvent reference energy: ',solv_energy)
            ref_mol_xyz = f'{ref_solu_dir}/is_opt_{ref_solv}/{ref_solu}.xyz'
        else:
            ref_mol_xyz = f'{ref_solu_dir}/opt/{ref_solu}.xyz'
            solv_energy = 0.0
        solu_energy,solu_model = get_ref_mol_energy(self.wrapper,ref_solu,ref_solv,calc_params,ref_mol_xyz,ref_solu_dir)
        if isinstance(solu_energy,np.ndarray):
            solu_energy = np.mean(solu_energy)
        print('# Solute reference energy: ',solu_energy)

        for i,frame in enumerate(trajin):
            n = len(frame)-len(solu_model)
            if n>0:
                n = int(n/len(solv_model))
            e = frame.get_potential_energy()
            e = e - solu_energy - n*solv_energy
            frame.calc.results["energy"] = e
            trajout.write(frame,**frame.calc.results)
        trajout.close()

    # Generate default arguments and return as parser
    def make_parser(self):

        import argparse

        main_help = ('ML_Testing.py: Test an ML-based Calculator by comparing it to trajectories.')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--seed','-s',type=str,help='Base name stem for the calculation (often the name of the molecule)')
        parser.add_argument('--calc_seed','-Z',default=None,type=str,help='Seed name for the calculator')
        parser.add_argument('--calc_suffix','-S',default="",type=str,help='Suffix for the calculator (often specifies ML hyperparameters)')
        parser.add_argument('--calc_dir_suffix','-D',default=None,type=str,help='Suffix for the calculator (often specifies ML hyperparameters)')
        parser.add_argument('--calc_prefix','-P',default="",type=str,help='Prefix for the calculator (often specifies directory)')
        parser.add_argument('--target','-t',default=0,type=int,help='Excited state index, zero for ground state')
        parser.add_argument('--output_traj','-o',default=None,type=str,help='Filename to which to write the calculated trajectory')
        parser.add_argument('--plotfile','-p',default=None,nargs='?',const="TkAgg",type=str,help='Image file to which to write comparison plot')
        parser.add_argument('--plotfunc','-F',default=None,help='Function for plotting')
        parser.add_argument('--cleanup','-C',default=True,type=bool,help='Remove reference-corrected test and merged trajectory files after finishing')
        parser.add_argument('--traj_prefix','-Q',default="",type=str,help='Prefix for the trajectory files being tested')
        parser.add_argument('--traj_suffix','-T',default="training",type=str,help='Suffix for the trajectory files being tested')
        parser.add_argument('--ntraj','-n',default=1,type=int,help='How many total trajectories (A,B,C...) with this naming are present')
        parser.add_argument('--which_trajs','-w',default=None,type=str,help='Which trajectories (A,B,C...) with this naming are to be trained against')
        parser.add_argument('--which_trajs_valid','-v',default=None,type=str,help='Which trajectories (A,B,C...) with this naming are to be validated against')
        parser.add_argument('--which_trajs_test','-u',default=None,type=str,help='Which trajectories (A,B,C...) with this naming are to be tested against')
        parser.add_argument('--traj_links','-L',default=None,type=dict,help='Targets for links to create for training trajectories')
        parser.add_argument('--traj_links_valid','-V',default=None,type=dict,help='Targets for links to create for validation trajectories')
        parser.add_argument('--traj_links_test','-U',default=None,type=dict,help='Targets for links to create for testing trajectories')
        parser.add_argument('--ref_mol_seed_dict','-z',default={},type=dict,help='Dictionary of seeds, for trajectory sets with varying seed names')
        parser.add_argument('--ref_mol_dir','-r',default=None,type=str,help='Location of output of solutes run from which to find reference energies')

        return parser

    def validate_args(args):
        default_args = make_parser().parse_args(['--seed','a'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Comparison of results from trajectory to calculator

# In[ ]:


import numpy as np

from esteem.trajectories import compare_traj_to_traj,atom_energy
    
def compare_wrapper_to_traj(wrapper,calc_params,trajin,trajout_file):
    """Compare the energy and force predictions of a calculator to results in an existing trajectory"""

    trajout = Trajectory(trajout_file,'w')

    # Loop over the frames in the input trajectory
    for i,frame in enumerate(trajin):

        # Read in total energy and forces from trajectory
        e_traj = frame.get_potential_energy()
        f_traj = frame.get_forces()
        d_traj = frame.get_dipole_moment()
        
        err=False
        seed=f"{calc_params['calc_seed']}{i:4d}"
        
        e_calc, f_calc, d_calc, calc_ml = wrapper.singlepoint(frame,seed,calc_params,
             forces=True,dipole=True)

        # Calculate RMS and Max force errors
        rms_fd = np.sqrt(np.mean((f_traj-f_calc)**2))
        max_fd = np.max(np.sqrt((f_traj-f_calc)**2))
        rms_dd = np.sqrt(np.mean((d_traj-d_calc)**2))

        # Print header for columns
        if (i==0):
            print('#Idx    E_traj (eV)  E_calc (eV)      E_diff (eV)       RMS_fd    MAX_fd   RMS_dd')

        if isinstance(calc_params["calc_suffix"],dict):
            print(f'{i:4d} {e_traj:12.5f} {np.mean(e_calc):12.5f} {np.std(e_calc):8.5f} {np.mean(e_traj-e_calc):8.5f} {np.mean(np.abs(e_traj-e_calc)):8.5f} {rms_fd:8.5f} {max_fd:8.5f}')
        else:
            print('%4d   %12.5f %12.5f     %12.8f     %8.5f  %8.5f %8.5f' %
              (i, e_traj, e_calc, e_traj-e_calc, rms_fd,max_fd,rms_dd))
        
        # Assemble dictionary of properties to write to the output trajectory
        kw = {'dipole': d_calc,
              #'charges': q_calc,
              'energy': e_calc,
              'forces': f_calc}

        # Write to trajectory
        trajout.write(frame,**kw)

    trajout.close()


# # Command-line driver

# In[ ]:


def get_parser():
    mltest = MLTestingTask()
    return mltest.make_parser()

if __name__ == '__main__':

    from esteem import wrappers
    mltest = MLTestingTask()
   
    # Parse command line values
    args = mltrain.make_parser().parse_args()
    for arg in vars(args):
        setattr(mltrain,arg,getattr(args,arg))
    print('#',args)
    mltest.wrapper = wrappers.amp.AMPWrapper()

    # Run main program
    mltest.run()

