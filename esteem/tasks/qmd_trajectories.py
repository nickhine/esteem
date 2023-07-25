#!/usr/bin/env python
# coding: utf-8

# # Run main program

# In[ ]:


import sys
import ase.units

from esteem.trajectories import generate_md_trajectory, recalculate_trajectory
from esteem.trajectories import find_initial_geometry, get_trajectory_list


class QMDTrajTask:

    def __init__(self,**kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'qmd'
        self.train_params = {}
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self,arg,getattr(args,arg))

    # Main routine
    def run(self):
        """
        Sets up and runs an ab initio molecular dynamics run on a given molecule (whose name is
        provided by ``args.seed``) in the ground or excited state (specified by ``args.target``).

        Results, including a trajectory file with the stored snapshots, are saved to files
        appending ``args.suffix`` to the seed and state, for use in future runs.

        The run is divided into equilibration (``args.nequil`` runs of ``args.qmd_steps`` MD steps
        each, with timestep ``args.qmd_timestep``), then snapshot generation (``args.nsnap`` runs of
        ``args.qmd_steps`` MD steps each, with timestep ``args.qmd_timestep``).

        A thermostat (wrapper-dependent) at temperature ``args.temp`` means we stay in the NVT ensemble.

        Constraints can be applied using ``args.constraints`` - the meaning depends on the underlying wrapper.

        Optionally can be used to recalculate singlepoint energies for the steps of a pre-existing trajectory.

        args: namespace or class
            Full set of arguments to the QMD_Trajectories task - see below for a listing.

            Key arguments include ``basis``, ``func``, ``qmd_timestep``, ``qmd_steps``, ``nsnap``,
            ``nequil``, ``temp``, ``constraints``

            Generate with a call to qmd_trajectories.make_parser()
        wrapper: namespace or class
            List of functions for running components of the job, with members including:

            ``singlepoint``, ``geom_opt`` and ``qmd``.
        """

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
        calc_params = {"basis": self.basis,
                       "func": self.func,
                       "target": self.target,
                       "disp": self.disp}
        
        if self.seed in self.charges:
            charge = self.charges[self.seed]
        else:
            charge = 0

        # Perform QM calculations on an existing trajectory
        if self.input_suffix is not None:
            for traj_label in which_trajs:
                input_target = None # re-do if ever necessary to start from anything else
                input_traj_range = self.input_traj_range
                recalculate_trajectory(self.seed,self.target,traj_label,self.traj_suffix,
                                       input_target,self.input_suffix,
                                       self.wrapper,calc_params,charge=charge,
                                       solvent=self.solvent,input_traj_range=input_traj_range)
            return

        # Generate training data in ntraj labelled trajectories
        for traj_label in which_trajs:

            # Find (or relax) initial geometry
            if isinstance(self.target,list):
                targ = self.target[0]
            else:
                targ = self.target
            calc_params['target'] = targ
            model = find_initial_geometry(self.seed,geom_opt_func=None,calc_params={}, #self.wrapper.geom_opt,calc_params,
                                          which_traj=traj_label,ntraj=self.ntraj)
            if self.constraints is not None:
                if 'spring' in self.constraints:
                    spring = self.constraints['spring']
                    if 'bond' in spring:
                        # Extract atoms indices and bondlength from NWChem constraint
                        # NWChem expects 1-indexed atom numbers, ASE expects 0-indexed
                        # so subtract 1
                        atoms0 = int(spring.split()[1]) - 1 
                        atoms1 = int(spring.split()[2]) - 1
                        bondlength = float(spring.split()[4])*ase.units.Bohr
                        print('Setting distance: ',atoms0,atoms1,bondlength)
                        model.set_distance(atoms0,atoms1,bondlength,fix=0)

            # Pass in routine to actually run MD into generic Snapshot MD driver
            generate_md_trajectory(model,self.seed,self.target,traj_label,self.traj_suffix,
                                   wrapper=self.wrapper,count_snaps=self.nsnap,count_equil=self.nequil,
                                   md_steps=self.qmd_steps,md_timestep=self.qmd_timestep,md_friction=None,
                                   temp=self.temp,calc_params=calc_params,
                                   solvent=self.solvent,charge=charge,constraints=self.constraints,
                                   store_full_traj=True)

 
    # Create parser to read command line values
    def make_parser(self):

        import argparse

        main_help = ('''Generate trajectory files by running QMD or by recalculating
                        energies and forces at previously evaluated trajectory points.''')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--seed','-s',type=str,help='')
        parser.add_argument('--traj_suffix','-S',default='training',type=str,help='String to append to names of trajectories')
        parser.add_argument('--geom_prefix','-G',default='',type=str,help='String to append to filenames for initial geometries')
        parser.add_argument('--basis','-b',default='6-311++G**',type=str,help='Basis set definition (or other run parameter)')
        parser.add_argument('--func','-f',default='PBE0',type=str,help='Exchange-Correlation functional (or other run parameter)')
        parser.add_argument('--disp','-d',default=True,type=bool,help='Apply Grimme-D3 Dispersion')
        parser.add_argument('--target','-t',default=None,type=int,help='Excited state index (None for ground state)')
        parser.add_argument('--charges','-C',default={},nargs='?',type=dict,help='Charges on molecular species. Not for command-line use')
        parser.add_argument('--restart','-r',default=False,type=bool,help='If the trajectory has been run for one excited state already, setting this to true attempts to make the calculator restart at each geometry')
        parser.add_argument('--md_timestep','-q',default=0.5,type=float,help='Molecular dynamics timestep (wrapper-dependent units, fs for NWChem)')
        parser.add_argument('--md_steps','-Q',default=100,type=int,help='Number of timesteps in each molecular dynamics run')
        parser.add_argument('--temp','-T',default=300.0,type=float,help='Thermostat temperature (NVT ensemble)')
        parser.add_argument('--ntraj','-n',default=1,type=int,help='Total number of named (A_Z,a-z) trajectories')
        parser.add_argument('--nsnap','-N',default=100,type=int,help='Number of snapshot runs')
        parser.add_argument('--solvent','-i',default=None,type=str,help='Solvent for implicit solvent runs.')
        parser.add_argument('--solvent_settings',default=None,type=str,help='Solvent settings for implicit solvent runs.')
        parser.add_argument('--input_suffix','-I',default=None,type=str,help='Appended string to identify input trajectory: if present, the run will resample an existing trajectory')
        parser.add_argument('--input_traj_range','-R',default=None,type=int,help='Range of snapshots to use from input trajectory')
        parser.add_argument('--nequil','-e',default=5,type=int,help='Number of equilibration runs.')
        parser.add_argument('--which-trajs','-w',default=None,type=str,help='Which trajectories should be generated, named with letters A-Z, a-z')
        parser.add_argument('--constraints','-c',default=None,type=str,help='Constraints (wrapper-dependent)')
        parser.add_argument('--dynamics','-X',default=None,type=str,help='Dynamics (ASE Dynamics class)')

        return parser

        # Notes on defaults:
        # 1 a.u. = 0.02419 fs, 100 steps of 10 aut = 24fs between snapshots

    def validate_args(args):
        default_args = make_parser().parse_args("")
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Handle Inputs

# In[ ]:


def get_parser():
    qmdtraj = QMDTrajTask()
    return qmdtraj.make_parser()

if __name__ == '__main__':

    # Set up NWChem by default
    from esteem.wrappers import nwchem
    
    qm_wrapper = nwchem.NWChemWrapper()
    qm_wrapper.nwchem_setup()

    # Parse command line arguments
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    
    # Run main program
    main(args,qm_wrapper)

