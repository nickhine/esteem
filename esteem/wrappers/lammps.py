#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines the LAMMPSWrapper class"""

from ase.calculators.lammpsrun import LAMMPS
from os import environ, path, makedirs
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.io import read
import subprocess
import shutil

#    mass = grams/mole
#    distance = Angstroms
#    time = picoseconds
#    energy = eV
#    velocity = Angstroms/picosecond
#    force = eV/Angstrom
#    torque = eV
#    temperature = Kelvin
#    pressure = bars
#    charge = multiple of electron charge (1.0 is a proton)
#    dipole = charge*Angstroms
#    electric field = volts/Angstrom

# See http://ambermd.org/tutorials/basic/tutorial1/section3.htm 
# and http://ambermd.org/tutorials/advanced/tutorial3/section1.htm for details

# From J. Chem. Phys. 148, 024110 (2018)
# The dyes are placed in large solvent boxes (see Table I for
# the number of atoms in the MD box for each system) and a
# two-step equilibration is carried out. First, a 20 ps temperature
# equilibration in the NVT ensemble is performed to raise the
# temperature of the system from 0 K to 300 K. This is followed
# by a 400 ps volume equilibration in the NPT ensemble. Since
# we are interested in generating uncorrelated snapshots rather
# than accurate short time scale dynamics, we run all produc-
# tion calculations in the NVT ensemble to guarantee a constant
# temperature. For the production trajectory of 8 ns in length,
# solute-solvent snapshots are extracted every 4 ps, producing a
# total of 2000 uncorrelated snapshots. All MD calculations are
# performed using a 2 fs time-step and a Langevin thermostat
# with a collision frequency of 1 ps^-1

class LAMMPSWrapper():
    """Sets up the LAMMPS Calculator (via ASE) for Molecular Dynamics runs"""

    def __init__(self):
        self.known_solvents = []

        # common parameters
        self.cut = 9.0       # 9 Ang cutoff for Ewald
        self.temp0 = 300.0   # 300 K
        self.ntt = 3         # Langevin thermostat
        self.gamma_ln = 1    # Collision frequency 1ps^-1
        self.dt = 0.002      # 2 fs timestep
        self.restraints = '' # Line to add to inputs for restraints
        self.ntb = 0         # Periodic box
        self.ntp = 0         # Periodic box

        # Common files (all run types will use same potentials, presumably, so use instance attribute to store)
        self.lammps_files  = ["watr_converted.lmp"]
        # Store a dictionary of common params, then merge dictionary with specifics for a given run?
        self.lammps_params_amberconv = {"keep_tmp_files": True,
                              "tmp_dir": ".",
                              "units": "real",
                              "atom_style": "full",
                              "dimension": "3",
                              "boundary": "p p p",
                              "kspace_style": "pppm 1e-8",
                              "bond_style": "hybrid harmonic",
                              "angle_style": "hybrid harmonic",
                              "special_bonds": "lj 0.0 0.0 1.0 coul 0.0 0.0 1.0",
                              #"read_data": "watr_converted.lmp",
                              "pair_style": f"lj/cut/coul/long {self.cut} {self.cut}",
                              "pair_modify": "tail yes",
                              "pair_coeff": ["1 1   0.2104000   3.0664734","2 2   0.0000000   0.0000000"],
                              "pair_modify": ["tail yes", "mix arithmetic"]
                             }
        self.damp = 1.00
        self.temp0 = 300
        self.seed = 123457
        self.lammps_md = {
            #"timestep": 0.002,
            "fix": [#"fix_nve all nve",
                    f"fix_lan all langevin {self.temp0} {self.temp0} {self.damp} {self.seed}"],
            "thermo_style": "custom step temp press ke pe etotal",
            "thermo_modify": "flush yes format float %20.10g",
            "thermo": 100
        }
        self.lammps_params_prophet = {
            "keep_tmp_files": True,
            "write_velocities": True,
            "tmp_dir": ".",
            "units": "metal"}

    def calc_filename(self,seed,target,prefix='',suffix=''):
    
        from esteem.wrappers.amp import AMPWrapper
        self.calc_ext = ".amp"
        self.log_ext = "-log.txt"
        return AMPWrapper.calc_filename(self,seed,target,prefix,suffix)

    def lammps_setup(self,lammps_cmd=None,nprocs=None):
        """Prepares run commands etc for LAMMPS calculations"""

        # Set up  executable command
        try:
            lammps_cmd = environ["ASE_LAMMPSRUN_COMMAND"]
        except KeyError:
            if lammps_cmd is None:
                lammps_cmd = "lmp_serial"
            nproc_cmd = ''
            mpirun = ''
            if nprocs is not None:
                nproc_cmd = f'-np {nprocs}'
                mpirun = 'mpirun'
            environ["ASE_LAMMPSRUN_COMMAND"]=f'{mpirun} {nproc_cmd} {lammps_cmd}'
             #PREFIX.nwi >> PREFIX.nwo 2> PREFIX.err

    def load(self,seed,target=None,prefix="",suffix="",**kwargs):
        """
        Loads an existing AMP Calculator and converts it into a PROPhet-LAMMPS calculator

        seed: str
        
        target: int

        suffix: str

        kwargs: dict
            other keywords to pass to AMP.load
        """

        # Load AMP calculator and convert to PROPhet format
        from amp import Amp, convert
        calcfn = self.calc_filename(seed,target,prefix=prefix,suffix=suffix)
        calc_amp = Amp.load(calcfn+self.calc_ext,**kwargs)
        Rcut = calc_amp._descriptor.parameters.cutoff['kwargs']['Rc']
        pair_coeff = [f"{i+1} potential_{e}" for i,e in enumerate(calc_amp._descriptor.parameters.elements)]
        convert.save_to_prophet(calc_amp)

        # Create LAMMPS calculator
        from os import environ
        environ["ASE_LAMMPSRUN_COMMAND"]='lmp_serial'
        calc_ml = LAMMPS()
        calc_ml.set(tmp_dir=".")
        calc_ml.set(keep_tmp_files=False) # set to true to debug
        calc_ml.set(units="metal")
        calc_ml.set(pair_style=f"nn {Rcut}")
        calc_ml.set(pair_coeff=pair_coeff)
        
        return calc_ml

    def geom_opt(self,model,seed,calc_params,driver_tol='default',
                 solvent=None,readonly=False):
        """
        Runs a singlepoint calculation with the LAMMPS ASE calculator

        model: ASE Atoms

        seed: str
        
        suffix: str

        dummy: str

        target: int or None

        solvent: str or None

        readonly: bool
        """
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']

        # TODO: Implement tolerances based on driver_tol
        etol = 1e-5
        ftol = 1.0e-8
        maxiter = 1000
        maxeval = 100000
        # LAMMPS does not produce correct answers for these potentials if
        # run with no cell - so add one
        if (model.get_cell() == [0.0, 0.0, 0.0]).all():
            model.center(10)
        # Load the appropriate Calculator
        print('in geom_opt:',calc_seed,target,suffix)
        calc_ml = self.load(calc_seed,target,prefix="",suffix=suffix)
        calc_ml.set(**self.lammps_params_prophet)
        calc_ml.set(minimize=f"{etol} {ftol} {maxiter} {maxeval}")
        model.calc = calc_ml
        return model.get_potential_energy(), model.get_forces(), model.get_positions()

    
    def run_mlmd(self,model,mdseed,calc_params,md_steps,md_timestep,superstep,temp,
                 solvent=None,restart=False,readonly=False,constraints=None,continuation=None):
        """
        Runs a Molecular Dynamics calculation with the AMP ASE calculator.

        model: ASE Atoms

        seed: str
        
        target: int
        
        suffix: str

        md_steps: int

        md_timestep: float

        superstep: int

        temp: float

        target: int or None

        solvent: str or None

        restart: bool

        readonly: bool
        """

        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase import units
        import numpy as np
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']
        
        # Load the appropriate Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix)
        calc_ml.set(**self.lammps_params_prophet)
        calc_ml.set(run=md_steps)
        
        # LAMMPS does not produce correct answers for these potentials if
        # run with no cell - so add one
        if (model.get_cell() == [0.0, 0.0, 0.0]).all():
            model.center(10)

        # Initialise velocities if this is first step, otherwise inherit from model
        if np.all(model.get_momenta() == 0.0):
            print(f'Initializing new momenta at {temp}K')
            MaxwellBoltzmannDistribution(model, temp * units.kB)
        
        Stationary(model)
        ZeroRotation(model)
        # TODO Set timestep
        #      Set thermostat
        model.calc = calc_ml
        return model.get_potential_energy(), model.get_forces()
        
        
    def singlepoint(self,model,seed,calc_params,
                    target=None,solvent=None,readonly=False):
        """
        Runs a singlepoint calculation with the LAMMPS ASE calculator

        model: ASE Atoms

        seed: str
        
        suffix: str

        dummy: str

        target: int or None

        solvent: str or None

        readonly: bool
        """
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']
        lammps_params = calc_params['lammps_params']

        # Load the appropriate Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix)
        # LAMMPS does not produce correct answers for these potentials if
        # run with no cell - so add one
        if (model.get_cell() == [0.0, 0.0, 0.0]).all():
            model.center(10)

        calc_ml.set(**lammps_params)
        model.calc = calc_ml
        return model.get_potential_energy(), model.get_forces()

    def minimise(self,solvated,minimised,seed):
        """Runs a geometry optimisation calculation with the LAMMPS ASE calculator"""

        calc_min = LAMMPS(files=lammps_files)
        calc_min.set(**lammps_params)
        model.calc = calc_min
        calc_min.set(minimize='1.0e-4 1.0e-6 1000 4000')

        return model.get_potential_energy()
        minimised.set_calculator(calc_min)
        print("Energy after minimisation: ", minimised.get_potential_energy())

    def heatup(self,minimised,heated,seed,nsteps):
        """Runs a heatup temperature-ramp calculation with the LAMMPS ASE calculator"""

        # NTC = 2, NTF = 2: hydrogens constrained at this stage with SHAKE
        # irest = 0 (new simulation)

        # Load the appropriate Calculator
        target = 0; suffix = "3x15_R8.0"; prefix='../'; calc_seed='cate'
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix)

        calc_ml.set(**self.lammps_params_amberconv)
        calc_ml.set(**self.lammps_md)
        calc_ml.set(fix=[f'fixlan all langevin {0} {self.temp0}  120  123456'])
        calc_ml.set(run=nsteps)
        heated = minimised.copy()
        heated.calc = calc_ml
        print("Energy after heating: ", heated.get_potential_energy())

    def densityequil(self,heated,densityeq,seed,nsteps):
        """Runs a density equilibration calculation with fixed hydrogens with the LAMMPS ASE calculator"""

        # NTB = 2, NTP = 1, TAUP = 1.0: Use constant pressure periodic boundary. Isotropic position scaling
        # should be used to maintain the pressure (NTP=1) and a relaxation time of 1 ps should be used (TAUP=1.0).
        # NTC = 2, NTF = 2: hydrogens constrained at this stage
        # irest = 1 (restart from previous simulation)

        calc_dens = LAMMPS(files=lammps_files,parameters=lammps_params)
        densityeq.calc = calc_dens
        print("Energy after density equilibration: ", densityeq.get_potential_energy())

    def equil(self,densityeq,equbd,seed,nsteps):
        """Runs an equilibration calculation at constant volume with flexible hydrogens with the LAMMPS ASE calculator"""
        # NTP = 0: No pressure scaling (constant volume)
        # NTC = 1: SHAKE not used - no constraints
        # Energies at this stage no longer comparable to previous steps
        # due to extra DOFs

        calc_equil = LAMMPS(files=lammps_files,parameters=lammps_params)
        equbd.calc = calc_equil
        print("Energy after equilibration:",  equbd.get_potential_energy())

    def snapshots(self,seed,snapin,snapout,nsnaps,nsteps):
        """Runs a long MD trajectory for snapshot generation with the LAMMPS ASE calculator"""
        # NTC = 1: SHAKE not used - no constraints

        step = 0
        trajname = seed+'.traj'
        traj = Trajectory(trajname, 'w')
        calc_snap = LAMMPS(files=lammps_files,parameters=lammps_params)
        for step in range(nsnaps):
            snapin.calc = calc_snap
            print("Energy after snapshot",str(step),":",  snapin.get_potential_energy())
            traj.write(snapout)
            write(f'{seed}_snap_solv{step:04}.xyz',snapout)


# In[ ]:


# l = LAMMPSWrapper()
# import os
# os.chdir("/home/theory/phspvr/cate_qmd")
# l.load('cate',suffix='3x10_R6.5')
# l.temp0 = 500
# from ase.io import read; cate = read("cate.xyz"); cate.center(20)
# l.heatup("cate",cate,cate,1000)


# In[ ]:


#from amp import Amp
#calc_amp = Amp.load('cate_gs_3x10_R6.5.amp')
# !ls -l log*
# !cat log_lammps0000018rs86xmj


# In[ ]:


#from amp import descriptor


# In[ ]:


#calc_amp._descriptor.parameters.elements


# In[ ]:




