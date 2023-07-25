#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Sets up and runs solvated Molecular Dynamics in Explicit Solvent"""


# # Setup routine for the Solvate task

# In[ ]:


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
# 64 with a collision frequency of 1 ps

from os import path
from ase.io import read
from ase.io.trajectory import Trajectory
from ase import Atoms

def counterion_charge(counterions):
    """
    Calculate the net charge on the solute, given a set of counterions
    """
    netcharge = 0
    for at in counterions:
        if at.symbol in ['H','Na','Li','K','Rb','Cs']:
            netcharge = netcharge - 1
        if at.symbol in ['Be','Mg','Ca','Sr','Ba']:
            netcharge = netcharge - 2
        if at.symbol in ['F','Cl','Br','I']:
            netcharge = netcharge + 1
    return netcharge

class SolvateTask:

    def __init__(self,**kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'solvate'
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self,arg,getattr(args,arg))

    # Main program for Solvated MD
    def setup_amber(self):
        """
        Handles setup of the solvated model using AmberTools.

        The setup has 5 main sections:

            1. Counterions are set up

            2. Amber input files for the solute molecule are created

            3. Amber input files for the solvent molecule are created

            4. A box of solvent is added to the solute, and counterions added

            5. Restraints are calculated for the counterions

        args: namespace or class

            Input variables to the routine - see listing under Command-Line usage for details

            Generate with a call to solvate.make_parser()

        wrapper: class
            Wrapper to run calculations of this task
        """

        # Check input args are valid
        #self.validate_args()

        # Find counterions, if set
        counterions = Atoms()
        if isinstance(self.counterions,dict):
            if self.solute in self.counterions:
                counterions = Atoms(self.counterions[self.solute])
        if isinstance(self.counterions,str):
            counterions = Atoms(self.counterions)

        # Count counterion charge
        netcharge = counterion_charge(counterions)     

        solvatedseed = self.solute+"_"+self.solvent+"_solv"

        # Prepare Solute inputs
        if not path.exists(self.solute+".prmtop"):
            if self.use_acpype:
                print('Preparing Amber inputs for solute using ACPYPE')
                self.wrapper.prepare_input_acpype(self.solute,netcharge=netcharge,offset=0)
            else:
                print('Preparing Amber inputs for solute using antechamber, parmchk and tleap')
                self.wrapper.prepare_input(self.solute,netcharge=netcharge,offset=0)
        else:
            print('Using pre-existing Amber inputs for solute')
        print('Counterion net charge = ',netcharge)

        # Prepare Solvent inputs
        if self.solvent not in self.wrapper.known_solvents:
            if not path.exists(self.solvent+".prmtop"):
                print('Preparing Amber inputs for solvent')
                if self.solvent == self.solute:
                    offset = 0
                else:
                    offset = 99
                if self.use_acpype:
                    print('Preparing Amber inputs for solvent using ACPYPE')
                    self.wrapper.prepare_input_acpype(self.solvent,netcharge=netcharge,offset=offset)
                else:
                    print('Preparing Amber inputs for solvent using antechamber, parmchk and tleap')
                    self.wrapper.prepare_input(self.solvent,netcharge=netcharge,offset=offset)
            else:
                print('Using pre-existing Amber inputs for solvent')

        # Add solvent box and output pdb file of solvated model
        print('Preparing solvated model of solute')
        self.wrapper.add_solvent_box(self.solute,self.solvent,self.counterions,solvatedseed,self.boxsize)
        #self.wrapper.crd_to_crdnc(solvatedseed,solvatedseed)
        self.wrapper.fix_amber_pdb(solvatedseed)

        # Handle restraints, if present
        if self.restraints is not None:
            self.wrapper.add_restraint(solvatedseed,self.solute[:3],self.restraints,self.rest_r)

        # Pass common inputs into Amber module
        self.wrapper.dt = self.timestep
        self.wrapper.temp0 = self.temp
        self.wrapper.cut = self.ewaldcut
        self.wrapper.gamma_ln = self.gammaln

    def setup_lammps(self):
        """
        Handles setup of the solvated model for a LAMMPS calculation (also uses AmberTools)

        args: namespace or class
            input variables to the routine - see listing under Command-Line usage for details
        """

        from esteem.wrappers.amber import AmberWrapper
        wrapper_amber = AmberWrapper()
        orig_wrapper = self.wrapper
        self.wrapper = wrapper_amber
        self.setup_amber()
        self.wrapper = orig_wrapper

        # Convert to LAMMPS input with Intermol
        from intermol import convert
        amb_in = [f"{self.solute}_{self.solvent}_solv.prmtop",f"{self.solute}_{self.solvent}_solv.crd"]
        system, prefix, prmtop_in, crd_in, amb_structure = convert._load_amber(amb_in)

        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-ls', '--lammpssettings', dest='lmp_settings',
                metavar='settings', default='pair_style lj/cut/coul/long 9.0 9.0\npair_modify tail yes\nkspace_style pppm 1e-8\n\n',
                help='pair_style string to use in the output file. Default is a periodic Ewald simulation')
        intermol_args = vars(parser.parse_args(""))
        oname = f"{self.solute}_{self.solvent}_solv"
        output_status = dict()
        print(system)
        convert._save_lammps(system,oname,output_status,intermol_args)

    # Main program for Solvated MD
    def run(self):
        """
        Handles running of MD on a solvated model using an MD Wrapper (currently Amber or LAMMPS).

        There are five phases to the task:

        Setup, Heating, Density Equilibration, Equilibration and Snapshot Generation

        Constraints are turned on and off as appropriate in different phases:

        SHAKE constraints on -H during heating and density equil, no constraints on -H during equil and snapshots

        Counterions are restrained from coming too close to the solute

        args: namespace or class
            Argument list for the whole job, with members including:
        wrapper: namespace or class
            Wrapper for running components of the job, with members including:

            ``singlepoint``, ``minimise``, ``heatup``, ``densityeq``, ``equil`` and ``snapshots``

        *Outputs*:
            A trajectory file, named '{solute}_{solvent}_{md_suffix}/{solute}_{solvent}_solv.traj' which
            contains ``self.nsnaps`` geometries of the solvated box, each spaced by ``self.nsteps`` steps
            of ``self.timestep`` units of time (wrapper-dependent).

            An example for catechol in water with the default ``md_suffix``
        """
        
        from os.path import isfile

        # Load in models from pdb files
        solute = read(self.solute+".pdb")
        solvent = read(self.solvent+".pdb")

        solvatedseed = self.solute+"_"+self.solvent+"_solv"
        solvated = read(solvatedseed+'.pdb')

        calc_params = {'calc_seed':'cate','calc_suffix':'3x15_R8.0','calc_prefix':'../','target':0}
        e0_am_solu = self.wrapper.singlepoint(solute,self.solute,calc_params)
        print('\nSolute ground state energy: ',e0_am_solu)
        e0_am_solv = self.wrapper.singlepoint(solvent,self.solvent,calc_params)
        print('\nSolvent ground state energy: ',e0_am_solv)
        e0_am_solvate = self.wrapper.singlepoint(solvated,solvatedseed,calc_params)
        print('\nSolvated model contains ',len(solvated.positions),' atoms')
        print('\nSolvated model energy: ',e0_am_solvate)

        calc_params = {}

        # Minimize first? (not helpful - commented out)
        minimised = solvated.copy()
        #wrapper.minimise(solvatedseed,solvated,minimised)

        heated = minimised.copy()
        if isfile('heat.rst'):
            print('\nReading from heat.rst')
            solvated.calc.read_coordinates(heated,'heat.rst')
        else:
            print('\nHeating model to target temperature')
            self.wrapper.heatup(heated,solvatedseed,calc_params=calc_params,nsteps=self.nheat)

        densityeq = heated.copy()
        if isfile('density.rst'):
            print('\nReading from density.rst')
            solvated.calc.read_coordinates(densityeq,'density.rst')
        else:
            print('\nEquilibrating density of model')
            self.wrapper.densityequil(densityeq,solvatedseed,calc_params=calc_params,nsteps=self.ndens)

        equbd = densityeq.copy()
        if isfile('equil.rst'):
            print('\nReading from equil.rst')
            solvated.calc.read_coordinates(equbd,'equil.rst')
        else:
            print('\nEquilibrating at fixed volume')
            self.wrapper.equil(equbd,solvatedseed,calc_params=calc_params,nsteps=self.nequil)

        snap = equbd.copy()
        start = 0
        for i in range(self.nsnaps):
            if isfile(f'snap{i:04}.rst'):
                start=i
        if start>0:
            print(f'\nResuming snapshot generation from i={start}')
            solvated.calc.read_coordinates(snap,f'snap{start:04}.rst')
        else:
            print('\nGenerating snapshots')
            
        self.wrapper.snapshots(snap,solvatedseed,calc_params=calc_params,
                               nsnaps=self.nsnaps,nsteps=self.nsteps,start=start)

    def make_parser(self):

        import argparse
    
        main_help = ('Generates Solvated MD trajectory files. There are five \n'+
                     'phases to the calculation: Setup, Heating, Density \n'+
                     'Equilibration, Equilibration and Snapshot Generation \n.'+
                     'Setup: solvates the solute molecule, and sets up and parameters \n'+
                     'Heating: ramps up the temperature from zero to target \n'+
                     'Density Equilibration: variable cell optimisation with constraints to find correct density \n'+
                     'Equilibration: removal of constraints, equilibration to correct ensemble \n'+
                     'Snapshot Generation: Multiple short sequences of MD, intended to be more \n'+
                     'than any correlation time, at the end of which a snapshot is saved to a trajectory.')
        epi_help = ('Note: Writes the output trajectory in the format \n'+
                    '<solute>_<solvent>_solv.traj')
        from argparse import ArgumentParser, SUPPRESS
        parser = ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--solute','-u',required=False,type=str,help='Name of solute molecule')
        parser.add_argument('--solvent','-v',required=False,type=str,help='Name of solvent molecule')
        parser.add_argument('--boxsize','-b',default=20,type=int,help='Size of simulation box for MD calculation')
        parser.add_argument('--timestep','-t',default=0.002,type=float,help='Time step of Molecular Dynamics runs')
        parser.add_argument('--temp','-T',default=300.0,type=float,help='Thermostat temperature for Molecular Dynamics runs')
        parser.add_argument('--nheat','-H',default=10000,type=int,help='Number of MD steps in Heating phase')
        parser.add_argument('--ndens','-D',default=50000,type=int,help='Number of MD steps in Density Equilibration phase')
        parser.add_argument('--nequil','-E',default=50000,type=int,help='Number of MD steps in Equilibration phase')
        parser.add_argument('--nsteps','-S',default=2000,type=int,help='Number of MD steps between each snapshot in Snapshots phase')
        parser.add_argument('--nsnaps','-N',default=200,type=int,help='Number of Snapshots to save to trajectory')
        parser.add_argument('--restraints','-R',default=None,action='append',nargs=4,type=str,help='Specifies restraint atoms')
        parser.add_argument('--rest_r','-r',default=None,nargs=6,type=float,help='Specifies parameters for restraint')
        parser.add_argument('--md_suffix','-m',default='md',nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--md_geom_prefix',default='gs_PBE0/is_opt',nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--counterions','-C',default={},type=str,help='Counterion(s) to add, eg Na')
        # Wrapper Dependent
        parser.add_argument('--ewaldcut','-e',default=12.0,type=float,help='Cutoff length for Ewald calculation (See Amber manual)')
        parser.add_argument('--gammaln','-g',default=1.0,type=float,help='Thermostat parameter: Collision frequency in ps^-1')
        parser.add_argument('--use_acpype','-P',default=False,type=bool,help='Use ACPYPE')

        return parser

    def validate_args(self):
        default_args = self.make_parser().parse_args(['--solute','a','--solvent','b'])
        for arg in vars(self):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")

def get_parser():
    return SolvateTask().make_parser()
                
if __name__ == '__main__':
    # Parse command line values
    from esteem.wrappers import amber
    args = make_parser().parse_args()
    print(args)
    solv = SolvateTask()
    solv.wrapper = amber.AmberWrapper()
    solv.wrapper.setup_amber()
    solv.run()


# In[ ]:




