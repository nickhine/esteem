#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines the PhysNetWrapper Class"""

import numpy as np

class PhysNetWrapper():
    """
    Sets up, trains and runs a PhysNet Neural Network Calculator to represent a
    potential energy surface.
    """

    # Class attribute: default training arguments
    default_train_args =  {'num_features': 128,
                           'num_basis': 64,
                           'num_blocks': 5,
                           'num_residual_atomic': 2,
                           'num_residual_interaction': 3,
                           'num_residual_output': 1,
                           'cutoff': 10.0,
                           'use_electrostatic': 1,
                           'use_dispersion': 1,
                           'grimme_s6': 1.0, #0.5,
                           'grimme_s8': 1.2177, #0.2130,
                           'grimme_a1': 0.4145, #0.0,
                           'grimme_a2': 4.8593, #6.0519,
                           'dataset': 'test.npz',
                           'num_train': 450,
                           'num_valid': 50,
                           'seed': 42,
                           'max_steps': 1000000,
                           'learning_rate': 0.001,
                           'max_norm': 1000.0,
                           'ema_decay': 0.999,
                           'keep_prob': 1.0,
                           'l2lambda': 0.0,
                           'nhlambda': 0.01,
                           'decay_steps': 10000000,
                           'decay_rate': 0.1,
                           'batch_size': 32,
                           'valid_batch_size': 50,
                           'force_weight': 52.91772105638412,
                           'charge_weight': 14.399645351950548,
                           'dipole_weight': 27.211386024367243,
                           'summary_interval': 10000,
                           'validation_interval': 10000,
                           'save_interval': 50000,
                           'record_run_metadata': 0}
    
    def __init__(self,**kwargs):
        """Sets up instance attributes for PhysNetWrapper """
        from copy import deepcopy
        self.train_args = deepcopy(self.default_train_args)

        # Allow overrides for this instance of the class
        for kw in self.train_args:
            if kw in kwargs:
                self.train_args[kw] = kwargs[kw]

        # Make a set of default loading arguments by copying in training arguments
        self.load_args = {}
        # These have different names between training and loading
        self.load_args['F'] = self.train_args['num_features']
        self.load_args['K'] = self.train_args['num_basis']
        self.load_args['sr_cut'] = self.train_args['cutoff']
        if 'lr_cut' in kwargs:
            self.load_args['lr_cut'] = kwargs['lr_cut']
        else:
            self.load_args['lr_cut'] = None # no need for an electrostatic cutoff
        for s in ['grimme_s6','grimme_s8','grimme_a1','grimme_a2']:
            kw = s.replace('grimme_','')
            if s in self.train_args:
                self.load_args[kw] = self.train_args[s]            
        # The rest have the same name
        for kw in ['num_residual_atomic','num_residual_interaction','num_residual_output',
                   'use_electrostatic','use_dispersion']:
            if kw in self.train_args:
                self.load_args[kw] = self.train_args[kw]
        for kw in ['use_ewald','ewald_alpha','ewald_kmax','ewald_Nmax','skin']:
            if kw in kwargs:
                self.load_args[kw] = kwargs[kw]
        self.calc = None
        self.calc_params = None
        self.atom_e = 0.0
        self.atom_energies = {}
        self.update_atom_e = False
    
    def calc_filename(self,seed,target,prefix='',suffix=''):
        if target is None or target == 0:
            calcfn = seed+"_gs_"+suffix
        else:
            calcfn = seed+"_es"+str(target)+"_"+suffix
            
        calcfn = prefix + calcfn
            
        return calcfn
    
    def load(self,seed,target=None,prefix="",suffix="",atoms=None):
        """
        Loads an existing PhysNet Calculator

        seed: str
        
        target: int

        suffix: str

        kwargs: dict
            other keywords
        """

        # Check against or store previous calculator parameters
        if self.calc_params is not None:
            if ((self.calc_params['target'] != target) or
                #(self.calc_params['calc_prefix'] != prefix) or
                (self.calc_params['calc_suffix'] != suffix) or
                (self.calc_params['calc_seed'] != seed)):
                raise Exception('Attempted change of calculator parameters for previously-loaded wrapper. Not supported.')
        self.calc_params = {'target': target,'calc_prefix': prefix,
                            'calc_suffix': suffix,'calc_seed': seed}
            
        if self.calc is not None: 
            return self.calc

        calcfn = self.calc_filename(seed,target,prefix=prefix,suffix=suffix)
        
        import NNCalculator
        import tensorflow as tf
        import numpy as np
        import os
        from ase.io import read
        
        # Backup for if calculator is loaded before atoms available
        if atoms is None:
            atoms = read(f"{prefix}{seed}.xyz")
            try:
                atoms = read(f"{prefix}{seed}_gs_opt.xyz")
            except:
                try:
                    print(f"# Loading {prefix}{seed}.xyz")
                    atoms = read(f"{prefix}{seed}.xyz")
                except:
                    raise Exception(f"# Could not find initial geometry .xyz file for {seed}")

        # Find checkpoint file for calculator            
        log_dir=f"{calcfn}/best"
        checkpoint = tf.train.latest_checkpoint(log_dir)
        if checkpoint is None:
            self.calc_params = None
            raise Exception(f"Checkpoint for {calcfn} not found in current directory {os.getcwd()}")
        print(f'# Loading Calculator from: {log_dir}, {checkpoint} with args: {self.load_args}',flush=True)
        self.calc = NNCalculator.NNCalculator(checkpoint,atoms,**self.load_args)
        #self.calc2 = NNCalculator.NNCalculator(checkpoint,atoms,nn_in=self.calc.nn,sess_in=self.calc.sess,**self.load_args)

        # Load atoms trajectory and calculate initial value of atom energy
        from ase.io import Trajectory
        from esteem.trajectories import atom_energy,atom_energies
        atom_traj_name = f'{prefix}{seed}_atoms_{suffix}.traj'
        try:
            atom_traj = Trajectory(atom_traj_name)
        except Exception as e:
            print(f"Could not load atom_traj file: {atom_traj_name} in {os.getcwd()}")
            raise e
        self.atom_energies = atom_energies(atom_traj)
        self.atom_e = atom_energy(atoms,self.atom_energies)

        return self.calc

    def traj_to_npz(self,seed,trajfile,suffix):
        from ase.io import Trajectory
        from esteem import trajectories
        atom_traj = Trajectory(f'{seed}_atoms_{suffix}.traj')
        traj = Trajectory(trajfile)
        trajoutfile = trajfile.replace(".traj","_atsub.traj")
        trajout = Trajectory(trajoutfile,'w')
        trajectories.subtract_atom_energies_from_traj(traj,atom_traj,trajout)
        trajout.close()
        traj.close()

        traj = Trajectory(trajoutfile)
        length = len(traj)
        max_at = max([len(a) for a in traj])
        N = np.zeros((length,), dtype=int) #number of atoms
        E = np.zeros(length) #energy
        Q = np.zeros(length) #total charge
        D = np.zeros((length,3)) #dipole moment vector
        Z = np.zeros((length,max_at)) #nuclear charge
        R = np.zeros((length,max_at,3)) #cartesian coordinates
        F = np.zeros((length,max_at,3)) #forces
        for i,a in enumerate(traj):
            try:
                e_raw  = a.get_potential_energy()
                charge = a.get_initial_charges()
                N[i] = len(a)
                E[i] = e_raw
                Q[i] = np.sum(charge)
                D[i] = a.get_dipole_moment()
                Z[i,0:len(a)] = a.get_atomic_numbers()
                R[i,0:len(a)] = a.get_positions()
                F[i,0:len(a)] = a.get_forces()
            except Exception as e:
                print(f'Error while converting frame {i}:')
                raise e
                
        outfilename = trajfile.replace(".traj",".npz")
        np.savez(outfilename, N=N, E=E, Q=Q, D=D, Z=Z, R=R, F=F)
        return outfilename, len(traj)
   
    def reset_loss(self,seed,prefix="",suffix="",target=None):
        """
        Runs training for PhysNet model using an input trajectory as training points

        seed: str
        
        target: int

        suffix: str
        
        prefix: str
        
        """

        from os import path

        # Not sensible to try training in other directory than current, so prefix is
        # suppressed here but used elsewhere (eg for retrieving trajs)
        label = self.calc_filename(seed,target,prefix="",suffix=suffix)
        
        best_loss_file = f"{label}/best/best_loss.npz"
        if path.isfile(best_loss_file):
            loss_file   = np.load(best_loss_file)                
            best_loss   = loss_file["loss"].item()
            best_emae   = loss_file["emae"].item()
            best_ermse  = loss_file["ermse"].item()
            best_fmae   = loss_file["fmae"].item()
            best_frmse  = loss_file["frmse"].item()
            best_qmae   = loss_file["qmae"].item()
            best_qrmse  = loss_file["qrmse"].item()
            best_dmae   = loss_file["dmae"].item()
            best_drmse  = loss_file["drmse"].item()
            best_step   = loss_file["step"].item()
            print(f"# {best_loss_file} previously contained")
            for f in loss_file:
                print(f"# {f}={loss_file[f]}")
            print(f"# Re-initializing {best_loss_file} to infinity")

            best_loss  = np.Inf #initialize best loss to infinity
            best_emae  = np.Inf
            best_ermse = np.Inf
            best_fmae  = np.Inf
            best_frmse = np.Inf
            best_qmae  = np.Inf
            best_qrmse = np.Inf
            best_dmae  = np.Inf
            best_drmse = np.Inf
            # best_step  = 0. # Not changing best_step
            np.savez(best_loss_file, loss=best_loss, emae=best_emae,   ermse=best_ermse,
                                                     fmae=best_fmae,   frmse=best_frmse,
                                                     qmae=best_qmae,   qrmse=best_qrmse,
                                                     dmae=best_dmae,   drmse=best_drmse,
                                                     step=best_step)
        else:
            print("Warning: reset_loss == True but best_loss_file {best_loss_file} was not present")

    def train(self,seed,prefix="",suffix="",trajfile="",target=None,restart=False,**kwargs):
        """
        Runs training for PhysNet model using an input trajectory as training points

        seed: str
        
        target: int

        suffix: str

        trajfile: str

        restart: bool

        kwargs: dict

        """

        # Not sensible to try training in other directory than current, so prefix is
        # suppressed here but used elsewhere (eg for retrieving trajs)
        label = self.calc_filename(seed,target,prefix="",suffix=suffix)

        # Sort out optional arguments to see if any overrides to defaults have been supplied
        from copy import deepcopy
        import os
        train_args = deepcopy(self.train_args)
        for kw in train_args:
            if kw in kwargs:
                train_args[kw] = kwargs[kw]
        
        print(f'Converting trajectory {trajfile} to npz format')
        npzfile, ntraj = self.traj_to_npz(seed,trajfile,suffix)
        train_args['dataset'] = npzfile
        valid_frac = 0.1 # NB: Hardcoded to 10%
        train_args['num_train'] = int(ntraj*(1.0-valid_frac))
        train_args['num_valid'] = int(ntraj*valid_frac)
        train_args['restart'] = label
        if restart:
            if not os.path.exists(label):
                raise Exception(f'Path {label} not found for calculator restart')
            print(f'Continuing with run found in {label}')
        
        # Write config.txt
        import sys
        store_argv = sys.argv
        #config_file = f'{label}_config.txt'
        #print(f'Writing PhysNet configuration to {config_file}')
        #with open(config_file,"w") as f:
        #    for kw in train_args:
        #        f.write(f'--{kw}={train_args[kw]}\n')
        #sys.argv = ['train.py',config_file]
        
        sys.argv = ['train.py']
        for kw in train_args:
            sys.argv.append(f'--{kw}')
            sys.argv.append(f'{train_args[kw]}')
        
        print(f'# Training PhysNet model using trajectory {npzfile} with parameters:')
        print('#',train_args)
        import train
        sys.argv = store_argv
    
    def traj_write(self,atoms,traj):
        kw = {'dipole': atoms.get_dipole_moment(),
              'charges': atoms.get_charges(),
              'energy': atoms.get_potential_energy(),
              'forces': atoms.get_forces()}
        traj.write(atoms,**kw)
        
    def process_dynamics(self,dynamics,model,mdseed,traj_write,md_timestep,temp):

        from ase.md import Langevin, npt
        from ase.io import Trajectory

        # Check for existing settings, retain them if present
        if hasattr(dynamics,"new_traj"):
            new_traj = dynamics.new_traj
        else:
            new_traj = True
        if hasattr(dynamics,"friction"):
            friction = dynamics.friction
        else:
            friction = 0.01
        if hasattr(dynamics,"type"):
            dyn_type = dynamics.type
        else:
            dyn_type = "LANG"
            
        # Set up new dynamics objects, if required, otherwise retain existing ones
        if dyn_type=="LANG":
            if type(dynamics)!=Langevin:
                dynamics = Langevin(model, timestep=md_timestep, temperature_K=temp, friction=friction)
            else: # in case they have changed
                dynamics.set_timestep(md_timestep)
                dynamics.set_friction(friction)
                dynamics.set_temperature(temperature_K=temp)
        if dyn_type=="NPT" and type(dynamics)!=npt.NPT:
            ttime=25*units.fs
            pfactor = 1.06e9*(units.J/units.m**3)*ttime**2 # Bulk modulus for ethanol
            dynamics = npt.NPT(model, timestep=md_timestep, temperature_K=temp, externalstress=0,
                               pfactor=pfactor,ttime=ttime)

        # Copy in extra info stored in dynamics objects, for later retrieval
        dynamics.new_traj = new_traj
        dynamics.friction = friction
        dynamics.type = dyn_type
        
        # Handle attachment of trajectory
        if new_traj:
            if hasattr(dynamics,'traj'):
                dynamics.traj.close()
                dynamics.observers = []
            dynamics.traj = Trajectory(mdseed+".traj", 'w', model)
            dynamics.attach(traj_write, interval=1, atoms=model, traj=dynamics.traj)
            dynamics.new_traj = False

        return dynamics

    def run_md(self,model,mdseed,calc_params,md_steps,md_timestep,superstep,temp,
                 solvent=None,charge=0,restart=False,readonly=False,constraints=None,dynamics=None,
                 continuation=None):
        """
        Runs a Molecular Dynamics calculation with the PhysNet ASE calculator.

        model: ASE Atoms

        seed: str
        
        calc_params: dict
        
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

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']

        # Load the PhysNet Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,atoms=model)
        model.calc = calc_ml

        # Initialise velocities if this is first step, otherwise inherit from model
        if np.all(model.get_momenta() == 0.0):
            MaxwellBoltzmannDistribution(model,temperature_K=temp)

        # For each ML superstep, remove C.O.M. translation and rotation    
        #Stationary(model)
        #ZeroRotation(model)
        #print(f'constraints: {model.constraints}')

        if readonly:
            model = read(mdseed+".xyz") # Read final image
            model.calc = calc_ml
            model.get_potential_energy() # Recalculate energy for final image
            return None
        else:
            dynamics = self.process_dynamics(dynamics,model,mdseed,self.traj_write,md_timestep,temp)
            dynamics.run(md_steps)
            return dynamics

    # Define a PhysNet geometry optimisation function
    def geom_opt(self,model,seed,calc_params,driver_tol='default',
                 solvent=None,charge=0,spin=0,writeonly=False,readonly=False,continuation=False,cleanup=False,
                 traj=None):
        """
        Runs a geometry optimisation calculation with the PhysNet ASE calculator

        model: ASE Atoms

        seed: str
        
        calc_params: dict

        dummy: str

        driver_tol:

        target: int or None

        solvent: str or None

        readonly: bool
        """
        from ase.io import Trajectory
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']
        
        from ase.optimize import BFGS
        from ase.units import Hartree, Bohr

        # Load the appropriate PhysNet Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,atoms=model)
        model.calc = calc_ml

        # Create instance of BFGS optimizer, run it and return results
        dyn = BFGS(model,trajectory=traj)

        # tolerances corresponding to NWChem settings
        fac=1
        if driver_tol=='default':
            fmax = 0.001*fac
        if driver_tol=='loose':
            fmax = 0.00450*fac
        if driver_tol=='tight':
            fmax = 0.000015*fac
        
        dyn.log = self.log
        dyn.run(fmax=fmax,steps=1000)

        if hasattr(calc_ml,'results'):
            calc_ml.results['dipole'] = model.get_dipole_moment()
            
        e_calc = model.get_potential_energy()
        if isinstance(e_calc,np.ndarray):
            e_calc = np.float64(e_calc[0])

        return e_calc, model.get_forces(), model.get_positions()

    def log(self): 
        pass

    def freq(self,model_opt,seed,calc_params,solvent=None,charge=0,
             temp=300,writeonly=False,readonly=False,continuation=False,
             summary=True,cleanup=True):
        """
        Runs a Vibrational Frequency calculation with the PhysNet ASE calculator
        
        model_opt: ASE Atoms

        seed: str
        
        suffix: str

        dummy: str

        driver_tol:

        target: int or None

        solvent: str or None
        
        temp: float

        readonly: bool
        """

        from ase.vibrations import Vibrations, Infrared
        from ase.constraints import FixAtoms
        from os import getcwd
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']

        # Load the appropriate PhysNet  Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,atoms=model_opt)
        model_opt.calc = calc_ml
        
        indices = list(range(len(model_opt)))
        if len(model_opt.constraints)>0:
            constraint_str = "%geom Constraints\n"
            for constr in model_opt.constraints:
                if isinstance(constr,FixAtoms):
                    indices = [i for i in indices if i not in constr.index]

        # Create instance of Vibrations class, run it and return results
        #vib = Vibrations(model_opt,name=self.calc_filename(seed,target,prefix=prefix,suffix=suffix))
        #vib.run()
        #freqs = vib.get_frequencies()
        #vib.summary()
        #vib.clean()
        calcname = self.calc_filename(seed,target,prefix="",suffix=suffix)
        #print(getcwd)
        ir = Infrared(model_opt,indices=indices,name=calcname)
        ir.run()
        #if summary:
        #    ir.summary()
        #ir.write_spectra(out=ir.name+'_ir_spectrum.dat',start=0,end=4000,width=20)
        #ir.clean()
        
        #print(freqs)
        return ir

    def singlepoint(self,model,seed,calc_params,solvent=None,charge=0,spin=0,forces=False,dipole=True,
                    readonly=False,continuation=False,cleanup=True):
        """
        Runs a singlepoint calculation with the PhysNet ASE calculator

        model: ASE Atoms

        seed: str
        
        suffix: str

        dummy: str

        target: int or None

        solvent: str or None

        readonly: bool
        """
        
        from esteem.trajectories import atom_energy
        from ase.calculators.singlepoint import SinglePointCalculator
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']
        
        # Load the appropriate PhysNet Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,atoms=model)
        model.calc = calc_ml
        e_calc = model.get_potential_energy()
        if isinstance(e_calc,np.ndarray):
            e_calc = np.float64(e_calc[0])
        if forces:
            f_calc = model.get_forces()
        if hasattr(calc_ml,'results'):
            d_calc = model.get_dipole_moment()
            calc_ml.results['dipole'] = d_calc
        if hasattr(self,'store_atom_e'):
            store_atom_e = self.store_atom_e
        else:
            store_atom_e = False
        if self.update_atom_e or store_atom_e:
            self.atom_e = atom_energy(model,self.atom_energies)
        if store_atom_e:
            calc_ml = SinglePointCalculator(model)
            calc_ml.results['energy'] = e_calc + self.atom_e
            if forces:
                calc_ml.results['forces'] = f_calc
            calc_ml.results['dipole'] = d_calc
            #calc_ml.energy = e_calc + self.atom_e

        if forces:
            if dipole:
                return e_calc + self.atom_e, f_calc, d_calc, calc_ml
            else:
                return e_calc + self.atom_e, f_calc, calc_ml
        else:
            if dipole:
                return e_calc + self.atom_e, d_calc, calc_ml
            else:
                return e_calc + self.atom_e, calc_ml


# In[ ]:




