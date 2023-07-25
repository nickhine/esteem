#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines the AMPWrapper Class"""

import numpy as np

class AMPWrapper():
    """
    Sets up and runs the AMP Calculator (via ASE) to use a neural-network representation of a
    potential energy surface.
    """

    def __init__(self):
        """Sets up instance attributes for AMPWrapper """
        self.calc_ext = ".amp"
        self.log_ext = "-log.txt"
    
    def calc_filename(self,seed,target,prefix='',suffix=''):
        if target is None or target == 0:
            calcfn = seed+"_gs_"+suffix
        else:
            calcfn = args.seed+"_es"+str(target)+"_"+suffix
            
        calcfn = prefix + calcfn
            
        return calcfn
    
    def load(self,seed,target=None,prefix="",suffix="",**kwargs):
        """
        Loads an existing AMP Calculator

        seed: str
        
        target: int

        suffix: str

        kwargs: dict
            other keywords to pass to AMP.load
        """

        from amp import Amp

        calcfn = self.calc_filename(seed,target,prefix=prefix,suffix=suffix)
        calc_ml = Amp.load(calcfn+self.calc_ext,**kwargs)
        calc_ml.label = calcfn
        calc_ml.dblabel = calcfn
        return calc_ml

    def train(self,seed,prefix="",suffix="",trajfile="",target=None,restart=False,**kwargs):
        """
        Runs training for AMP ASE calculator using an input trajectory as training points

        seed: str
        
        target: int

        suffix: str

        trajfile: str

        restart: bool

        kwargs: dict

        """

        from amp.descriptor.gaussian import Gaussian
        from amp.model.neuralnetwork import NeuralNetwork
        from amp.model import LossFunction
        from amp.utilities import Annealer

        label = self.calc_filename(seed,target,prefix=prefix,suffix=suffix)

        # Sort out optional arguments according to which routine they need to go to
        desc_kw = {key: kwargs.pop(key) for key in {'cutoff'} if key in kwargs}
        model_kw = {key: kwargs.pop(key) for key in {'hiddenlayers'} if key in kwargs}
        calc_kw = {key: kwargs.pop(key) for key in {'cores'} if key in kwargs}
        convergence = {key: kwargs.pop(key) for key in {'energy_rmse','force_rmse',
                       'energy_maxresid','force_maxresid'} if key in kwargs}
        lossfn_kw = {key: kwargs.pop(key) for key in {'force_coefficient'} if key in kwargs}
        annealer_kw = {key: kwargs.pop(key) for key in {'Tmax','Tmin','steps'} if key in kwargs}

        # Anything left is unrecognised
        for key in kwargs:
            print(f'WARNING: Unrecognised keyword {key}')

        # Toggle whether to retrain from scratch or load existing model
        if not restart:
            descriptor = Gaussian(**desc_kw)
            model = NeuralNetwork(**model_kw)
            calc_ml = Amp(descriptor,label=label,model=model,**calc_kw)
        else:
            calc_ml = self.load(seed,target,prefix=prefix,suffix=suffix,**calc_kw)
            calc_ml.label = label
            calc_ml.dblabel = label

        # Define convergence criteria for the Loss Function
        calc_ml.model.lossfunction = LossFunction(convergence=convergence,**lossfn_kw)

        # Optionally run the annealer
        if 'steps' in annealer_kw:
            steps = annealer_kw['steps']
            if steps is not None and steps>0:
                Annealer(calc=calc_ml, images=trajfile, **annealer_kw)

        # Now train the calculator
        calc_ml.train(images=trajfile,overwrite=True)
        return calc_ml

    def run_md(self,model,mdseed,calc_params,md_steps,md_timestep,superstep,temp,
               solvent=None,restart=False,readonly=False,constraints=None,dynamics=None,
               continuation=None):
        """
        Runs a Molecular Dynamics calculation with the AMP ASE calculator.

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
        from ase.md import VelocityVerlet, Langevin
        from ase.io import write
        from ase import units
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']
        
        # Load the appropriate AMP Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix)
        model.calc = calc_ml

        # Initialise velocities if this is first step, otherwise inherit from model
        if np.all(model.get_momenta() == 0.0):
            MaxwellBoltzmannDistribution(model, temp * units.kB)

        # For each ML superstep, remove C.O.M. translation and rotation    
        #Stationary(model)
        #ZeroRotation(model)
        #print(f'constraints: {model.constraints}')
        
        if readonly:
            model = read(mdseed+".xyz") # Read final image
            model.calc = calc_ml
            model.get_potential_energy() # Recalculate energy for final image
        else:
            if dynamics is None:
                dynamics = Langevin(model, timestep=md_timestep, temperature=units.kB * temp, friction=0.002)
            dynamics.run(md_steps)
            print("MD: %5d %10.6f" % (superstep, model.get_potential_energy()))
            if "forces" in model.arrays:
                del model.arrays["forces"]
            write(mdseed+".xyz",model)

    # Define an AMP geometry optimisation function
    def geom_opt(self,model,seed,calc_params,driver_tol='default',
                 solvent=None,readonly=False):
        """
        Runs a geometry optimisation calculation with the AMP ASE calculator

        model: ASE Atoms

        seed: str
        
        calc_params: dict

        dummy: str

        driver_tol:

        target: int or None

        solvent: str or None

        readonly: bool
        """
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']
        
        from ase.optimize import BFGS
        from ase.units import Hartree, Bohr

        # Load the appropriate AMP Calculator
        calc_ml = self.load(seed,target,prefix=prefix,suffix=suffix)
        model.calc = calc_ml

        # Create instance of BFGS optimizer, run it and return results
        dyn = BFGS(model)

        # tolerances corresponding to NWChem settings
        fac=1
        if driver_tol=='default':
            fmax = 0.00045*fac
        if driver_tol=='loose':
            fmax = 0.00450*fac
        if driver_tol=='tight':
            fmax = 0.000015*fac
            
        dyn.run(fmax=fmax)

        return model.get_potential_energy(), model.get_forces(), model.get_positions()

    def freq(self,model_opt,seed,calc_params,solvent=None,charge=0,
             temp=300,writeonly=False,readonly=False,continuation=False,cleanup=True):
        """
        Runs a Vibrational Frequency calculation with the AMP ASE calculator
        
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

        from ase.vibrations import Vibrations
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']

        # Load the appropriate AMP Calculator
        calc_ml = self.load(seed,target,prefix=prefix,suffix=suffix)
        model_opt.calc = calc_ml

        # Create instance of Vibrations class, run it and return results
        vib = Vibrations(model_opt,name=self.calc_filename(seed,target,prefix=prefix,suffix=suffix))
        vib.run()
        freqs = vib.get_frequencies()
        vib.summary()
        vib.clean()
        
        print(freqs)
        return freqs

    def singlepoint(self,model,seed,suffix,dummy,
                    target=None,solvent=None,readonly=False):
        """
        Runs a singlepoint calculation with the AMP ASE calculator

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

        # Load the appropriate AMP Calculator
        calc_ml = self.load(seed,target,prefix=prefix,suffix=suffix)
        model_opt.calc = calc_ml
        return model.get_potential_energy(), model.get_forces()

