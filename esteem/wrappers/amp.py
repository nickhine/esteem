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

    def calc_filename(self, seed, target, prefix='', suffix=''):
        if target is None or target == 0:
            calcfn = seed + "_gs_" + suffix
        else:
            calcfn = seed + "_es" + str(target) + "_" + suffix

        calcfn = prefix + calcfn

        return calcfn

    def load(self, seed, target=None, prefix="", suffix="", **kwargs):
        """
        Loads an existing AMP Calculator from ``{prefix}{seed}_{targ}_{suffix}.amp``

        seed: str

        target: int

        prefix: str

        suffix: str

        kwargs: dict
            other keywords to pass to Amp.load

        *Returns*: the Amp calculator
        """

        from amp import Amp

        calcfn = self.calc_filename(seed, target, prefix=prefix, suffix=suffix)
        calc_ml = Amp.load(calcfn + self.calc_ext, **kwargs)
        calc_ml.label = calcfn
        calc_ml.dblabel = calcfn
        return calc_ml

    def train(self, seed, prefix="", suffix="", trajfile="", target=None, restart=False, **kwargs):
        """
        Runs training for AMP ASE calculator using an input trajectory as training points

        seed: str

        prefix: str

        suffix: str

        trajfile: str
            Filename of the training trajectory

        target: int

        restart: bool
            If True, continue training the existing calculator instead of starting afresh

        kwargs: dict
            ``cutoff`` (for the Gaussian descriptor), ``hiddenlayers`` (for the neural
            network), ``cores`` (for Amp), ``energy_rmse``, ``force_rmse``,
            ``energy_maxresid``, ``force_maxresid`` (convergence criteria),
            ``force_coefficient`` (for the loss function) and ``Tmax``, ``Tmin``,
            ``steps`` (for the Annealer, which is run first if ``steps`` > 0).
            Other keywords are ignored with a warning.

        *Returns*: the trained Amp calculator
        """

        from amp import Amp
        from amp.descriptor.gaussian import Gaussian
        from amp.model.neuralnetwork import NeuralNetwork
        from amp.model import LossFunction
        from amp.utilities import Annealer

        label = self.calc_filename(seed, target, prefix=prefix, suffix=suffix)

        # Sort out optional arguments according to which routine they need to go to
        desc_kw = {key: kwargs.pop(key) for key in {'cutoff'} if key in kwargs}
        model_kw = {key: kwargs.pop(key) for key in {'hiddenlayers'} if key in kwargs}
        calc_kw = {key: kwargs.pop(key) for key in {'cores'} if key in kwargs}
        convergence = {key: kwargs.pop(key) for key in {'energy_rmse', 'force_rmse',
                       'energy_maxresid', 'force_maxresid'} if key in kwargs}
        lossfn_kw = {key: kwargs.pop(key) for key in {'force_coefficient'} if key in kwargs}
        annealer_kw = {key: kwargs.pop(key) for key in {'Tmax', 'Tmin', 'steps'} if key in kwargs}

        # Anything left is unrecognised
        for key in kwargs:
            print(f'WARNING: Unrecognised keyword {key}')

        # Toggle whether to retrain from scratch or load existing model
        if not restart:
            descriptor = Gaussian(**desc_kw)
            model = NeuralNetwork(**model_kw)
            calc_ml = Amp(descriptor, label=label, model=model, **calc_kw)
        else:
            calc_ml = self.load(seed, target, prefix=prefix, suffix=suffix, **calc_kw)
            calc_ml.label = label
            calc_ml.dblabel = label

        # Define convergence criteria for the Loss Function
        calc_ml.model.lossfunction = LossFunction(convergence=convergence, **lossfn_kw)

        # Optionally run the annealer
        if 'steps' in annealer_kw:
            steps = annealer_kw['steps']
            if steps is not None and steps > 0:
                Annealer(calc=calc_ml, images=trajfile, **annealer_kw)

        # Now train the calculator
        calc_ml.train(images=trajfile, overwrite=True)
        return calc_ml

    def run_md(self, model, mdseed, calc_params, md_steps, md_timestep, superstep, temp,
               solvent=None, restart=False, readonly=False, constraints=None, dynamics=None,
               continuation=None):
        """
        Runs a Molecular Dynamics calculation with the AMP ASE calculator.

        Runs Langevin dynamics (friction 0.002) unless ``dynamics`` is given, and writes
        the final structure to ``{mdseed}.xyz``.

        model: ASE Atoms

        mdseed: str

        calc_params: dict
            Calculator parameters: ``calc_seed``, ``target``, ``calc_suffix`` and ``calc_prefix``

        md_steps: int

        md_timestep: float
            Timestep in ASE time units

        superstep: int
            Index of this run, for printing

        temp: float
            Temperature in K

        readonly: bool
            If True, recalculate the energy of the final image in ``{mdseed}.xyz``
            instead of running MD

        dynamics:
            ASE dynamics object to continue, or None

        The other arguments are ignored. Returns None.
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
        calc_ml = self.load(calc_seed, target, prefix=prefix, suffix=suffix)
        model.calc = calc_ml

        # Initialise velocities if this is first step, otherwise inherit from model
        if np.all(model.get_momenta() == 0.0):
            MaxwellBoltzmannDistribution(model, temp * units.kB)

        # For each ML superstep, remove C.O.M. translation and rotation
        # Stationary(model)
        # ZeroRotation(model)
        # print(f'constraints: {model.constraints}')

        if readonly:
            from ase.io import read
            model = read(mdseed + ".xyz")  # Read final image
            model.calc = calc_ml
            model.get_potential_energy()  # Recalculate energy for final image
        else:
            if dynamics is None:
                dynamics = Langevin(model, timestep=md_timestep, temperature=units.kB * temp, friction=0.002)
            dynamics.run(md_steps)
            print("MD: %5d %10.6f" % (superstep, model.get_potential_energy()))
            if "forces" in model.arrays:
                del model.arrays["forces"]
            write(mdseed + ".xyz", model)

    # Define an AMP geometry optimisation function
    def geom_opt(self, model, seed, calc_params, driver_tol='default',
                 solvent=None, readonly=False):
        """
        Runs a BFGS geometry optimisation calculation with the AMP ASE calculator

        model: ASE Atoms

        seed: str
            Seed name used to load the calculator (``calc_params['calc_seed']`` is not used)

        calc_params: dict
            Calculator parameters: ``target``, ``calc_suffix`` and ``calc_prefix``

        driver_tol: str
            Force tolerance: 'default' (0.00045 eV/Ang), 'loose' (0.0045) or 'tight' (0.000015)

        solvent: str or None
            Ignored

        readonly: bool
            Ignored

        *Returns*: energy (eV), forces (eV/Ang) and positions (Ang) of the optimised structure
        """

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']

        from ase.optimize import BFGS
        from ase.units import Hartree, Bohr

        # Load the appropriate AMP Calculator
        calc_ml = self.load(seed, target, prefix=prefix, suffix=suffix)
        model.calc = calc_ml

        # Create instance of BFGS optimizer, run it and return results
        dyn = BFGS(model)

        # tolerances corresponding to NWChem settings
        fac = 1
        if driver_tol == 'default':
            fmax = 0.00045 * fac
        if driver_tol == 'loose':
            fmax = 0.00450 * fac
        if driver_tol == 'tight':
            fmax = 0.000015 * fac

        dyn.run(fmax=fmax)

        return model.get_potential_energy(), model.get_forces(), model.get_positions()

    def freq(self, model_opt, seed, calc_params, solvent=None, charge=0,
             temp=300, writeonly=False, readonly=False, continuation=False, cleanup=True):
        """
        Runs a Vibrational Frequency calculation with the AMP ASE calculator, using
        ASE's ``Vibrations`` class

        model_opt: ASE Atoms

        seed: str
            Seed name used to load the calculator (``calc_params['calc_seed']`` is not used)

        calc_params: dict
            Calculator parameters: ``target``, ``calc_suffix`` and ``calc_prefix``

        The other arguments are ignored.

        *Returns*: array of the vibrational frequencies (cm^-1)
        """

        from ase.vibrations import Vibrations

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        prefix = calc_params['calc_prefix']

        # Load the appropriate AMP Calculator
        calc_ml = self.load(seed, target, prefix=prefix, suffix=suffix)
        model_opt.calc = calc_ml

        # Create instance of Vibrations class, run it and return results
        vib = Vibrations(model_opt, name=self.calc_filename(seed, target, prefix=prefix, suffix=suffix))
        vib.run()
        freqs = vib.get_frequencies()
        vib.summary()
        vib.clean()

        print(freqs)
        return freqs

    def singlepoint(self, model, seed, suffix, dummy,
                    target=None, solvent=None, readonly=False):
        """
        Runs a singlepoint calculation with the AMP ASE calculator

        Unlike the other ML wrappers, this takes the calculator seed, suffix and target
        as separate arguments rather than a ``calc_params`` dict.

        model: ASE Atoms

        seed: str

        suffix: str

        dummy: str
            Ignored

        target: int or None

        solvent: str or None
            Ignored

        readonly: bool
            Ignored

        *Returns*: energy (eV) and forces (eV/Ang)
        """

        # Load the appropriate AMP Calculator
        calc_ml = self.load(seed, target, suffix=suffix)
        model.calc = calc_ml
        return model.get_potential_energy(), model.get_forces()
