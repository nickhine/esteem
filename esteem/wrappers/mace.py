#!/usr/bin/env python
# coding: utf-8

"""Defines the MACEWrapper Class"""
import numpy as np
from ase.io import read, write
from os import path


class MACEWrapper():
    """
    Sets up, trains and runs a MACE Neural Network Calculator to represent a
    potential energy surface.

    ``MACEWrapper(xmace=True)`` uses X-MACE (the ``xmace`` package) instead of MACE.
    Setting the ``foundation_model`` attribute to a (family, model) tuple, eg
    ``('polar', 'polar-1-m')``, makes ``load`` return that MACE foundation model
    instead of a trained one. The ``charge`` attribute (default 0) is the total
    charge used for charge-aware models when the task does not supply one.
    """

    from mace import tools
    default_train_args = tools.build_default_arg_parser().parse_args(["--name", "dummy", "--train_file", "train.xyz"])

    def __init__(self, xmace=False, **kwargs):
        """Sets up instance attributes for MACEWrapper

        xmace: bool
            Use X-MACE (the xmace package) instead of MACE: one model for several
            electronic states, trained with a target dict such as {0: 'gs', 1: 'es1'},
            whose head names select the state to evaluate

        kwargs: dict
            Ignored. Change training arguments through the ``train_args`` attribute.
        """
        from copy import deepcopy
        self.xmace = xmace
        if xmace:
            from xmace import tools as xmace_tools
            self.train_args = xmace_tools.build_default_arg_parser().parse_args(
                ["--name", "dummy", "--train_file", "train.xyz"])
            self.train_args.max_num_epochs = -1
            self.train_args.model = 'ExcitedMACE'
            # energies and forces only: EnergyNacsDipoleMAE needs dipole (and NAC) data
            self.train_args.error_table = 'PerAtomRMSE'
            self.train_args.restart_latest = True
        else:
            self.train_args = deepcopy(self.default_train_args)
            self.train_args.max_num_epochs = -1
            self.train_args.model = 'EnergyDipolesMACE'
            self.train_args.loss = 'energy_forces_dipole'
            self.train_args.error_table = 'EnergyDipoleRMSE'
            self.train_args.restart_latest = True

        # Make a set of default loading arguments by copying in training arguments
        self.load_args = {}
        self.calc_ext = ""
        self.log_ext = ""
        self.calc = None
        self.calc_params = None
        self.atom_e = 0.0
        self.atom_energies = {}
        self.atoms_on_load = None
        self.output = "mace"
        # Use a MACE foundation model instead of a trained one: (family, model), eg
        # ('polar', 'polar-1-m') loads mace.calculators.mace_polar('polar-1-m')
        self.foundation_model = None
        self.foundation_kwargs = {}
        # Total charge written to atoms.info['charge'], for charge-aware models (eg MACE-POLAR),
        # used when the task does not supply one
        self.charge = 0

    def calc_filename(self, seed, target, prefix='', suffix=''):
        if isinstance(target, dict):
            calcfn = seed + "_"
            for targ in target:
                if targ == "diff":
                    continue
                calcfn += "es" + str(targ) if targ != 0 else "gs"
            calcfn += "_" + suffix
        else:
            if target is None or target == 0:
                calcfn = f"{seed}_gs_{suffix}"
            elif isinstance(target, str):
                calcfn = f"{seed}_{target}_{suffix}"
            else:
                calcfn = f"{seed}_es{str(target)}_{suffix}"

        calcfn = prefix + calcfn

        return calcfn

    def load(self, seed, target=None, prefix="", suffix="", dir_suffix=""):
        """
        Loads an existing MACE Calculator, or a committee of them

        Calculators are cached, so loading the same parameters again returns the
        already-loaded calculator(s). If ``foundation_model`` is set, the foundation
        model is returned and the other arguments are ignored.

        seed: str
            Seed name of the calculator

        target: int, str, dict or list
            Target state (eg 0 or 1), a dict of targets for a multi-head or X-MACE
            model (eg ``{0: 'gs', 1: 'es1'}``), or a list of targets to load one
            calculator for each

        prefix: str
            Prefix to the directory containing the calculator

        suffix: str or dict
            Suffix of the calculator name. A dict, mapping each calculator suffix to
            the random seed it was trained with, loads a committee.

        dir_suffix: str
            Suffix of the directory ``{seed}_{targ}_{dir_suffix}`` containing the model file(s)

        *Returns*: a MACECalculator, or a list of them for a committee or a list of targets
        """

        # Check against or store previous calculator parameters
        # if self.calc_params is not None:
        #    if ((self.calc_params['target'] != target) or
        #        (self.calc_params['calc_suffix'] != suffix) or
        #        (self.calc_params['calc_seed'] != seed)):
        #        raise Exception('Attempted change of calculator parameters for previously-loaded wrapper. Not supported.')
        if self.calc_params is None:
            self.calc_params = {'target': target, 'calc_prefix': prefix,
                                'calc_suffix': suffix, 'calc_seed': seed}

        if self.foundation_model is not None:
            if self.calc is None:
                import mace.calculators
                family, model = self.foundation_model
                self.calc = getattr(mace.calculators, f'mace_{family}')(model=model, **self.foundation_kwargs)
            return self.calc

        # One calculator (or committee) is kept for each set of parameters, so that one wrapper
        # can evaluate several targets in turn (eg S1 then S0 at each snapshot)
        if not hasattr(self, 'loaded_calcs'):
            self.loaded_calcs = {}
        key = (seed, repr(target), prefix, repr(suffix), dir_suffix)
        if key in self.loaded_calcs:
            self.calc = self.loaded_calcs[key]
            return self.calc
        from mace.calculators import MACECalculator

        # Find checkpoint file(s) for calculator
        calctarget = target
        if isinstance(suffix, dict) or isinstance(target, list):
            self.calc = []
        suffixes = suffix if isinstance(suffix, dict) else {suffix: self.train_args.seed}
        if isinstance(calctarget, list):
            targets = calctarget
        if isinstance(calctarget, dict):
            targets = ""
            for targ in calctarget:
                if targ == "diff":
                    continue
                targets += "es" + str(targ) if targ != 0 else "gs"
            targets = [targets]
        else:
            targets = [calctarget]
        targets = calctarget if isinstance(calctarget, list) else [calctarget]
        for suff in suffixes:
            for targ in targets:
                dirname = self.calc_filename(seed, targ, prefix=prefix, suffix=dir_suffix)
                calcfn = self.calc_filename(seed, targ, prefix="", suffix=suff)
                modelfile = self.find_best_model(dirname, calcfn, suffixes[suff])
                if self.xmace:
                    from xmace.calculators import MACECalculator as XMACECalculator
                    calc = XMACECalculator(modelfile, device="cuda", default_dtype='float64', model_type='MACE',
                                           n_energies=len(self.xmace_states(targ)))
                else:
                    calc = MACECalculator(modelfile, device="cuda", default_dtype='float64',
                                          model_type='EnergyDipoleMACE')
                if isinstance(suffix, dict) or isinstance(calctarget, list):
                    self.calc.append(calc)
                else:
                    self.calc = calc

        self.loaded_calcs[key] = self.calc
        return self.calc

    def traj_to_extxyz(self, trajfile, outfilename, use_json=False):
        """
        Converts an ASE Trajectory file to an extxyz format text file.
        Prefixes the data tags with ``REF_`` so that ASE does not overwrite them.

        trajfile: str or list
            Trajectory filename, or (with ``use_json``) a list of trajectories of the
            same frames, one per state

        outfilename: str
            Name of the extxyz file to write

        use_json: bool
            If True, write the energies, dipoles and forces of all the states in each
            frame as JSON strings, as X-MACE expects

        *Returns*: (outfilename, number of frames written)
        """

        from ase.io import Trajectory
        from ase.io.extxyz import write_xyz
        import json

        if not isinstance(trajfile, list):
            trajs = [Trajectory(trajfile)]
        else:
            trajs = [Trajectory(tr) for tr in trajfile]
            assert all([len(tr) == len(trajs[0]) for tr in trajs])
        with open(outfilename, "w") as f:
            for i, t in enumerate(trajs[0]):
                tp = t.copy()
                if not use_json:
                    tp.info["REF_energy"] = t.calc.results["energy"]
                    tp.info["REF_dipole"] = t.calc.results["dipole"]
                    # forces may be (natoms,3), or (ntargets,natoms,3) from a multi-target singlepoint
                    forces = np.asarray(t.calc.results["forces"])
                    tp.arrays["REF_forces"] = forces[0] if forces.ndim == 3 else forces
                else:
                    # all states in one frame, as X-MACE expects: energy (1, n_states),
                    # forces (n_atoms, n_states, 3)
                    en = np.array([traj[i].calc.results["energy"] for traj in trajs])
                    tp.info["REF_energy"] = "_JSON " + json.dumps([en.tolist()])
                    dip = np.stack([traj[i].calc.results["dipole"] for traj in trajs])
                    tp.info["REF_dipole"] = "_JSON " + json.dumps(dip.tolist())
                    fo = np.stack([traj[i].calc.results["forces"] for traj in trajs])
                    fo = fo.transpose(1, 0, 2)
                    tp.info["REF_forces"] = "_JSON " + json.dumps(fo.tolist())
                write_xyz(f, tp)
        return outfilename, len(trajs[0])

    def reset_loss(self, seed, prefix="", suffix="", target=None):
        """
        Placeholder for resetting the loss before further training. For MACE it
        does nothing, so a ``reset_loss`` request has no effect.

        seed: str

        target: int

        suffix: str

        prefix: str

        """

        # Not sensible to try training in other directory than current, so prefix is
        # suppressed here but used elsewhere (eg for retrieving trajs)
        label = self.calc_filename(seed, target, prefix="", suffix=suffix)

    def find_best_model(self, dirname, calcfn, seed_suffix):
        """
        MACE keeps changing its mind about the naming scheme for the finished model.
        So we loop over all the possibilities and try them all.
        We can probably remove this nonsense eventually but keep it for now

        dirname: str
            Directory containing the model
        calcfn: str
            Calculator name
        seed_suffix: str
            Random seed used in training, which MACE adds as ``_run-{seed}``

        *Returns*: the path of the first model file found, or None
        """
        best_model = None
        modelfiles = []
        for swastr in ["_stagetwo", "_swa", ""]:
            for dirstr in [f"{dirname}/", f"{dirname}/checkpoints/"]:
                for suffix in ["", f"_run-{seed_suffix}"]:
                    modelfiles.append(f"{dirstr}{calcfn}{suffix}{swastr}.model")
        for modelfile in modelfiles:
            found = False
            if path.exists(modelfile):
                found = True
                best_model = modelfile
                print(f'# Model file {modelfile} found')
                break
        if best_model is None:
            print(f'ERROR: No model found among: {modelfiles}')
        return best_model

    def train(self, seed, prefix="", suffix="", dir_suffix="", trajfile="", validfile=None, testfile=None,
              targets_in=None, restart=False, **kwargs):
        """
        Runs training for MACE model using input trajectories as training points

        Training runs in the directory ``{seed}_{targ}_{dir_suffix}``, which must contain
        the atom energies trajectory ``{seed}_atoms_{suffix}.traj``. It is skipped if a
        finished model already exists. Set training options through ``train_args``.

        seed: str
            Seed name of the calculator
        prefix: str
            Ignored: training always runs relative to the current directory
        suffix: str
            Suffix of the calculator name
        dir_suffix: str
            Suffix of the training directory
        trajfile: dict
            Training trajectory filename for each target, keyed by target string
            (the values of ``targets_in``, or ``str(targets_in)`` for a single target)
        validfile: dict or None
            Validation trajectory filenames, keyed as for ``trajfile``. If None, a
            validation fraction of 5% of the training data is used.
        testfile: dict or None
            Test trajectory filenames, keyed as for ``trajfile``. If None, the training
            data are also used as the test set.
        targets_in: int or dict
            Target state, or a dict such as ``{0: 'gs', 1: 'es1'}`` to train a multi-head
            model (or, with X-MACE, one model for all the states)
        restart: bool
            Ignored (MACE restarts from its latest checkpoint by default)
        kwargs: dict
            Ignored
        """

        # Not sensible to try training in other directory than current, so prefix is
        # suppressed here but used elsewhere (eg for retrieving trajs)
        dirname = self.calc_filename(seed, targets_in, prefix="", suffix=dir_suffix)

        # Sort out optional arguments to see if any overrides to defaults have been supplied
        from copy import deepcopy
        import os
        train_args = deepcopy(self.train_args)

        # Check if training already complete
        calcfn = self.calc_filename(seed, targets_in, prefix="", suffix=suffix)
        best_model = self.find_best_model(dirname, calcfn, train_args.seed)
        if best_model is not None:
            print(f'# Skipping training as finished model {best_model} already exists')
            print(f'# Delete this if training is intended to be restarted / extended')
            return

        use_json = False
        if not isinstance(targets_in, dict):
            # if we supplied a single target, convert to a dictionary
            targets = {targets_in: str(targets_in)}
            heads = None
        elif self.xmace or train_args.model in ("ExcitedMACE", "AutoencoderExcitedMACE"):
            # X-MACE: one model learns all the states, from one file with every state per frame
            targets = {targ: targets_in[targ] for targ in self.xmace_states(targets_in)}
            heads = None
            use_json = True
            train_args.n_energies = len(targets)
            print(f'# X-MACE model for {train_args.n_energies} states: {targets}')
        else:
            # if we supplied multiple targets set up heads dictionary
            targets = targets_in
            heads = {}
            for target in targets:
                targetstr = targets[target]
                heads[targetstr] = {}

        # Convert trajectories from .traj to .xyz format
        convtrajfile = {'train': trajfile}
        if validfile is not None:
            convtrajfile['valid'] = validfile
        else:  # assume we want to use a validation fraction of 5%
            if self.xmace:  # X-MACE tests valid_file is None, rather than its absence
                train_args.valid_file = None
            else:
                del train_args.valid_file
            if not hasattr(train_args, 'valid_fraction'):
                train_args.valid_fraction = 0.05
        if testfile is not None:
            convtrajfile['test'] = testfile
        for key in convtrajfile:
            if not use_json:
                trajfile_dict = convtrajfile[key]
                targets_loop = targets
            else:
                from esteem.trajectories import targstr
                all_targs = "".join(targstr(t) for t in targets)
                trajfile_dict = {all_targs: [convtrajfile[key][targets[target]] for target in targets]}
                targets_loop = {all_targs: all_targs}
            for target in targets_loop:
                targetstr = targets_loop[target]
                trajf = trajfile_dict[targetstr]
                print(f'# Converting trajectory {trajf} to extxyz format as {key} file for target {targetstr}')
                extxyzfile = self.calc_filename(seed, target, prefix=dirname + "/", suffix=suffix) + f"_{key}.xyz"
                extxyzfile, ntraj = self.traj_to_extxyz(trajf, extxyzfile, use_json=use_json)
                print(f'# Wrote {ntraj} frames to {extxyzfile} in extxyz format')
                extxyzfile = self.calc_filename(seed, target, prefix="", suffix=suffix) + f"_{key}.xyz"
                if heads is not None:
                    heads[targetstr][f'{key}_file'] = extxyzfile
                else:
                    setattr(train_args, f'{key}_file', extxyzfile)
                if key == 'train':
                    if 'test' not in convtrajfile:
                        train_args.test_file = extxyzfile
                    if self.train_args.max_num_epochs < 0:
                        train_args.max_num_epochs = int(round(200000 * train_args.batch_size / ntraj / 50) * 50)

        # Open atom_traj
        from ase.io import Trajectory
        atom_traj = Trajectory(f'{dirname}/{seed}_atoms_{suffix}.traj')

        # Now switch working directory to ensure all outputs are together
        orig_dir = os.getcwd()
        os.chdir(dirname)

        # Calculate E0s from atom_traj
        from esteem.trajectories import atom_energies
        from ase.data import atomic_numbers
        atom_en = atom_energies(atom_traj)
        atom_traj.close()
        E0s = {}
        for sym in atom_en:
            E0s[atomic_numbers[sym]] = atom_en[sym]
        if heads is not None:
            for target in targets:
                targetstr = targets[target]
                if target != 'diff':
                    heads[targetstr]['E0s'] = str(E0s)
                else:
                    heads[targetstr]['E0s'] = str({Z: 0 for Z in E0s})
        else:
            train_args.E0s = str(E0s)
        # Use the elements with E0s as the element table, so that elements in the atoms
        # trajectory but absent from the training data (eg O from a water reference)
        # do not give a mismatch between atomic energies and node attributes
        if getattr(train_args, 'atomic_numbers', None) is None:
            train_args.atomic_numbers = str(sorted(E0s))

        # Set up input data
        train_args.name = self.calc_filename(seed, targets_in, prefix="", suffix=suffix)
        train_args.device = 'cuda'
        if not self.xmace:  # X-MACE has no heads argument
            train_args.heads = str(heads) if heads is not None else heads
            print('# Setting heads to:', train_args.heads)

        # Some fixes to the input parameter list that prevent breakage
        if train_args.start_swa is None:
            train_args.start_swa = train_args.max_num_epochs // 4 * 3

        # Write config.txt
        config_file = self.calc_filename(seed, targets_in, prefix="", suffix=suffix) + '_config.txt'
        print(f'# Writing MACE configuration to {config_file}')
        with open(config_file, "w") as f:
            for kw in train_args.__dict__:
                eq = '=' if getattr(train_args, kw) != "" else ""
                f.write(f'--{kw}{eq}{getattr(train_args, kw)}\n')

        extxyzfile = train_args.train_file
        print(f'# Training MACE model using trajectory {extxyzfile} with parameters:')
        print('#', train_args)

        if self.xmace:
            from xmace.cli import run_train
        else:
            from mace.cli import run_train
        run_train.run(train_args)

        os.chdir(orig_dir)

    def xmace_states(self, target):
        """States learned by an X-MACE model for this target, in model order (no 'diff' heads)"""
        if not isinstance(target, dict):
            return [target]
        return [targ for targ in target if targ != 'diff']

    def select_head(self, calc_ml, model, target, head):
        """Selects the head (state) of a multi-target model that the calculator(s) evaluate"""
        calcs = calc_ml if isinstance(calc_ml, list) else [calc_ml]
        if self.xmace:
            names = [target[targ] for targ in self.xmace_states(target)]
            if head not in names:
                raise Exception(f"Head {head} not among the X-MACE model states {names}")
            for calc in calcs:
                calc.state = names.index(head)
        else:
            self.check_head(calc_ml, head)
            model.info["head"] = head

    def check_head(self, calc_ml, head):
        if isinstance(calc_ml, list):
            head_found = [head in calc.models[0].heads for calc in calc_ml]
            heads_list = [calc.models[0].heads for calc in calc_ml]
        else:
            head_found = head in calc_ml.models[0].heads
            heads_list = calc_ml.models[0].heads
        if not head_found:
            raise Exception(f"Head {head} not found in MACE calculator heads list: {heads_list}")

    def traj_write(self, atoms, traj):
        kw = {  # 'dipole': atoms.get_dipole_moment(),
            # 'charges': atoms.get_charges(),
              'energy': atoms.get_potential_energy(),
              'forces': atoms.get_forces()}
        traj.write(atoms, **kw)

    def restore_from_coordinates(self, model, crdfile):
        model = read(crdfile, index=-1)
        return model

    def heatup(self, model, seed, calc_params={}, nsteps=100):
        """
        Runs a heating up Molecular Dynamics calculation with the MACE ASE calculator.

        The temperature is ramped up to ``temp0`` in 10 stages of Langevin dynamics,
        using the ``temp0``, ``dt`` and ``friction`` attributes (set by the Solvate task).
        The final structure of each stage is written to heat.xyz.

        model: ASE Atoms

        seed: str

        calc_params: dict
            Calculator parameters, as for ``run_md``

        nsteps: int
            Total number of MD steps over all the stages
        """

        if not hasattr(self, 'dynamics') or self.dynamics is None:
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "LANG"
            self.dynamics.friction = self.friction
            self.dynamics.new_traj = True
        nramp = 10
        models_ramped = []
        for iramp in range(nramp):
            ramp_temp = self.temp0 * (iramp + 1) / nramp
            self.dynamics = self.run_md(model, seed + "_heat", calc_params, nsteps / nramp, self.dt, 1, ramp_temp, dynamics=self.dynamics)
            models_ramped.append(model.copy())
            write('heat.xyz', models_ramped)

    def densityequil(self, model, seed, calc_params={}, nsteps=100):
        """
        Runs a density equilibration Molecular Dynamics calculation with the MACE ASE calculator.

        Uses NPT dynamics at ``temp0``, with the ``dt`` and ``ttime`` attributes, and
        writes the final structure to density.xyz.

        model: ASE Atoms

        seed: str

        calc_params: dict
            Calculator parameters, as for ``run_md``

        nsteps: int
        """

        if ((not hasattr(self, 'dynamics') or self.dynamics is None) or
                (self.dynamics.type != "NPT")):
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "NPT"
            self.dynamics.friction = self.friction
            self.dynamics.ttime = self.ttime
            from ase import units
            self.dynamics.pfactor = 2 * 1.06e9 * (units.J / units.m**3) * self.ttime**2
            self.dynamics.new_traj = True
        self.dynamics = self.run_md(model, seed + "_dens", calc_params, nsteps, self.dt, 1, self.temp0, dynamics=self.dynamics)
        write('density.xyz', model)

    def equil(self, model, seed, calc_params={}, nsteps=100):
        """
        Runs an equilibration Molecular Dynamics calculation with the MACE ASE calculator.

        Uses Langevin dynamics at fixed volume at ``temp0``, and writes the final
        structure to equil.xyz.

        model: ASE Atoms

        seed: str

        calc_params: dict
            Calculator parameters, as for ``run_md``

        nsteps: int
        """

        if ((not hasattr(self, 'dynamics') or self.dynamics is None) or
                (self.dynamics.type != "LANG")):
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "LANG"
            self.dynamics.friction = self.friction
            self.dynamics.new_traj = True
        self.dynamics = self.run_md(model, seed + "_equil", calc_params, nsteps, self.dt, 1, self.temp0, dynamics=self.dynamics)
        write('equil.xyz', model)

    def snapshots(self, model, seed, calc_params={}, nsnaps=1, nsteps=100, start=0, nat_solu=0, nat_solv=0):
        """
        Generates snapshots from Langevin Molecular Dynamics with the MACE ASE calculator.

        Each snapshot follows ``nsteps`` steps of MD; it is reimaged (if there is a cell)
        and written to ``{seed}.traj`` and to snapNNNN.xyz.

        model: ASE Atoms

        seed: str

        calc_params: dict
            Calculator parameters, as for ``run_md``

        nsnaps: int
            Number of snapshots
        nsteps: int
            Number of MD steps between snapshots
        start: int
            Snapshot to resume from (0 starts a new trajectory)
        nat_solu: int
            Number of atoms in the solute, for reimaging
        nat_solv: int
            Number of atoms in each solvent molecule, for reimaging
        """
        from ase.io import Trajectory

        trajname = seed + '.traj'
        if start == 0:
            traj = Trajectory(trajname, 'w')
        else:
            traj = Trajectory(trajname)
            prevframe = len(traj)
            print(f'# Found trajectory file {trajname} containing {prevframe} frames.')
            if prevframe != start + 1:
                raise Exception(f'Error: This does not agree with the resumption point {start}.')
            traj.close()
            traj = Trajectory(trajname, 'a')
        if True:
            # ((not hasattr(self,'dynamics') or self.dynamics is None) or
            # (self.dynamics.type!="LANG")):
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "LANG"
            self.dynamics.friction = self.friction
            self.dynamics.new_traj = True
        snapout = model.copy()
        for step in range(start, nsnaps):
            self.dynamics = self.run_md(snapout, seed + "_snaps", calc_params, nsteps, self.dt, 1, self.temp0, dynamics=self.dynamics)
            from esteem.tasks.clusters import reimage_cluster
            snap_reimage = snapout.copy()
            if snapout.cell.volume > 0.0:
                reimage_cluster(snap_reimage, nat_solv, nat_solu)
            traj.write(snap_reimage)
            write(f'snap{step:04}.xyz', snap_reimage)

    def run_md(self, model, mdseed, calc_params, md_steps, md_timestep, superstep, temp,
               solvent=None, charge=0, restart=False, readonly=False, constraints=None, dynamics=None,
               continuation=None):
        """
        Runs a Molecular Dynamics calculation with the MACE ASE calculator.

        model: ASE Atoms

        mdseed: str
            Name for the output trajectory ``{mdseed}.traj``

        calc_params: dict
            Calculator parameters: ``calc_seed``, ``target``, ``calc_suffix``,
            ``calc_dir_suffix``, ``calc_prefix``, and ``calc_head`` if ``target`` is a dict

        md_steps: int

        md_timestep: float
            Timestep in ASE time units

        superstep: int
            Ignored

        temp: float
            Temperature in K

        solvent: str or None
            Ignored

        charge: int
            Total charge, stored in ``model.info['charge']`` (the ``charge`` attribute is used if 0)

        restart: bool
            Ignored

        readonly: bool
            If True, recalculate the energy of the final image in ``{mdseed}.xyz``
            instead of running MD

        dynamics:
            Either an object with attributes ``type`` ('LANG' or 'NPT'), ``friction`` and
            ``new_traj`` (plus ``ttime`` and ``pfactor`` for NPT), or the ASE dynamics
            object returned by a previous call, which is then continued. Must not be None.

        *Returns*: the ASE dynamics object, or None if ``readonly``
        """

        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md import Langevin, npt
        from ase.io import Trajectory
        from ase import units

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']

        # Load the MACE Calculator
        calc_ml = self.load(calc_seed, target, prefix=prefix, suffix=suffix, dir_suffix=dir_suffix)
        model.info['charge'] = charge if charge else self.charge
        if isinstance(target, dict):
            self.select_head(calc_ml, model, target, calc_params['calc_head'])
        model.calc = calc_ml

        # Initialise velocities if this is first step, otherwise inherit from model
        if np.all(model.get_momenta() == 0.0):
            MaxwellBoltzmannDistribution(model, temperature_K=temp)

        # For each ML superstep, remove C.O.M. translation and rotation
        # Stationary(model)
        # ZeroRotation(model)
        # print(f'constraints: {model.constraints}')

        if readonly:
            model = read(mdseed + ".xyz")  # Read final image
            model.calc = calc_ml
            model.get_potential_energy()  # Recalculate energy for final image
            return None
        else:
            new_traj = dynamics.new_traj
            friction = dynamics.friction
            dyn_type = dynamics.type
            if dyn_type == "LANG":
                if type(dynamics) != Langevin:
                    dynamics = Langevin(model, timestep=md_timestep, temperature_K=temp, friction=friction)
                else:  # in case they have changed
                    dynamics.set_timestep(md_timestep)
                    dynamics.set_friction(friction)
                    dynamics.set_temperature(temperature_K=temp)
            if dyn_type == "NPT" and type(dynamics) != npt.NPT:
                ttime = dynamics.ttime
                pfactor = dynamics.pfactor  # None #1.06e9*(units.J/units.m**3)*ttime**2 # Bulk modulus for ethanol
                dynamics = npt.NPT(model, timestep=md_timestep, temperature_K=temp, externalstress=0,
                                   pfactor=pfactor, ttime=ttime, mask=np.eye(3, dtype=bool))
            dynamics.new_traj = new_traj
            dynamics.friction = friction
            dynamics.type = dyn_type
            if new_traj:
                if hasattr(dynamics, 'traj'):
                    dynamics.traj.close()
                    dynamics.observers = []
                dynamics.traj = Trajectory(mdseed + ".traj", 'w', model)
                dynamics.attach(self.traj_write, interval=1, atoms=model, traj=dynamics.traj)
                dynamics.new_traj = False

            dynamics.run(md_steps)
            return dynamics

    # Define a MACE geometry optimisation function
    def geom_opt(self, model, seed, calc_params, driver_tol='default',
                 solvent=None, charge=0, spin=0, writeonly=False, readonly=False, continuation=False, cleanup=False,
                 traj=None):
        """
        Runs a BFGS geometry optimisation calculation with the MACE ASE calculator

        model: ASE Atoms

        seed: str
            Ignored

        calc_params: dict
            Calculator parameters, as for ``run_md``

        driver_tol: str
            Force tolerance: 'default' (0.00045 eV/Ang), 'loose' (0.0045), 'veryloose' (0.1)
            or 'tight' (0.000015)

        charge: int
            Total charge, as for ``run_md``

        traj: str or None
            Trajectory file for the optimisation steps

        The other arguments are ignored.

        *Returns*: energy (eV), forces (eV/Ang) and positions (Ang) of the optimised structure
        """
        from ase.io import Trajectory

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']

        from ase.optimize import BFGS
        from ase.units import Hartree, Bohr

        # Load the appropriate MACE Calculator
        calc_ml = self.load(calc_seed, target, prefix=prefix, suffix=suffix, dir_suffix=dir_suffix)
        model.info['charge'] = charge if charge else self.charge
        if isinstance(target, dict):
            self.select_head(calc_ml, model, target, calc_params['calc_head'])
        model.calc = calc_ml

        # Create instance of BFGS optimizer, run it and return results
        dyn = BFGS(model, trajectory=traj)

        # tolerances corresponding to NWChem settings
        fac = 1
        if driver_tol == 'default':
            fmax = 0.00045 * fac
        if driver_tol == 'loose':
            fmax = 0.00450 * fac
        if driver_tol == 'veryloose':
            fmax = 0.1 * fac
        if driver_tol == 'tight':
            fmax = 0.000015 * fac

        dyn.run(fmax=fmax)

        return model.get_potential_energy(), model.get_forces(), model.get_positions()

    def freq(self, model_opt, seed, calc_params, solvent=None, charge=0,
             temp=300, writeonly=False, readonly=False, continuation=False, cleanup=True):
        """
        Runs a Vibrational Frequency calculation with the MACE ASE calculator

        Uses ASE's ``Infrared`` class, and writes the IR spectrum to ``{name}_ir_spectrum.dat``.

        model_opt: ASE Atoms

        seed: str
            Seed for the name of the vibrational calculation

        calc_params: dict
            Calculator parameters, as for ``run_md``

        charge: int
            Total charge, as for ``run_md``

        The other arguments are ignored.

        *Returns*: the ASE ``Infrared`` object
        """

        from ase.vibrations import Vibrations, Infrared

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']

        # Load the appropriate MACE  Calculator
        calc_ml = self.load(calc_seed, target, prefix=prefix, suffix=suffix, dir_suffix=dir_suffix)
        model_opt.info['charge'] = charge if charge else self.charge
        if isinstance(target, dict):
            self.select_head(calc_ml, model_opt, target, calc_params['calc_head'])
        model_opt.calc = calc_ml

        # Create instance of Vibrations class, run it and return results
        # vib = Vibrations(model_opt,name=self.calc_filename(seed,target,prefix=prefix,suffix=suffix))
        # vib.run()
        # freqs = vib.get_frequencies()
        # vib.summary()
        # vib.clean()
        ir = Infrared(model_opt, name=self.calc_filename(seed, target, prefix=prefix, suffix=suffix))
        ir.run()
        ir.summary()
        ir.write_spectra(out=ir.name + '_ir_spectrum.dat', start=0, end=4000, width=20)
        ir.clean()

        # print(freqs)
        return ir

    def singlepoint(self, model, seed, calc_params, solvent=None, charge=0, spin=0,
                    forces=False, dipole=False, calc=False, readonly=False, continuation=False,
                    cleanup=True):
        """
        Runs a singlepoint calculation with the MACE ASE calculator, or a committee

        model: ASE Atoms

        seed: str
            Ignored

        calc_params: dict
            Calculator parameters, as for ``run_md``. A dict ``calc_suffix`` gives a committee.

        charge: int
            Total charge, as for ``run_md``

        forces: bool
            Also return the forces

        dipole: bool
            Also return the dipole (None if the model does not provide one)

        calc: bool
            Also return the calculator(s)

        The other arguments are ignored.

        *Returns*: a list of the energy (eV), then forces (eV/Ang), dipole (e Ang) and
        calculator if requested. For a committee, each result is an array over members,
        and ``model.calc`` becomes a SinglePointCalculator holding them (plus
        ``energy_std`` etc).
        """

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']

        # Load the appropriate MACE Calculator(s)
        calc_ml = self.load(calc_seed, target, prefix=prefix, suffix=suffix, dir_suffix=dir_suffix)
        model.info['charge'] = charge if charge else self.charge
        if isinstance(target, dict):
            self.select_head(calc_ml, model, target, calc_params['calc_head'])
        if isinstance(calc_ml, list):
            e_calc = []
            f_calc = []
            d_calc = []
            # some models (eg X-MACE) do not provide a dipole
            has_dipole = all('dipole' in tcalc.implemented_properties for tcalc in calc_ml)
            for tcalc in calc_ml:
                tcalc.results = {}
                model.calc = tcalc
                e_calc.append(model.get_potential_energy())
                if forces:
                    f_calc.append(model.get_forces())
                if dipole and has_dipole:
                    d_calc.append(model.get_dipole_moment())
            e_calc = np.array(e_calc)
            model.calc.results["energy"] = e_calc
            model.calc.results["energy_std"] = np.std(e_calc)
            if dipole and not has_dipole:
                d_calc = None
            elif dipole:
                d_calc = np.array(d_calc)
                model.calc.results["dipole"] = d_calc
                model.calc.results["dipole_std"] = np.std(d_calc, axis=1)
            if forces:
                f_calc = np.array(f_calc)
                model.calc.results["forces"] = f_calc
                model.calc.results["forces_std"] = np.std(f_calc, axis=1)
            # Keep the committee results (arrays over models) on the structure, so that
            # snapshots written to trajectories carry them (needed for uncertainty-based
            # selection); previously model.calc was replaced by the list of calculators
            from ase.calculators.singlepoint import SinglePointCalculator
            from ase.calculators.calculator import all_properties
            committee_results = dict(model.calc.results)
            spc = SinglePointCalculator(model, **{k: v for k, v in committee_results.items()
                                                  if k in all_properties})
            spc.results.update(committee_results)
            model.calc = spc
        else:
            model.calc = calc_ml
            e_calc = model.get_potential_energy()
            if forces:
                f_calc = model.get_forces()
            if dipole:
                # some models (eg MACE-POLAR foundation models) do not provide a dipole
                if 'dipole' in calc_ml.implemented_properties:
                    d_calc = model.get_dipole_moment()
                else:
                    d_calc = None
        res = [e_calc]
        if forces:
            res.append(f_calc)
        if dipole:
            res.append(d_calc)
        if calc:
            res.append(calc_ml)
        return res
