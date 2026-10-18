#!/usr/bin/env python
# coding: utf-8

"""Defines the OpenQPWrapper Class

A minimal wrapper for OpenQP (https://github.com/Open-Quantum-Platform/openqp).
Each calculation writes an OpenQP input file, runs the ``openqp`` command on it,
and reads the results from the JSON file OpenQP writes next to its log.

Supported:
    * ground-state RHF/RKS energies and forces (target None or 0)
    * MRSF-TDDFT energies and forces for any set of states (target int > 0, or list),
      using a triplet ROHF reference. OpenQP computes one state gradient per run,
      so forces for several states take one run per state.
    * implicit solvent (PCM, ddX backend) for energies only: OpenQP's PCM does not
      yet provide gradients, so requesting forces in solvent raises NotImplementedError
    * gas-phase geometry optimisation of one state (OpenQP's runtype=optimize)

Not yet supported: MD, dispersion corrections.
"""

import json
import subprocess
from os import environ, path, replace

import numpy as np
from ase.units import Hartree, Bohr


class OpenQPWrapper():
    """Sets up and runs OpenQP calculations for ESTEEM tasks"""

    def __init__(self):
        """Sets up instance attributes for OpenQPWrapper"""
        self.nprocs = 1
        self.output = "openqp"
        # Command used to run OpenQP; it is given the input file name as its argument
        self.openqp_cmd = environ.get("ESTEEM_OPENQP_COMMAND", "openqp")
        self.guess = "huckel"
        # extra lines to append to the input, keyed by section, eg {'scf': ['conv=1e-7']}
        self.extra_input = {}
        # Dielectric constants for implicit solvent, keyed by ESTEEM solvent short names
        self.solvent_epsilon = {'watr': 78.3553, 'wtr': 78.3553, 'meth': 32.613,
                                'etoh': 24.852, 'acet': 35.688, 'actn': 20.493,
                                'dich': 8.93, 'cycl': 2.0165, 'diox': 2.25}
        self.pcm_backend = "ddx"
        self.pcm_model = "ddpcm"

    def setup(self, nprocs=None, openqp_cmd=None):
        """Sets the number of OpenMP threads and the command used to run OpenQP"""
        if nprocs is not None:
            self.nprocs = nprocs
        if openqp_cmd is not None:
            self.openqp_cmd = openqp_cmd

    def get_completed_file_exts(self):
        return ['.json']

    def get_restart_file_exts(self):
        return []

    def unpack_params(self, calc_params):
        for key in ['basis', 'func', 'target']:
            if key not in calc_params:
                raise Exception(f"{key} not specified in calc_params")
        return calc_params['basis'], calc_params['func'], calc_params['target'], calc_params.get('disp', False)

    def _write_input(self, model, label, basis, xc, charge, spin, runtype,
                     mrsf_nstate=None, grad_state=None, opt_state=None, solvent=None):
        """Writes an OpenQP input file {label}.inp"""
        lines = ["[input]", "system="]
        for num, pos in zip(model.get_atomic_numbers(), model.get_positions()):
            lines.append(f"   {num:3d} {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}")
        lines += [f"charge={charge}", f"runtype={runtype}", f"basis={basis.lower()}",
                  f"functional={xc.lower()}",
                  f"method={'tdhf' if mrsf_nstate is not None else 'hf'}", "",
                  "[guess]", f"type={self.guess}", "",
                  "[scf]"]
        if mrsf_nstate is not None:
            # MRSF-TDDFT uses a triplet ROHF reference
            lines += ["type=rohf", "multiplicity=3", "",
                      "[tdhf]", "type=mrsf", f"nstate={mrsf_nstate}"]
        else:
            mult = int(2*spin+1)
            lines += [f"type={'rhf' if mult == 1 else 'rohf'}", f"multiplicity={mult}"]
        lines.append("")
        if grad_state is not None:
            lines += ["[properties]", f"grad={grad_state}", ""]
        if opt_state is not None:
            lines += ["[optimize]", f"istate={opt_state}", ""]
        if solvent is not None:
            if solvent not in self.solvent_epsilon:
                raise Exception(f'# No dielectric constant known for solvent {solvent}: add it to solvent_epsilon')
            lines += ["[pcm]", "enabled=true", f"backend={self.pcm_backend}",
                      "mode=reference_scf", f"model={self.pcm_model}",
                      f"epsilon={self.solvent_epsilon[solvent]}", ""]
        for section, extra in self.extra_input.items():
            lines += [f"[{section}]"] + list(extra) + [""]
        with open(f"{label}.inp", "w") as f:
            f.write("\n".join(lines) + "\n")

    def _run(self, label):
        """Runs OpenQP on {label}.inp and returns the parsed {label}.json results"""
        cmd = f"{self.openqp_cmd} {label}.inp"
        env = dict(environ, OMP_NUM_THREADS=str(self.nprocs))
        result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
        if result.returncode != 0 or not path.isfile(f"{label}.json"):
            raise Exception(f"# OpenQP failed for {label} (exit code {result.returncode}):\n"
                            f"{result.stderr[-2000:]}")
        return self._read_json(label)

    def _read_json(self, label):
        with open(f"{label}.json") as f:
            return json.load(f)

    def _state_energy(self, results, state, mrsf):
        """Total energy (eV) of a state: for MRSF, reference energy + td_energies[state]"""
        if not mrsf:
            return results['energy']*Hartree
        td = results.get('td_energies')
        if td is None or state >= len(td) or td[state] is None:
            raise Exception(f'# MRSF state {state} not available in OpenQP results')
        return (results['energy'] + td[state])*Hartree

    def singlepoint(self, model, label, calc_params={}, solvent=None, charge=0, spin=0,
                    forces=False, dipole=False, continuation=False, readonly=False, calc=False,
                    cleanup=True):
        """
        Runs a singlepoint calculation with OpenQP

        For a list of targets, returns arrays over targets (as the ORCA wrapper does).
        Dipoles are not yet read from OpenQP, so None is returned in their place.
        """
        basis, xc, target, disp = self.unpack_params(calc_params)
        if disp:
            raise NotImplementedError("# OpenQPWrapper: dispersion corrections are not yet supported")
        if forces and solvent is not None:
            raise NotImplementedError("# OpenQPWrapper: OpenQP's PCM does not yet provide gradients, "
                                      "so forces in implicit solvent are not available")
        targets = target if isinstance(target, list) else [0 if target is None else target]
        mrsf = isinstance(target, list) or any(t > 0 for t in targets)
        nstate = max(targets) + 1 if mrsf else None

        if not forces:
            # One energy run gives every state
            run_label = label
            results = self._read_json(run_label) if readonly else None
            if results is None:
                self._write_input(model, run_label, basis, xc, charge, spin, "energy",
                                  mrsf_nstate=nstate, solvent=solvent)
                results = self._run(run_label)
            energies = [self._state_energy(results, t, mrsf) for t in targets]
            f_all = None
        else:
            # One gradient run per state (MRSF state numbering starts at 1 for S0)
            energies, f_all = [], []
            for t in targets:
                run_label = f"{label}_grad{t}" if len(targets) > 1 else label
                results = self._read_json(run_label) if readonly else None
                if results is None:
                    self._write_input(model, run_label, basis, xc, charge, spin, "grad",
                                      mrsf_nstate=nstate, grad_state=(t+1 if mrsf else None),
                                      solvent=solvent)
                    results = self._run(run_label)
                energies.append(self._state_energy(results, t, mrsf))
                f_all.append(-np.array(results['grad']).reshape(-1, 3)*Hartree/Bohr)

        if isinstance(target, list):
            e_calc = np.array(energies)
            f_calc = np.array(f_all) if forces else None
            d_calc = np.array([None]*len(targets)) if dipole else None
        else:
            e_calc = energies[0]
            f_calc = f_all[0] if forces else None
            d_calc = None
        from ase.calculators.singlepoint import SinglePointCalculator
        model.calc = SinglePointCalculator(model, energy=energies[0],
                                           forces=(f_all[0] if forces else None))
        res = [e_calc]
        if forces:
            res.append(f_calc)
        if dipole:
            res.append(d_calc)
        if calc:
            res.append(model.calc)
        return res

    def geom_opt(self, model_opt, label, calc_params={}, driver_tol='default',
                 solvent=None, continuation=False, charge=0, spin=0, readonly=False,
                 calconly=False, cleanup=True):
        """
        Optimises the geometry of one state with OpenQP (runtype=optimize)

        The optimised positions are written into model_opt. driver_tol is not used:
        OpenQP's own convergence criteria apply. An optimisation that does not
        converge raises an Exception.
        """
        basis, xc, target, disp = self.unpack_params(calc_params)
        if disp:
            raise NotImplementedError("# OpenQPWrapper: dispersion corrections are not yet supported")
        if solvent is not None:
            raise NotImplementedError("# OpenQPWrapper: OpenQP's PCM does not yet provide gradients, "
                                      "so geometry optimisation in implicit solvent is not available")
        if isinstance(target, list):
            raise Exception("# OpenQPWrapper: geometry optimisation needs a single target")
        state = 0 if target is None else target
        mrsf = state > 0
        # OpenQP's optimiser writes opt.xyz (latest geometry, in Angstrom), opt_geom.xyz
        # (every step) and opt_status.txt into the run directory: keep them as {label}_opt*
        opt_files = {'opt.xyz': f"{label}_opt.xyz", 'opt_geom.xyz': f"{label}_opt_geom.xyz",
                     'opt_status.txt': f"{label}_opt_status.txt"}
        if readonly:
            results = self._read_json(label)
        else:
            # istate numbers states as [properties] grad does: MRSF S0 is 1
            self._write_input(model_opt, label, basis, xc, charge, spin, "optimize",
                              mrsf_nstate=(state+1 if mrsf else None),
                              opt_state=(state+1 if mrsf else 0))
            results = self._run(label)
            for src, dest in opt_files.items():
                if path.isfile(src):
                    replace(src, dest)
            with open(f"{label}.log") as f:
                if "Has Converged" not in f.read():
                    raise Exception(f"# OpenQP geometry optimisation did not converge for {label}")
        model_opt.positions = np.loadtxt(opt_files['opt.xyz'], skiprows=2, usecols=(1, 2, 3)).reshape(-1, 3)
        energy = self._state_energy(results, state, mrsf)
        forces = -np.array(results['grad']).reshape(-1, 3)*Hartree/Bohr if 'grad' in results else None
        from ase.calculators.singlepoint import SinglePointCalculator
        model_opt.calc = SinglePointCalculator(model_opt, energy=energy, forces=forces)
        return energy, forces, model_opt.positions

    def run_md(self, *args, **kwargs):
        raise NotImplementedError("# OpenQPWrapper: MD is not yet supported")
