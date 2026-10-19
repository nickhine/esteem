#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines the ORCAWrapper Class"""

# Set up imports
import ase
from ase.calculators.orca import ORCA
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import Hartree, invcm, fs
from os import environ, path, makedirs, chdir, getcwd, remove, symlink
from shutil import copyfile
import numpy as np


class ORCAWrapper():
    """Sets up and runs the ORCA Calculator (via ASE) for DFT and TDDFT calculations"""

    def __init__(self):
        """Sets up instance attributes for ORCAWrapper """

        # Set up some defaults
        self.nprocs = 24
        self.maxcore = 2000
        self.temproot = "/tmp"
        self.output = "orca"
        self.origdir = None
        self.exts_to_link = ['out', 'err', 'gbw', 'cis', 'inp', 'engrad', '_property.txt', 'mdrestart']
        self.exts_to_save = ['log', 'txt', 'xyz', 'hess', 'engrad']
        self.scf_block = None

    def setup(self, nprocs=None, orca_cmd=None, mpi_options=None, maxcore=None):
        """
        Sets up the internal variables of the ORCAWrapper class, including run command

        ``orca_cmd`` is the path to the ORCA executable, used to create the ASE
        ``OrcaProfile`` that runs ORCA. If it is not given, any existing profile is
        kept, or else a default path is used. ``nprocs`` and ``maxcore`` set the
        ``%pal nprocs`` and ``%maxcore`` (MB per process) values written to the input.
        ``ASE_ORCA_COMMAND`` is also set from ``orca_cmd`` and ``mpi_options``, if not
        already set, for older versions of ASE.
        """
        profile_cmd = orca_cmd
        try:
            orca_cmd = environ["ASE_ORCA_COMMAND"]
        except KeyError:
            if orca_cmd is None:
                orca_cmd = "~/orca5/orca"
            if mpi_options is None:
                mpi_options = "--oversubscribe --mca opal_warn_on_missing_libcuda 0"
            environ["ASE_ORCA_COMMAND"] = f'{orca_cmd} PREFIX.inp "{mpi_options}" >> PREFIX.out 2> PREFIX.err'
        from ase.calculators.orca import OrcaProfile
        # Only replace the profile if a command is given, so repeated setup() calls
        # (eg to change nprocs) do not discard a user-supplied orcaprofile
        if profile_cmd is not None:
            self.orcaprofile = OrcaProfile(command=profile_cmd)
        elif not hasattr(self, 'orcaprofile'):
            self.orcaprofile = OrcaProfile(command="/storage/nanosim/orca5/orca")
        if maxcore is not None:
            self.maxcore = maxcore
        if nprocs is not None:
            self.nprocs = nprocs

    def get_completed_file_exts(self):
        return ['.out', '.engrad']

    def get_restart_file_exts(self):
        return ['.mdrestart', '.gbw']

    def move_to_tempdir(self, seed):
        self.origdir = getcwd()
        self.tempdir = f'{self.temproot}/{seed}'
        if not path.exists(self.tempdir):
            # print(f'Creating temporary directory {self.tempdir}')
            makedirs(self.tempdir)
        # print(f'Changing to temporary directory {self.tempdir}')
        try:
            chdir(self.tempdir)
        except:
            print(f'Failed to change to {self.tempdir}')
        if self.origdir != self.tempdir:
            # print(f'Making symlinks for {self.exts_to_link} files back to {self.origdir}')
            for ext in self.exts_to_link:
                ext = '.' + ext if not '.' in ext else ext
                if path.islink(f'./{seed}{ext}') or path.isfile(f'./{seed}{ext}'):
                    remove(f'./{seed}{ext}')
                symlink(f'{self.origdir}/{seed}{ext}', f'./{seed}{ext}')

    def return_from_tempdir(self, seed):
        import glob
        if self.origdir is None:
            raise Exception(f'Original directory has not been set!')
        # print(f'Copying {self.exts_to_save} files back to {self.origdir}')
        for ext in self.exts_to_save:
            ext = '.' + ext if not '.' in ext else ext
            extfiles = glob.glob(f"{seed}*{ext}")
            for f in extfiles:
                if path.isfile(f) and not path.islink(f):
                    copyfile(f, f'{self.origdir}/{f}')
        # Delete temporary copies of files linked back
        if self.origdir != self.tempdir:
            for ext in self.exts_to_link:
                ext = '.' + ext if not '.' in ext else ext
                if path.islink(f'{seed}{ext}'):
                    remove(f'{seed}{ext}')
        # print(f'Returning to {self.origdir}')
        chdir(self.origdir)

    def cleanup(self, seed):
        """Cleans up temporary files created by a ORCA run that are of no further use"""

        # Cleanup temporary files - TODO
        for dir in "./", seed + "/":
            for ext in ["densities", "err"]:
                ext = '.' + ext if not '.' in ext else ext
                if path.isfile(f'{dir}{seed}{ext}'):
                    remove(f'{dir}{seed}{ext}')

    def unpack_params(self, calc_params):
        if 'basis' in calc_params:
            basis = calc_params['basis']
        else:
            raise Exception("Basis not specified in calc_params")
        if 'func' in calc_params:
            xc = calc_params['func']
        else:
            raise Exception("XC Func not specified in calc_params")
        if 'target' in calc_params:
            target = calc_params['target']
        else:
            raise Exception("Target not specified in calc_params")
        if 'disp' in calc_params:
            disp = calc_params['disp']
        else:
            disp = None
        return basis, xc, target, disp

    def _cosmo_seed(self, solvent):
        # See here for a list of solvents:
        # https://www.orcasoftware.de/tutorials_orca/prop/CPCM.html#cpcm-and-cosmo
        if solvent == 'meth':
            return 'methanol'
        elif solvent == 'etoh':
            return 'ethanol'
        elif solvent == 'etgl':  # ethylene glycol not recognised by orca - set in cpcm_block below
            return 'ethanol'
        elif solvent == 'glyc':  # glycerol not recognised by orca - set in cpcm_block below
            return 'ethanol'
        elif solvent == 'acet':
            return 'acetonitrile'
        elif solvent == 'actn':
            return 'acetone'
        elif solvent == 'dich':
            return 'CH2Cl2'
        elif solvent == 'diox':
            return 'toluene'  # closest match to properties - fixed later
        elif solvent == 'dich':
            return 'CH2Cl2'
        elif solvent == 'cycl':
            return 'cyclohexane'
        elif solvent == 'watr' or solvent == 'wtr':
            return 'water'
        else:
            return solvent

# data from
# https://www.engineeringtoolbox.com/liquid-dielectric-constants-d_1263.html#google_vignette
# https://www.engineeringtoolbox.com/refractive-index-d_1264.html
# https://depts.washington.edu/eooptic/linkfiles/dielectric_chart%5B1%5D.pdf
    def _add_cpcm_block(self, calc, solvent):
        if solvent == 'etgl':
            self.cpcm_block = (f"\n%cpcm\n" +
                               f"epsilon 37.0\n" +
                               f"refrac 1.43\nend\n")
        elif solvent == 'glyc':
            self.cpcm_block = (f"\n%cpcm\n" +
                               f"epsilon 46.5\n" +
                               f"refrac 1.47\nend\n")
        elif solvent == 'diox':
            self.cpcm_block = (f"\n%cpcm\n" +
                               f"epsilon 2.25\n" +
                               f"refrac 1.4224\nend\n")
        else:
            self.cpcm_block = None
            # for reference, ethanol: 25.3, 1.36
        if self.cpcm_block is not None:
            calc.parameters['orcablocks'] += self.cpcm_block

    def check_func(self, func):

        fullfunc = func
        return fullfunc

    def set_careful_scf(self, sthresh=4e-8):
        self.scf_block = f"\n%scf\n"
        self.scf_block += f"maxiter 50\n"
        self.scf_block += f"sthresh {sthresh}\n"
        # self.scf_block += f"ConvForced False\n"
        # self.scf_block += f"SOSCFstart 0.00005 SOSCFMaxIt 12\n"
        # self.scf_block += f"AutoTRAH False\n"
        self.scf_block += f"end\n"

    def load():
        pass

    def _add_solvent(self, calc, solvent):

        calc.parameters['orcasimpleinput'] += f" CPCMC({self._cosmo_seed(solvent)})"
        self._add_cpcm_block(calc, solvent)

    def _add_tddft(self, calc, nroots, target=None, all_forces=False):

        calc.parameters['orcablocks'] += f"\n%tddft\n  nroots {nroots}\n  tda false"
        if target is not None:
            if isinstance(target, list):
                targliststr = ",".join([str(t) for t in target])
                calc.parameters['orcablocks'] += f"\n  irootlist {targliststr}"
                if all_forces:
                    calc.parameters['orcablocks'] += f"\n  sgradlist {targliststr}"
            else:
                calc.parameters['orcablocks'] += f"\n  iroot {target}"
        calc.parameters['orcablocks'] += f"\n  end"

    def singlepoint(self, model, label, calc_params={}, solvent=None, charge=0, spin=0,
                    forces=False, dipole=True, continuation=False, readonly=False, calc=False,
                    excitations=False, cleanup=True):
        """
        Runs a singlepoint calculation with the ORCA ASE calculator

        ``calc_params`` must contain ``basis``, ``func`` and ``target`` (and optionally
        ``disp``). A non-zero integer target runs TDDFT for that root. A list of targets
        runs TDDFT with gradients for every root in the list, and the results are
        arrays over the targets. With ``readonly``, results are read from
        ``{label}.out`` without running ORCA. ``continuation`` is not used.

        Returns a list containing the energy (eV), then, if requested, the forces
        (eV/Angstrom, if ``forces``), the dipole moment (e*Angstrom, if ``dipole``), the
        excitation spectrum (if ``excitations`` and target is a list) and the
        calculator (if ``calc``).
        """
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        methstr = '' if not forces else 'engrad '
        extra = ""  # "defgrid2"
        calc_orca = ORCA(label=label, orcasimpleinput=f"{methstr}{xc} {dispstr} {basis} {extra}",
                         profile=self.orcaprofile,
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n',
                         mult=int(2 * spin + 1), charge=charge)
        calc_orca.template.inputname = label + ".inp"
        calc_orca.template.outputname = label + ".out"

        if (target is not None) and (target != 0):
            if isinstance(target, list):
                nroots = max(target)
                all_forces = True
            else:
                nroots = target
                all_forces = False
            self._add_tddft(calc_orca, nroots, target, all_forces)
        if solvent is not None:
            self._add_solvent(calc_orca, solvent)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        # run the calculation
        if not readonly:
            model.calc = calc_orca
            self.move_to_tempdir(label)
            e_calc = model.get_potential_energy()
        old_format = False
        if old_format:
            if readonly:
                from ase.calculators.singlepoint import SinglePointCalculator
                results = calc_orca.template.read_results(calc_orca.directory)  # skip calculation
                calc_orca = SinglePointCalculator(atoms=model, **results)
                model.calc = calc_orca
            if forces:
                f_calc = model.get_forces()
            if dipole:
                d_calc = model.get_dipole_moment()
        else:
            from ase.io import read
            if isinstance(target, list):
                models_read = read(calc_orca.template.outputname, index=":",
                                   separate_excitations=True,)
                model_read = models_read[0]
            else:
                model_read = read(calc_orca.template.outputname)
            model.calc = model_read.calc
            model.positions = model_read.positions
            if isinstance(target, list):
                for i in target:
                    model.calc.atoms = model_read
                e_calc = np.array([models_read[i].calc.get_potential_energy() for i in target])
                if forces:
                    f_calc = np.array([models_read[i].calc.get_forces() for i in target])
                if dipole:
                    d_calc = np.array([models_read[i].calc.get_dipole_moment() for i in target])
                if "excitation_spectrum" in models_read[i].calc.results:
                    exc_calc = np.array([models_read[i].calc.results['excitation_spectrum'] for i in target])
                else:
                    exc_calc = None
            else:
                model.calc.atoms = model_read
                if 'initial_charges' in model.arrays:
                    del model.arrays['initial_charges']
                e_calc = model_read.get_potential_energy()
                if forces:
                    f_calc = model_read.get_forces()
                if dipole:
                    d_calc = model_read.get_dipole_moment()
        if cleanup and not readonly:
            self.cleanup(label)
        if not readonly:
            self.return_from_tempdir(label)

        res = [e_calc]
        if forces:
            res.append(f_calc)
        if dipole:
            res.append(d_calc)
        if excitations and isinstance(target, list):
            res.append(exc_calc)
        if calc:
            res.append(model.calc)
        return res

    def geom_opt(self, model_opt, label, calc_params={}, driver_tol='default',
                 solvent=None, continuation=False, charge=0, spin=0, readonly=False,
                 calconly=False, cleanup=True):
        """
        Runs a Geometry Optimisation calculation with the ORCA ASE calculator

        If ``driver_tol`` contains ``'tight'`` (eg ``'tight'``, ``'verytight'``), it is
        prepended to the ORCA ``opt`` keyword. ``FixAtoms`` constraints on
        ``model_opt`` are passed to ORCA as Cartesian constraints.

        Returns ``(energy, forces, positions)``, in eV, eV/Angstrom and Angstrom, and
        updates the positions of ``model_opt`` from the final frame of
        ``{label}_trj.xyz``. With ``calconly``, returns only the calculator, without
        running it.
        """
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        methstr = ''
        extra = ''
        taskstr = 'opt'
        if driver_tol.lower().find('tight') > -1:
            taskstr = driver_tol + 'opt'
        calc_orca = ORCA(label=label, orcasimpleinput=f"{taskstr} {methstr}{xc} {dispstr} {basis} {extra}", profile=self.orcaprofile,
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n',
                         mult=int(2 * spin + 1), charge=charge, task='opt')
        calc_orca.template.inputname = label + ".inp"
        calc_orca.template.outputname = label + ".out"
        if (target is not None) and (target != 0):
            self._add_tddft(calc_orca, target, target)
        if solvent is not None:
            self._add_solvent(calc_orca, solvent)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        if hasattr(self, "sym_thresh"):
            sym_block = f"\n%sym SymThresh {self.sym_thresh} end\n"
            calc_orca.parameters['orcablocks'] += sym_block

        if len(model_opt.constraints) > 0:
            from ase.constraints import FixAtoms
            constraint_str = "\n%geom Constraints\n"
            for constr in model_opt.constraints:
                if isinstance(constr, FixAtoms):
                    # fix the Cartesian position of each atom (ORCA atom indices start at 0)
                    for i in constr.index:
                        constraint_str = constraint_str + f'  {{ C {i} C }}\n'
            constraint_str = constraint_str + "  end\nend\n"
            calc_orca.parameters['orcablocks'] += constraint_str
        # Set up spin
        # calc_orca.set(mult=int(2*spin+1))
        # Set up charge
        # calc_orca.set(charge=charge)
        # Set task
        # calc_orca.set(task='opt')
        model_opt.calc = calc_orca
        if calconly:
            return calc_orca
        if readonly:
            calc_orca.atoms = model_opt
            calc_orca.read_results()  # skip calculation
        if not readonly:
            self.move_to_tempdir(label)
        forces = model_opt.get_forces()
        energy = calc_orca.results['energy']  # avoid second calculation
        # energy = model_opt.get_potential_energy()
        from ase.io import read
        model_opt.positions = read(f"{label}_trj.xyz").positions
        if cleanup and not readonly:
            self.cleanup(label)
        if not readonly:
            self.return_from_tempdir(label)
        return energy, forces, model_opt.positions

    def freq(self, model_opt, label, calc_params={}, solvent=None, charge=0, spin=0,
             temp=300, writeonly=False, readonly=False, continuation=False,
             cleanup=True, summary=False):
        """
        Runs a Vibrational Frequency calculation with the ORCA ASE calculator

        Note that the ORCA ``freq`` keyword is not currently added to the input, so
        this runs a singlepoint energy calculation. Returns None: the frequencies are
        not read back (:meth:`read_freq` is not called). ``temp``, ``writeonly``, ``continuation`` and ``summary`` are not used.
        """
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        taskstr = 'freq'
        calc_orca = ORCA(label=label, orcasimpleinput=f"{xc} {dispstr} {basis}",
                         profile=self.orcaprofile,
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n',
                         mult=int(2 * spin + 1), charge=charge)
        calc_orca.template.inputname = label + ".inp"
        calc_orca.template.outputname = label + ".out"
        if (target is not None) and (target != 0):
            self._add_tddft(calc_orca, target, target)
        if solvent is not None:
            self._add_solvent(calc_orca, solvent)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block

        model_opt.calc = calc_orca
        if readonly:
            calc_orca.atoms = model_opt
            # print("Reading results")
            try:
                calc_orca.read_results()  # skip calculation
            except Exception as e:
                print(f'Failed reading results: exception was {e}')
        else:
            self.move_to_tempdir(label)
        try:
            forces = model_opt.get_potential_energy()  # Run frequencies
        except Exception as e:
            print(f'Failed reading results: exception was {e}')
        # self.read_freq(calc_orca)
        if not readonly:
            self.return_from_tempdir(label)

        if cleanup:
            self.cleanup(label)

    def run_md(self, model, steplabel, calc_params, qmd_steps, qmd_timestep, superstep, temp,
               solvent=None, charge=0, spin=0, continuation=False, readonly=False,
               constraints=None, dynamics=None, cleanup=True):
        """
        Runs a Quantum Molecular Dynamics calculation with ORCA's MD module

        Runs ``qmd_steps`` steps of timestep ``qmd_timestep`` (in ASE time units) with a
        CSVR thermostat at ``temp`` (K), restarting from ``{steplabel}.mdrestart`` if it
        exists. ``superstep`` is used only in the printed summary, and ``continuation``
        and ``dynamics`` are not used.

        Returns None. The positions of ``model`` are updated to those of the last MD
        step, and a ``SinglePointCalculator`` holding that step's energy and forces is
        attached to it.
        """
        import random

        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        taskstr = 'md'
        calc_orca = ORCA(label=steplabel, orcasimpleinput=f"{xc} {dispstr} {basis}",
                         profile=self.orcaprofile,
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n',
                         mult=int(2 * spin + 1), charge=charge)
        calc_orca.template.inputname = steplabel + ".inp"
        calc_orca.template.outputname = steplabel + ".out"
        timecon = 10.0
        random.seed(steplabel)
        randseed = random.randint(0, 100000)
        md_block = f'''
%md
timestep {qmd_timestep / fs}_fs
initvel {temp}_K no_overwrite
restart ifexists
randomize {randseed}
thermostat csvr {temp}_K timecon {timecon}_fs
dump engrad
run {qmd_steps}
end
'''
        calc_orca.parameters['orcablocks'] += md_block
        if (target is not None) and (target != 0):
            self._add_tddft(calc_orca, target, target)
        if solvent is not None:
            self._add_solvent(calc_orca, solvent)
        if constraints is not None:
            calc_orca.set(constraints=constraints)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        # ORCA's MD module does not write single-point output that ase.io.orca can
        # parse, so run ORCA directly and take the positions, energy and forces of the
        # last MD step from its dumped .engrad file
        if readonly:
            print("Reading results")
        else:
            self.move_to_tempdir(steplabel)
            calc_orca.write_inputfiles(model, ['energy', 'forces'])
            calc_orca.template.execute(calc_orca.directory, calc_orca.profile)
            self.return_from_tempdir(steplabel)
        positions, energy, forces = self._read_md_engrad(steplabel)
        from ase.calculators.singlepoint import SinglePointCalculator
        model.positions = positions
        model.calc = SinglePointCalculator(model, energy=energy, forces=forces)

        if cleanup and not readonly:
            self.cleanup(steplabel)

        print(superstep, steplabel, energy, forces[0], model.positions[0])
        return None

    def _read_md_engrad(self, steplabel):
        """Reads positions (Angstrom), energy (eV) and forces (eV/Angstrom) of the last step
        of an ORCA MD run, from the ``{steplabel}-stepNNNNNN.engrad`` files written by
        ``dump engrad``"""
        import glob
        from ase.units import Bohr
        engrad_files = sorted(glob.glob(f'{steplabel}-step*.engrad'))
        if len(engrad_files) == 0:
            raise Exception(f'# No MD engrad files found for {steplabel}')
        with open(engrad_files[-1]) as f:
            values = [l.split() for l in f if not l.startswith('#') and l.strip()]
        natoms = int(values[0][0])
        energy = float(values[1][0]) * Hartree
        gradient = np.array([float(v[0]) for v in values[2:2 + 3 * natoms]]).reshape(natoms, 3)
        forces = -gradient * Hartree / Bohr
        positions = np.array([[float(x) for x in v[1:4]] for v in values[2 + 3 * natoms:2 + 4 * natoms]]) * Bohr
        return positions, energy, forces

    def excitations(self, model, label, calc_params={}, nroots=1, solvent=None, charge=0, spin=0,
                    writeonly=False, readonly=False, continuation=False, cleanup=True,
                    plot_homo=None, plot_lumo=None, plot_trans_den=None):
        """
        Calculates TDDFT excitations with the ORCA ASE calculator

        Solves for ``nroots`` roots; the target in ``calc_params`` is not used. The
        ``plot_*`` arguments are not used.

        Returns ``(s_excit, energy)``, where ``s_excit`` is the tuple returned by
        :meth:`read_excitations` and ``energy`` is the ground state energy in eV.
        With ``writeonly``, writes the input file and returns 0.
        """
        # Set up calculator
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        calc_orca = ORCA(label=label, orcasimpleinput=f"{xc} {dispstr} {basis}",
                         profile=self.orcaprofile,
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n',
                         mult=int(2 * spin + 1), charge=charge, task='energy')
        calc_orca.template.inputname = label + ".inp"
        calc_orca.template.outputname = label + ".out"
        self._add_tddft(calc_orca, nroots, target=None)
        if solvent is not None:
            self._add_solvent(calc_orca, solvent)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        model.calc = calc_orca
        if writeonly:
            calc_orca.write_input(atoms=model)
            return 0
        if readonly:
            calc_orca.read_results()
            print("Reading excitations")
            s_excit = self.read_excitations(label + ".out")
            energy = calc_orca.get_property('energy', atoms=model, allow_calculation=False)
        else:
            self.move_to_tempdir(label)
            energy = model.get_potential_energy()
            print("Reading excitations")
            s_excit = self.read_excitations(label + ".out")
            self.return_from_tempdir(label)
        if cleanup:
            self.cleanup(label)
        return s_excit, energy

    def read_excitations(self, filename):
        """
        Read Excitations from an ORCA output file (ORCA 5 or 6 format)

        Returns ``(s_excit, trans_lines)``: ``s_excit`` is a list of
        ``(root, energy in eV, oscillator strength)`` tuples, from the absorption
        spectrum via transition electric dipole moments, and ``trans_lines`` is an
        empty list (transition origins are not read). If the file contains several
        spectra, only the last is returned.
        """

        # if hasattr(calc,'label'):
        #    filename = calc.label+'.out'
        # else:
        #    filename = calc.template.outputname
        file = open(filename, 'r')
        lines = file.readlines()
        file.close()

        s_excit = []
        trans_lines = []
        getexcit = False
        for i, line in enumerate(lines):
            if line.find('ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS') >= 0:
                getexcit = True
                # In case of multiple copies of TDDFT output in file,
                # return only the last one
                s_excit = []
                trans_lines = []
                continue
            if getexcit and "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line:
                getexcit = False
                continue
            if getexcit and len(line.split()) > 1:
                # ORCA5 version
                if line.split()[0].isdigit():
                    # print(line)
                    root = int(line.split()[0])
                    energy = float(line.split()[1]) * invcm
                    if 'spin' in line.split()[3]:
                        osc = 0.0
                    else:
                        osc = float(line.split()[3])
                    s_excit.append((root, energy, osc))
                # ORCA6 version
                elif line.split()[1] == '->':
                    print(line)
                    line.split()[0].isdigit()
                    root = int(line.split()[2].split("-")[0])
                    energy = float(line.split()[4]) * invcm
                    if 'spin' in line.split()[3]:
                        osc = 0.0
                    else:
                        osc = float(line.split()[6])
                    s_excit.append((root, energy, osc))
        # calc.results['excitations'] = np.array(s_excit)
        # calc.results['transition_origins'] = trans_lines
        return s_excit, trans_lines

    def read_freq(self, calc):
        """
        Read Vibrational Frequencies and Normal Modes from the output file of an ORCA calculator

        Reads ``calc.label + '.out'``, stores the results in ``calc.results`` and
        returns the frequencies. The parsing of normal modes, derivative dipoles and
        IR intensities follows the NWChem output format, and is not called by
        :meth:`freq`.
        """
        file = open(calc.label + '.out', 'r')
        lines = file.readlines()
        file.close()
        nat = len(calc.atoms)
        nRa = 3 * nat
        freq = []
        getfreq = False
        for i, line in enumerate(lines):
            if line.find('VIBRATIONAL FREQUENCIES') >= 0:
                # In case of multiple copies of frequency output in file,
                # return only the last one
                getfreq = True
                freq = []
                continue
            if getfreq and 'NORMAL MODES' in line:
                getfreq = False
                continue
            if getfreq and 'cm**' in line:
                f = float(line.split()[1]) * invcm
                nmode = np.zeros((nRa, nRa))
                maxRa = 0
                minRa = 0
                Rb = -1
                for j, line2 in enumerate(lines[i:i + int(nRa * nRa / 6 + nRa / 6 * 10)]):
                    words = line2.split()
                    if len(words) > 0:
                        if words == [str(e) for e in range(maxRa + 1, maxRa + len(words) + 1)]:
                            minRa = maxRa + 1
                            maxRa = minRa + len(words) - 1
                            Rb = 0
                        if words[0] == str(Rb + 1) and len(words) == maxRa - minRa + 2:
                            nmode[Rb, minRa - 1:maxRa] = [float(w) for w in words[1:]]
                            Rb = Rb + 1
                    if line2.find('Frequency') >= 0:
                        new_freqs = [float(s) for s in words[1:]]
                        freq[minRa - 1:maxRa] = new_freqs
            if line.find('Derivative Dipole Moments') >= 0:
                ddip = np.zeros((nRa, 3))
                Rb = 0
                for j, line2 in enumerate(lines[i:i + nRa + 10]):
                    words = line2.split()
                    if len(words) > 0:
                        if words[0] == str(Rb + 1):
                            ddip[Rb, 1:3] = [float(w) for w in words[4:6]]
                            Rb = Rb + 1
            if line.find('Infra Red Intensities') >= 0:
                intense = np.zeros((nRa))
                Rb = 0
                for j, line2 in enumerate(lines[i:i + nRa + 10]):
                    words = line2.split()
                    if len(words) > 0:
                        if words[0] == str(Rb + 1):
                            intense[Rb] = float(words[3])
                            Rb = Rb + 1
        print(nmode)
        print(freq)
        print(ddip)
        print(intense)
        calc.results['frequencies'] = freq
        calc.results['normal modes'] = nmode
        calc.results['IR intensities'] = intense
        calc.results['derivative dipole moments'] = ddip
        return freq


# In[ ]:
