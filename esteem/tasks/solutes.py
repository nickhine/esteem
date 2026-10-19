#!/usr/bin/env python
# coding: utf-8

"""Runs DFT and TDDFT calculations on Solute (or Solvent) molecules, in implicit solvent"""

# # Main driver routine


class SolutesTask:

    def __init__(self, **kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = "solutes"
        default_args = self.make_parser().parse_args("")
        # Set values of all parameters (defaults unless overridden)
        for arg in vars(default_args):
            if arg in kwargs:
                val = kwargs[arg]
            else:
                val = getattr(default_args, arg)
            setattr(self, arg, val)

    def run(self, namelist):
        """
        Main routine for the Solutes task. Iterates through the sub-tasks, some of which are optional:

        Geometry optimisation, rotamer checks, rotation to plane, excited state calculations,
        and vibrational frequency calculations, in gas phase and then (if ``solvent`` is set)
        in implicit solvent.

        namelist: list of str
            Short names of the solutes to be optimised. Initial geometries
            should be present in './xyz' directory.

        Settings are taken from the task attributes (see the argument listing for details), including:

            ``solvent``, ``target``, ``rotate``, ``rotamers``, ``vibrations``, ``nroots``, ``charges``,
            ``calc_params`` (or ``basis``, ``func``, ``disp``), ``fc_fallback_dir``,
            ``plot_homo``, ``plot_lumo``, ``plot_trans_den``

        ``self.wrapper`` must be set to a wrapper for running components of the job, with members including:

            ``singlepoint``, ``geom_opt``, ``excitations``, ``freq``

        *Outputs for each species* (named ``{solute}_es{target}`` for an excited-state ``target``):

            In the directory 'xyz': initial geometries (either input, or from get_xyz_files)

            In the directory 'geom': working directories of the geometry optimisations.

            In the directory 'opt': optimised gas-phase geometries.

            In the directories 'best_rota' and 'opt_rot': geometries after the rotamer search and
            rotation to the xy-plane, if requested.

            In the directories 'tddft' and 'freq': gas-phase TDDFT and vibrational frequency results,
            if requested and no solvent is set.

            In the directory 'is_opt_{solvent}': optimised implicit solvent geometries. These may be used as inputs for the Solvate task.

            In the directory 'is_tddft_{solvent}': implicit solvent tddft results. These may be used as inputs for spectral warping in the Spectra task.

            In the directory 'is_freq_{solvent}': implicit solvent vibrational frequencies, if requested.

        """

        prevdir = "xyz"
        if self.calc_params is None:
            self.calc_params = {}
        if self.calc_params == {}:
            self.calc_params = {'basis': self.basis,
                                'func': self.func,
                                'target': self.target,
                                'disp': self.disp}

        # Geometry Optimisation
        driver_tol = 'default'
        if self.vibrations and self.solvent is None:
            driver_tol = 'tight'
        nextdir = "opt"
        self.geom_opt_all(namelist, prevdir, nextdir, self.wrapper.geom_opt, self.calc_params,
                          driver_tol, charges=self.charges)
        prevdir = nextdir
        # Rotamer search, if requested
        if self.rotamers:
            nextdir = "best_rota"
            self.find_best_rotas(namelist, prevdir, nextdir, self.wrapper.singlepoint,
                                 self.wrapper.geom_opt, self.calc_params, charges=self.charges)
            prevdir = nextdir
        if self.rotate:
            nextdir = "opt_rot"
            self.rotate_all_to_xy_plane(namelist, prevdir, nextdir, target=self.target)
            prevdir = nextdir
        if self.nroots > 0 and self.solvent is None:
            nextdir = "tddft"
            self.calc_all_excited_states(namelist, prevdir, nextdir, self.wrapper.excitations,
                                         self.calc_params, self.nroots, charges=self.charges)
        # Vibrational frequency calculations, if requested and no higher
        # level of theory will follow later
        if self.vibrations and self.solvent is None:
            nextdir = "freq"
            self.calc_vib_freq(namelist, prevdir, nextdir, self.wrapper.freq, self.calc_params, charges=self.charges)
        # Solvated calculations
        if self.solvent is not None:
            nextdir = "is_opt_" + self.solvstr(self.solvent)
            if self.vibrations:
                driver_tol = 'tight'
            # Geometry Optimisation
            self.geom_opt_all(namelist, prevdir, nextdir, self.wrapper.geom_opt,
                              self.calc_params, driver_tol, solvent=self.solvent,
                              charges=self.charges)
            prevdir = nextdir
            # TDDFT calculations
            if self.nroots > 0:
                nextdir = "is_tddft_" + self.solvstr(self.solvent)
                self.calc_all_excited_states(namelist, prevdir, nextdir, self.wrapper.excitations,
                                             self.calc_params, nroots=self.nroots,
                                             solvent=self.solvent, charges=self.charges,
                                             plot_homo=self.plot_homo, plot_lumo=self.plot_lumo,
                                             plot_trans_den=self.plot_trans_den)
            # Vibrational frequency calculations, if requested
            if self.vibrations:
                nextdir = "is_freq_" + self.solvstr(self.solvent)
                self.calc_vib_freq(namelist, prevdir, nextdir, self.wrapper.freq,
                                   self.calc_params, solvent=self.solvent,
                                   charges=self.charges)

    from os import makedirs, path

    # Download xyz files if they do not already exist
    def get_xyz_files(self, namelist, out_path):
        """
        Downloads initial geometries from PubChem (via ASE), based on their full names.

        These geometries are usually not great, but are a reasonable starting point for optimisation.

        Visit https://pubchem.ncbi.nlm.nih.gov to check that your names are recognised before use.

        This needs internet access, so if the machine you are using cannot reach PubChem,
        this routine will fail, in which case put starting point geometries in the directory
        ``out_path``. Molecules that already have a file there are not downloaded. All the
        conformers PubChem returns for a molecule are written to its file.

        namelist: dict
            Keys are shortnames (eg "cate"), entries are full names (eg "catechol" or "1,2-dihydroxybenzene")
        out_path: str
            String for directory name where xyz files will be written (eg "xyz"). Created if not present

        *Returns*: list of the shortnames
        """
        from os import path, makedirs
        from ase.io import write
        import subprocess

        # Download sdf file from cactus.nci.nih.gov
        if not path.exists(out_path):
            makedirs(out_path)
        for seed in namelist:
            if not path.exists(out_path + "/" + seed + ".xyz"):
                if False:
                    # TODO: use urllib here
                    wget_str = ("wget -O \"" + out_path + "/" + seed + ".xyz\" " +
                                "\"https://cactus.nci.nih.gov/chemical/structure/" +
                                namelist[seed] + "/file?format=xyz\"")
                    print(wget_str)
                    errorcode = subprocess.call(wget_str, shell=True)
                    if errorcode:
                        raise RuntimeError('{} returned an error: {}'
                                           .format('wget', errorcode))
                else:
                    from ase.data.pubchem import pubchem_atoms_conformer_search
                    all_conformers = pubchem_atoms_conformer_search(namelist[seed])
                    write(f'{out_path}/{seed}.xyz', all_conformers)

            else:
                print(f'Skipping download: {out_path}/{seed}.xyz already exists')

        # strip out long names and convert to list
        shortnames = [x for x in namelist]
        return shortnames

    from os import path, makedirs, getcwd, chdir
    from ase.io import read, write

    def solvstr(self, solvent):
        if isinstance(solvent, str):
            return solvent
        if isinstance(solvent, dict):
            return solvent['solvent']

    def get_solute(self, seed, target, infile, outfile, task, sol_str, in_path, baseseed):
        from ase.io import read, write
        from os import path
        if not path.exists(infile):
            # Try basename without _esX
            if target is not None and target != 0:
                infile = f'{in_path}/{baseseed}.xyz'
            if not path.exists(infile):
                print(f'Skipping {task} {sol_str}for: {seed} - no input file')
                return (None, None, None)
        solute = read(infile, index=slice(0, None))
        nconf = len(solute)
        if nconf > 1:
            print(f'Input file contains {nconf} conformers')
        if path.exists(outfile):
            solute_done = read(outfile, index=slice(0, None))
            nconf_done = len(solute_done)
            if nconf_done == nconf:
                print(f'Skipping {task} {sol_str}for: {seed} - output file already present')
                return (None, None, None)
            else:
                print(f'Continuing {task} {sol_str}for: {seed} - output file contains {nconf_done} conformers')
                solute[0:nconf_done] = solute_done[0:nconf_done]
        else:
            nconf_done = 0

        return solute, nconf, nconf_done

    # Optimize geometries of solute selection
    def geom_opt_all(self, solute_names, in_path, out_path, geom_opt_func, calc_params,
                     driver_tol='default', solvent=None, charges={}):
        """
        Geometry optimise all of a list of solutes

        solute_names: list of str
            Short names of the solutes to be optimised
        in_path: str
            Directory where .xyz files are expected to be found. Any not present are skipped.
        out_path: str
            Directory where optimised structure .xyz files are written. Created if not present.
        geom_opt_func: function
            A function wrapping creation of an ASE calculator and using it to perform geometry optimisation.
        calc_params: dict
            Contents varies between different wrappers, but generally specifies basis, functional etc
        driver_tol: str
            Geometry optimisation tolerance level passed to the wrapper (eg 'default', 'tight', 'loose')
        solvent: str or dict
            Implicit solvent name (or dict of solvent settings), or None for gas-phase
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule

        The excited state to optimise is taken from ``calc_params['target']``. Calculations run in
        'geom/{seed}'. If an excited-state optimisation fails and ``fc_fallback_dir`` is set,
        ``fc_fallback`` is used instead.
        """
        from ase.io import read, write
        from os import path, makedirs, getcwd, chdir

        if solvent is not None:
            sol_str = f'in {self.solvstr(solvent)} solvent '
        else:
            sol_str = ''

        # Make directory for optimised structures
        if not path.exists(out_path):
            makedirs(out_path)
        target = calc_params['target']

        for seed in solute_names:
            if target is not None and target != 0:
                baseseed = seed
                seed = f'{seed}_es{str(target)}'
            else:
                baseseed = seed
            # charges are keyed by the molecule name, not the excited-state seed
            if baseseed in charges:
                charge = charges[baseseed]
            else:
                charge = 0
            infile = f'{in_path}/{seed}.xyz'
            outfile = f'{out_path}/{seed}.xyz'
            task = "geometry optimisation"
            solute, nconf, nconf_done = self.get_solute(seed, target, infile, outfile, task, sol_str, in_path, baseseed)
            if solute is None:
                continue
            origdir = getcwd()
            wdir = f'geom/{seed}'
            if not path.exists(wdir):
                makedirs(wdir)
            chdir(wdir)
            for iconf in range(nconf_done, nconf):
                if nconf > 1:
                    confstr = f' conformer {iconf}'
                    confsuff = f'_c{iconf}'
                else:
                    confstr = ''
                    confsuff = ''
                print(f'Geometry optimization {sol_str}for: {seed}{confstr}')
                label = seed + confsuff
                if solvent is not None:
                    label = f'{label}_{self.solvstr(solvent)}'
                try:
                    geom_opt_func(solute[iconf], label, calc_params, driver_tol, solvent=solvent, charge=charge)
                except KeyboardInterrupt:
                    raise Exception('Keyboard Interrupt')
                except SyntaxError:
                    raise Exception('Syntax Error')
                except Exception as e:
                    # Excited-state relaxation can fail (eg heading for an S1/S0 crossing):
                    # if requested, fall back to the ground-state geometry (Franck-Condon point)
                    if target is None or target == 0 or self.fc_fallback_dir is None:
                        raise
                    self.fc_fallback(solute, iconf, label, baseseed, out_path, origdir, wdir, calc_params,
                                     solvent, charge, e)
                if '' in solute[iconf].info:
                    del solute[iconf].info['']
                write(f'{origdir}/{outfile}', solute[0:iconf + 1])

            chdir(origdir)
            print(f'Writing to {outfile}')
            write(outfile, solute)

    def fc_fallback(self, solute, iconf, label, baseseed, out_path, origdir, wdir, calc_params, solvent, charge, error):
        """
        Replaces a failed excited-state geometry optimisation with the ground-state optimised
        geometry (the Franck-Condon point) from the same stage of the ground-state Solutes run,
        found at {fc_fallback_dir}/{out_path}/{baseseed}.xyz relative to this task's directory.

        A singlepoint for the excited state is then run at that geometry under the same label,
        so later readers of this calculation (eg reference energies for Clusters) find a valid
        result. The output of the failed optimisation is kept as {label}_failed_opt.out.
        """
        from ase.io import read
        from os import path, rename, getcwd, chdir
        fc_file = f'{origdir}/{self.fc_fallback_dir}/{out_path}/{baseseed}.xyz'
        print(f'# Excited-state geometry optimisation of {label} failed: {error}')
        # The failed run may have left us in the wrapper's temporary directory: bring its
        # files back and return to this optimisation's working directory before continuing
        work_dir = f'{origdir}/{wdir}'
        if getcwd() != path.realpath(work_dir) and hasattr(self.wrapper, 'return_from_tempdir'):
            try:
                self.wrapper.return_from_tempdir(label)
            except Exception as e2:
                print(f'# Could not recover files from the temporary directory: {e2}')
        chdir(work_dir)
        if not path.isfile(fc_file):
            raise Exception(f'# Franck-Condon fallback geometry {fc_file} not found: '
                            f'the ground-state Solutes run must finish this stage first') from error
        print(f'# Falling back to the ground-state (Franck-Condon) geometry from {fc_file}')
        fc_frames = read(fc_file, index=':')
        solute[iconf].positions = fc_frames[min(iconf, len(fc_frames) - 1)].positions
        for ext in ['.out', '.engrad']:
            if path.isfile(label + ext):
                rename(label + ext, f'{label}_failed_opt{ext}')
        self.wrapper.singlepoint(solute[iconf], label, calc_params, solvent=solvent, charge=charge,
                                 forces=True, dipole=False)
        solute[iconf].info['fc_fallback'] = True

    # Attempt to find best rotamer for each solute
    def find_best_rotas(self, solute_names, in_path, out_path, singlepoint_func,
                        geom_opt_func, calc_params, solvent=None, charges={}):
        """
        Finds the lowest energy rotamer for each of a list of solutes.
        Proceeds by identifying -OH groups attached to C-C units, and tries 'flipping' the dihedral, then optimising
        the resulting geometry if its energy is within 0.2 (in the wrapper's energy units, normally eV) of the
        original energy. If any lower energy structure is found, this will be returned instead of the original one.

        solute_names: list of str
            Short names of the solutes to be tested
        in_path: str
            Directory where .xyz files are expected to be found. Any not present are skipped.
        out_path: str
            Directory where best rotamer structure .xyz files are written. Created if not present.
        singlepoint_func: function
            A function wrapping creation of an ASE calculator and using it to perform a singlepoint calculation.
        geom_opt_func: function
            A function wrapping creation of an ASE calculator and using it to perform geometry optimisation.
        calc_params: dict
            Contents varies between different wrappers, but generally specifies basis, functional etc
        solvent: str
            Implicit solvent name, or None for gas-phase (currently not passed to the calculations)
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule
        """
        from os import path, makedirs, getcwd, chdir
        from ase.io import read, write

        # Hard-coded logic for what constitutes a rotatable bond
        # Works OK for -OH groups in organic compounds but will need editing
        # for anything else. Assumes anything within 1.5A is a bond.
        rota_elem = ['H', 'O', 'C', 'C']
        rota_max_dist = [1.5, 1.5, 1.5]
        rota_dih_range = 40
        rota_opt_thresh = 0.2

        # Make directory for optimised structures
        if not path.exists(out_path):
            makedirs(out_path)
        target = calc_params['target']

        for seed in solute_names:
            if seed in charges:
                charge = charges[seed]
            else:
                charge = 0
            if target is not None and target != 0:
                seed = seed + "_es" + str(target)
            outfile = out_path + '/' + seed + '.xyz'
            infile = in_path + "/" + seed + ".xyz"
            if not path.exists(infile):
                print('Skipping Rotamer Search for: ', seed,
                      ' - input structure not found')
                continue
            if path.exists(outfile):
                print('Skipping Rotamer Search for: ', seed,
                      ' - output file already present')
                continue

            print('Finding best rotamer for: ', seed)
            sol_opt = read(infile)
            # Load defaults from module
            elem = rota_elem
            max_dist = rota_max_dist
            dih_range = rota_dih_range
            opt_thresh = rota_opt_thresh
            nrot = 0
            ijkl = []
            sym = sol_opt.get_chemical_symbols()
            for i in range(len(sol_opt)):
                if sym[i] == elem[0]:
                    for j in range(len(sol_opt)):
                        if sym[j] == elem[1] and sol_opt.get_distance(i, j) < max_dist[0]:
                            for k in range(len(sol_opt)):
                                if sym[k] == elem[2] and sol_opt.get_distance(j, k) < max_dist[1]:
                                    for l in range(len(sol_opt)):
                                        if l != k and sym[l] == elem[3] and sol_opt.get_distance(k, l) < max_dist[2]:
                                            dih = sol_opt.get_dihedral(l, k, j, i)
                                            if dih > 180 - dih_range and dih < 180 + dih_range:
                                                ijkl.append([i, j, k, l])
                                                nrot = nrot + 1
                                                break
            if nrot == 0:
                print('# No rotatable OH bonds found')
                if '' in sol_opt.info:
                    del sol_opt.info['']
                write(outfile, sol_opt)
            else:
                print(nrot, 'rotatable bonds found, generating all', 2**nrot,
                      'rotamers')
                origdir = getcwd()
                wdir = f'{out_path}/{seed}'
                if not path.exists(wdir):
                    makedirs(wdir)
                chdir(wdir)
                flip = [0 for i in range(len(ijkl))]
                sol_opt_rota = []
                energy_rota = []
                for rota in range(2**len(ijkl)):
                    sol_opt_rota.append(sol_opt.copy())
                    flip = [(rota & (2**(len(ijkl) - i - 1))) >> (len(ijkl) - i - 1) for i in range(len(ijkl))]
                    for oH in range(len(ijkl)):
                        i = ijkl[oH][0]
                        j = ijkl[oH][1]
                        k = ijkl[oH][2]
                        l = ijkl[oH][3]
                        if flip[oH]:
                            sol_opt_rota[rota].set_dihedral(l, k, j, i, 0)
                    label = 'rota' + repr(rota).zfill(3)
                    driver_tol = 'loose'
                    opt = 0
                    try:
                        energy, _ = singlepoint_func(sol_opt_rota[rota], label, calc_params, charge=charge)
                        if rota == 0:
                            energy_rota.append(energy)
                        if rota > 0:
                            if energy < energy_rota[0] + opt_thresh:
                                opt = 1
                                energy, forces, positions = geom_opt_func(sol_opt_rota[rota],
                                                                          label + "_geom", calc_params, driver_tol, charge=charge)
                                write(label + '.xyz', sol_opt_rota[rota])
                            energy_rota.append(energy)
                        print(rota, ':', flip, energy_rota[rota], 'opt=', opt)
                    except KeyboardInterrupt:
                        raise Exception('Keyboard Interrupt')
                    except:
                        print('Rotamer energy calculation failed for: ', seed)
                chdir(origdir)

                min_energy, best_rota = min((energy_rota[i], i) for i in range(len(energy_rota)))
                write(outfile, sol_opt_rota[best_rota])

    def rotate_all_to_xy_plane(self, solute_names, in_path, out_path, target=None):
        """
        Rotates each of a list of solutes so that the plane defined by its two furthest-apart pairs of C atoms
        lies in the xy plane, and centres it on the origin, using ``clusters.rotate_and_center_solute``.
        Molecules with fewer than two C atoms are copied unchanged, and those of 10 atoms or fewer are
        centred but not rotated.

        solute_names: list of str
            Short names of the solutes to be tested
        in_path: str
            Directory where .xyz files are expected to be found. Any not present are skipped.
        out_path: str
            Directory where rotated structure .xyz files are written. Created if not present.
        target: int
            Excited state index, or None for ground state
        """
        from os import path, makedirs
        from ase.io import read, write

        # Make directory for rotated structures
        if not path.exists(out_path):
            makedirs(out_path)

        # Load optimised xyz files and write rotated/translated ones
        for seed in solute_names:
            if target is not None and target != 0:
                seed = seed + "_es" + str(target)
            infile = in_path + "/" + seed + ".xyz"
            outfile = out_path + "/" + seed + ".xyz"
            if not path.exists(infile):
                print(f'Skipping rotate and centre operation for: {seed}',
                      '- no input file')
                continue
            try:
                rot = read(infile, 0)
            except FileNotFoundError:
                print(f'Could not read file: {infile}')
                continue
            if rot.get_chemical_symbols().count('C') < 2:
                print(f'Skipping rotate and centre operation for: {seed}',
                      '- not enough C atoms')
                write(outfile, rot)
                continue
            try:
                # Attempt to rotate molecules to lie in xy-plane
                # with long axis along x, short axis along y
                # and central C(=O) atom put at (10,10,10)
                from esteem.tasks.clusters import rotate_and_center_solute
                rot = rotate_and_center_solute(rot)
            except Exception as e:
                raise Exception(f'Rotate and centre failed for {seed} with exception: {e}')
            if '' in rot.info:
                del rot.info['']
            write(outfile, rot)

    import numpy as np
    from os import path, getcwd, chdir
    from ase.io import read

    def find_range_sep(self, solute_names, in_path, out_path, wrapper,
                       calc_params, solvent=None, charges={}, rs_range=[0.1, 0.2, 0.3], all_readonly=False):
        r'''
        Runs the calculations needed to optimise the range separation parameter gamma in a
        range-separated hybrid functional with Yukawa-switching.
        See 'Using optimally tuned range separated hybrid functionals in ground-state
        calculations: Consequences and caveats',  Andreas Karolewski, Leeor Kronik, and Stephan Kümmel
        J. Chem. Phys. 138, 204115 (2013)
        https://aip.scitation.org/doi/10.1063/1.4807325
        https://pubs.acs.org/doi/abs/10.1021/ct5000617
        and 'Electronic Band Shapes Calculated with Optimally Tuned Range-Separated Hybrid Functionals'
        B. Moore, et al, D. Jacquemin, J. Chem. Theory Comput. 2014, 10, 10, 4599
        https://pubs.acs.org/doi/10.1021/ct500712w

        The optimal gamma minimises::

            J^2 = \Sum_i=0^1 [eps_H[N+i] + IP(N+i)))]^2
            where IP(N) = E(N-1) - E(N)
            so J^2 = \Sum_i=0^1 [eps_H[N+i] + E(N-1+i) - E(N+i)]^2
                   = [eps_H[N] + E(N-1) - E(N)]^2 + [eps_H[N+1] + E(N) - E(N+1)]^2

        For each value of gamma in ``rs_range``, singlepoint calculations are run with
        ``wrapper.singlepoint`` for net charges Q, Q+1 and Q-1, in '{out_path}/{seed}'. The
        minimisation of J^2 itself is not performed here.

        solute_names: list of str
            Short names of the solutes, read from '{in_path}/{seed}.xyz'
        wrapper: class
            Wrapper whose ``singlepoint`` returns the energy and a calculator object
        calc_params: dict
            Calculator parameters; ':{gamma}' is appended to its 'func' entry for each calculation
        solvent: str
            Implicit solvent name, or None for gas-phase
        charges: dict
            Net charge Q of each solute, keyed by shortname (default 0)
        rs_range: list of float
            Values of gamma to try
        all_readonly: bool
            If True, read existing outputs rather than running any calculations

        *Returns*: dicts ``energy``, ``calc``, ``evals``, ``occs``, keyed by (gamma string, charge),
        for the first solute only.
        '''
        from os import path, makedirs, getcwd, chdir
        from ase.io import read
        from ase.units import Ha
        wdir = f'{out_path}'
        if not path.exists(wdir):
            makedirs(wdir)

        for seed in solute_names:
            if seed in charges:
                charge = charges[seed]
            else:
                charge = 0

            # Find input and check if output already exists
            infile = in_path + "/" + seed + ".xyz"
            if not path.exists(infile):
                print('Skipping optimal RS tuning for: ', seed, ' - no input file')

            # Read the geometry
            solute = read(infile)

            # Check if output directory exists, create it if not, and change to it
            origdir = getcwd()
            wdir = f'{out_path}/{seed}'
            if not path.exists(wdir):
                makedirs(wdir)
            chdir(wdir)

            func = calc_params['func']
            basis = calc_params['basis']
            energy = {}
            calc = {}
            evals = {}
            occs = {}
            rs_mid = 0.4
            rs_del = 0.1
            for rsf in rs_range:
                for c in [charge, charge + 1, charge - 1]:
                    rs = f'{rsf:.2f}'
                    label = f"ot_pars_cam{rs}_Q{c}"
                    outfile = f'{label}'
                    print(f'Optimal RS tuning with rs={rs} for: {seed} (Q={c}): ({basis} {func} {self.solvstr(solvent)})')
                    if path.exists(outfile + ".out") or path.exists(outfile + ".nwo") or all_readonly:
                        print('Skipping - Output file already present')
                        readonly = True
                    else:
                        readonly = False
                    # Assume when c==charge, system is closed-shell
                    if c != charge:
                        s = 0.5
                        occfac = 0.99
                    else:
                        s = 0
                        occfac = 1.99
                    calc_params['func'] = func + f':{rs}'
                    energy[rs, c], calc[rs, c] = wrapper.singlepoint(solute, label, calc_params,
                                                                     solvent=solvent, charge=c, spin=s, readonly=readonly)
                    print(rs, c, energy[rs, c] / Ha)
                    evals[rs, c] = calc[rs, c].calc.kpts[0].eps_n
                    occs[rs, c] = calc[rs, c].calc.kpts[0].f_n
                    if c != charge:
                        evals[rs, c, s] = calc[rs, c].calc.kpts[1].eps_n
                        occs[rs, c, s] = calc[rs, c].calc.kpts[1].f_n
            chdir(origdir)
            return energy, calc, evals, occs

    # Calculate Excited states of all molecules
    def calc_all_excited_states(self, solute_names, in_path, out_path, excit_func,
                                calc_params, nroots, solvent=None, charges={},
                                plot_homo=None, plot_lumo=None, plot_trans_den=None):
        """
        Calculate excited states for each of a list of solutes

        solute_names: list of str
            Short names of the solutes to be tested
        in_path: str
            Directory where .xyz files are expected to be found. Any not present are skipped.
        out_path: str
            Directory where output files are written. Created if not present.
        excit_func: function
            A function wrapping creation of an ASE calculator and using it to perform a electronic excitation calculations.
        calc_params: dict
            Contents varies between different wrappers, but generally specifies basis, functional etc
        nroots: int
            Number of excitations to find
        solvent: str or dict
            Implicit solvent name (or dict of solvent settings), or None for gas-phase
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule
        plot_homo, plot_lumo, plot_trans_den: int
            Numbers of occupied orbitals, unoccupied orbitals and transition densities to write as cube files,
            or None for none (passed to the wrapper)

        The excited state whose geometry is read (``{seed}_es{target}``) is taken from ``calc_params['target']``.
        Calculations run in '{out_path}/{seed}', and the geometries are copied to '{out_path}/{seed}.xyz'.
        """
        from ase.io import read, write
        from os import path, makedirs, getcwd, chdir

        if solvent is not None:
            sol_str = f'in {self.solvstr(solvent)} solvent '
        else:
            sol_str = ''
        target = calc_params['target']
        method = 'tddft'
        wdir = f'{out_path}'
        if not path.exists(wdir):
            makedirs(wdir)
        for seed in solute_names:
            if seed in charges:
                charge = charges[seed]
            else:
                charge = 0
            if target is not None and target != 0:
                seed = seed + "_es" + str(target)

            # Find input and check if output already exists
            infile = f'{in_path}/{seed}.xyz'
            outfile = f'{out_path}/{seed}.xyz'
            task = f'{method} excitations'
            solute, nconf, nconf_done = self.get_solute(seed, target, infile, outfile, task, sol_str, in_path, seed)
            if solute is None:
                continue

            # Check if output directory exists, create it if not, and change to it
            origdir = getcwd()
            wdir = f'{out_path}/{seed}'
            if not path.exists(wdir):
                makedirs(wdir)
            chdir(wdir)
            for iconf in range(nconf_done, nconf):
                if nconf > 1:
                    confstr = f' conformer {iconf}'
                    confsuff = f'_c{iconf}'
                else:
                    confstr = ''
                    confsuff = ''
                print(f'Excitation Calculation {sol_str}for: {seed}{confstr}')

                # Now run the excitations function of the wrapper
                label = f'{seed}{confsuff}_{method}'
                readonly = iconf < nconf_done
                try:
                    excit_func(solute[iconf], label, calc_params, nroots, solvent, charge, readonly=readonly,
                               plot_homo=plot_homo, plot_lumo=plot_lumo, plot_trans_den=plot_trans_den)
                except KeyboardInterrupt:
                    raise Exception('Keyboard Interrupt')
                except SyntaxError as e:
                    raise Exception('Syntax Error:', e)
                except Exception as e:
                    print('Excited state calculation failed for: ', seed)
                    import traceback
                    traceback.print_exception(e)
                write(origdir + "/" + outfile, solute[0:iconf + 1])
            chdir(origdir)

    # Calculate vibrational frequencies of all molecules
    def calc_vib_freq(self, solute_names, in_path, out_path, freq_func,
                      calc_params, solvent=None, charges={}):
        """
        Calculate vibrational frequencies for each of a list of solutes

        solute_names: list of str
            Short names of the solutes to be tested
        in_path: str
            Directory where .xyz files are expected to be found. Any not present are skipped.
        out_path: str
            Directory where output files are written. Created if not present.
        freq_func: function
            A function wrapping creation of an ASE calculator and using it to perform a vibrational frequency calculation.
        calc_params: dict
            Contents varies between different wrappers, but generally specifies basis, functional etc
        solvent: str or dict
            Implicit solvent name (or dict of solvent settings), or None for gas-phase
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule

        The excited state whose geometry is read (``{seed}_es{target}``) is taken from ``calc_params['target']``.
        Calculations run in '{out_path}/{seed}', and the geometries are copied to '{out_path}/{seed}.xyz'.
        """
        from ase.io import read, write
        from os import path, makedirs, getcwd, chdir

        if solvent is not None:
            sol_str = f'in {self.solvstr(solvent)} solvent '
        else:
            sol_str = ''
        wdir = f'{out_path}'
        target = calc_params['target']
        if not path.exists(wdir):
            makedirs(wdir)
        for seed in solute_names:
            if seed in charges:
                charge = charges[seed]
            else:
                charge = 0
            if target is not None and target != 0:
                seed = seed + "_es" + str(target)
            method = 'freq'
            task = 'vibrational frequencies'
            infile = f'{in_path}/{seed}.xyz'
            outfile = f'{out_path}/{seed}.xyz'
            solute, nconf, nconf_done = self.get_solute(seed, target, infile, outfile, task, sol_str, in_path, seed)
            if solute is None:
                continue
            origdir = getcwd()
            wdir = f'{out_path}/{seed}'
            if not path.exists(wdir):
                makedirs(wdir)
            chdir(wdir)
            for iconf in range(nconf_done, nconf):
                if nconf > 1:
                    confstr = f' conformer {iconf}'
                    confsuff = f'_c{iconf}'
                else:
                    confstr = ''
                    confsuff = ''
                label = f'{seed}{confsuff}_{method}'
                print(f'Vibrational Frequencies {sol_str}for: {seed}{confstr}')
                try:
                    freq_func(solute[iconf], label, calc_params, solvent, charge)
                except KeyboardInterrupt:
                    raise Exception('Keyboard Interrupt')
                except SyntaxError:
                    raise Exception('Keyboard Interrupt')
                except Exception as e:
                    print(f'Vibrational Frequencies {sol_str}failed for: {seed}')
                    print('Exception was: ', e)
                write(origdir + "/" + outfile, solute[0:iconf + 1])
            chdir(origdir)

    # Create a parser for the solutes program
    def make_parser(self):

        import argparse

        # Parse command line values
        main_help = ('Prepares DFT-optimised model structures for a list of solutes.\n' +
                     'for each solvent, loops over various tasks:\n' +
                     'download xyz, geom opt, TDDFT\n' +
                     'rotamer search, rotation to xy-plane,\n' +
                     'solvated geom opt and TDDFT.')
        epi_help = ('Names are read in from the file provided in the "namefile"\n' +
                    'argument. All other\n' +
                    'arguments are optional and turn on other tasks. To provide\n' +
                    'pre-optimised inputs, put their structures in a pre-existing\n' +
                    'directory named "xyz". If not present, an attempt will be\n' +
                    'made to download a structure from the NIH chemical name\n' +
                    'resolver (CACTUS).')
        parser = argparse.ArgumentParser(description=main_help, epilog=epi_help)
        parser.add_argument('--namefile', '-N', default='names', type=str, help="File listing the molecules to process, one per line as 'shortname: full name'; the full name is used to download a starting geometry from PubChem if xyz/shortname.xyz does not exist. Only read when solutes.py is run directly from the command line; workflow scripts pass the molecules to the driver instead.")
        parser.add_argument('--solvent', '-S', default=None, type=str, help="Implicit solvent for the current run, or None for gas phase. The solutes and solvents drivers set this for each entry of all_solvents in turn (a solvent named 'gas' becomes None), so scripts do not normally set it; it may also be a dict with key 'solvent' plus wrapper-specific settings (see solvent_settings).")
        parser.add_argument('--nroots', '-n', default=5, type=int, help="Number of TDDFT excited states to calculate, in gas phase (output in 'tddft') when there is no solvent, or in implicit solvent (output in 'is_tddft_<solvent>'). Set to 0 to skip the excited-state calculations.")
        parser.add_argument('--target', '-t', default=None, type=int, help='Electronic state whose geometry is optimised: None or 0 for the ground state, or an integer n for the nth TDDFT excited state, in which case output files are named <solute>_es<n>. Copied into calc_params unless calc_params is set explicitly.')
        parser.add_argument('--rotate', '-r', default=True, type=bool, help="If True, rotate each gas-phase optimised molecule so that the plane defined by its furthest-apart pairs of C atoms lies in the xy-plane, and centre it on the origin (output in 'opt_rot'). Molecules with fewer than two C atoms, or 10 atoms or fewer, are not rotated.")
        parser.add_argument('--rotamers', '-R', default=False, nargs='?', const=True, type=bool, help="If True, search for a lower-energy rotamer of each gas-phase optimised molecule by flipping all rotatable C-C-O-H dihedrals, re-optimising any rotamer within 0.2 eV of the original and keeping the lowest-energy one (output in 'best_rota').")
        parser.add_argument('--vibrations', '-V', default=False, nargs='?', const=True, type=bool, help="If True, calculate vibrational frequencies at the final level of theory: in implicit solvent (output in 'is_freq_<solvent>') when a solvent is set, otherwise in gas phase (output in 'freq'). The preceding geometry optimisation then uses tight convergence.")
        parser.add_argument('--charges', '-Q', default={}, nargs='?', type=dict, help="Net charge of each molecule, as a dict keyed by molecule shortname, eg {'AQS': -1}; molecules not listed are neutral. Set this in a script: it cannot be given on the command line.")
        parser.add_argument('--directory', '-D', default=None, nargs='?', type=str, help="Directory, relative to where the script is run, in which this task's calculations are run, eg 'gs_PBE0'. It is created if needed and the 'xyz' directory is copied into it; None runs in the current directory.")
        parser.add_argument('--fc_fallback_dir', default=None, type=str, help="Directory of a ground-state Solutes run, relative to this task's directory (eg '../gs_PBE0'). If set, an excited-state geometry optimisation that fails is replaced by the ground-state geometry (Franck-Condon point) from the same stage of that run, followed by an excited-state singlepoint.")
        parser.add_argument('--calc_params', '-Z', default={}, nargs='?', type=str, help="Dict of calculator parameters passed to the wrapper for every calculation. If empty or None, it is built from basis, func, target and disp when the task runs; if set explicitly, it must contain the keys the wrapper expects (normally 'basis', 'func', 'target' and 'disp'), and those four attributes are then ignored.")
        # Wrapper Dependent
        parser.add_argument('--solvent_settings', default=None, type=str, help="Optional dict of extra implicit-solvent settings for the wrapper, eg {'cosmo_smd': True, 'cosmo_vem': 0} for NWChem. If set, the drivers pass this dict, with its 'solvent' key set to the current solvent name, as the solvent for each run; None uses just the solvent name.")
        parser.add_argument('--plot_homo', '-H', default=None, type=int, help='Number of occupied orbitals, counting down from the HOMO, to write as cube files after each implicit-solvent TDDFT calculation; None plots none. Currently only used by the NWChem wrapper.')
        parser.add_argument('--plot_lumo', '-L', default=None, type=int, help='Number of unoccupied orbitals, counting up from the LUMO, to write as cube files after each implicit-solvent TDDFT calculation; None plots none. Currently only used by the NWChem wrapper.')
        parser.add_argument('--plot_trans_den', '-T', default=None, type=int, help='Number of excitations, starting from the lowest, for which to write the transition density as a cube file after each implicit-solvent TDDFT calculation; None plots none. Currently only used by the NWChem wrapper.')
        parser.add_argument('--basis', '-b', default='6-311++G**', type=str, help="Basis set name, in the form expected by the wrapper's code (eg '6-311++G**' or 'def2-TZVP'), used for all geometry optimisations, TDDFT and frequency calculations. Copied into calc_params unless calc_params is set explicitly.")
        parser.add_argument('--func', '-f', default='PBE0', type=str, help="Exchange-correlation functional name, in the form expected by the wrapper's code (eg 'PBE0' or 'CAM-B3LYP'), used for all geometry optimisations, TDDFT and frequency calculations. Copied into calc_params unless calc_params is set explicitly.")
        parser.add_argument('--disp', '-d', default=True, type=bool, help='If True, add a Grimme D3 dispersion correction (D3BJ in ORCA, D3 in NWChem; not supported by OpenQP). Copied into calc_params unless calc_params is set explicitly.')

        return parser

    def validate_args(self, args):
        default_args = self.make_parser().parse_args("")
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


def get_parser():
    return SolutesTask().make_parser()


# Main program
if __name__ == '__main__':
    args = get_parser().parse_args()
    if (args.target == 0):
        args.target = None
    print(args)
    sols = SolutesTask()

    # get list of molecules and sanitize
    with open(args.namefile) as f:
        namelist = f.readlines()
    namelist = [x.split(":") for x in namelist]
    namelist = [[y.strip() for y in x] for x in namelist]

    # duplicate if a shortname has not been assigned
    namelist = [[x[0], x[0]] if len(x) == 1 else x for x in namelist]

    # make it into a dictionary
    namelist = {x[0]: x[1] for x in namelist}

    # download xyz files from cactus if not already present
    namelist = sols.get_xyz_files(namelist, "xyz")

    # run main driver
    from esteem.wrappers import nwchem
    sols.wrapper = nwchem.NWChemWrapper()
    sols.run(namelist)
