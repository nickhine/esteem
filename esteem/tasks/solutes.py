#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Runs DFT and TDDFT calculations on Solute (or Solvent) molecules, in implicit solvent"""


# # Main driver routine

# In[ ]:


class SolutesTask:

    def __init__(self,**kwargs):
        self.wrapper = None
        self.script_settings = None        
        default_args = self.make_parser().parse_args("")
        # Set values of all parameters (defaults unless overridden)
        for arg in vars(default_args):
            if arg in kwargs:
                val = kwargs[arg]
            else:
                val = getattr(default_args,arg)
            setattr(self,arg,val)
    
    def run(self,namelist):
        """
        Main routine for the Solutes task. Iterates through the sub-tasks, some of which are optional:

        Geometry optimisation, rotamer checks, rotation to plane, excited state calculations,
        and vibrational frequency calculations.

        namelist: list of str
            Short names of the solutes to be optimised. Initial geometries
            should be present in './xyz' directory.
        self: namespace or class
            Member variables contain argument list for the whole job, with members including:

            ``solvent``, ``target``, ``rotate``, ``rotamers``, ``vibrations``, ``charges``, ``directory``
            ``solvent_settings``, ``basis``, ``func``, ``nroots``

            Generate default arguments first with a call to solutes.make_parser(), then adjust the
            member variables as required.
        wrapper: class
            Wrapper for running components of the job, with members including:

            ``singlepoint``, ``geom_opt``, ``excitations``, ``freq``

        *Outputs for each species*:

            In the directory 'xyz': initial geometries (either input, or from get_xyz_files)

            In the directory 'geom': outputs of geometry optimisations.

            In the directory 'opt': optimised gas-phase geometries.

            In the directory 'is_opt': optimised implicit solvent geometries. These may be used as inputs for the Solvate task.

            In the directory 'is_tddft_{solvent}': implicit solvent tddft results. These may be used as inputs for spectral warping in the Spectra task.

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
        self.geom_opt_all(namelist,prevdir,nextdir,self.wrapper.geom_opt,self.calc_params,
                     driver_tol,charges=self.charges)
        prevdir = nextdir
        # Rotamer search, if requested
        if self.rotamers:
            nextdir = "best_rota"
            self.find_best_rotas(namelist,prevdir,nextdir,self.wrapper.singlepoint,
                            self.wrapper.geom_opt,self.calc_params,charges=self.charges)
            prevdir = nextdir
        if self.rotate:
            nextdir = "opt_rot"
            self.rotate_all_to_xy_plane(namelist,prevdir,nextdir,target=self.target)
            prevdir = nextdir
        if self.nroots > 0 and self.solvent is None:
            nextdir = "tddft"
            self.calc_all_excited_states(namelist,prevdir,nextdir,self.wrapper.excitations,
                                    self.calc_params,self.nroots,charges=self.charges)
        # Vibrational frequency calculations, if requested and no higher
        # level of theory will follow later
        if self.vibrations and self.solvent is None:
            nextdir = "freq"
            self.calc_vib_freq(namelist,prevdir,nextdir,self.wrapper.freq,self.calc_params,charges=self.charges)
        # Solvated calculations
        if self.solvent is not None:
            nextdir = "is_opt_"+self.solvstr(self.solvent)
            if self.vibrations:
                driver_tol = 'tight'
            # Geometry Optimisation
            self.geom_opt_all(namelist,prevdir,nextdir,self.wrapper.geom_opt,
                         self.calc_params,driver_tol,solvent=self.solvent,
                         charges=self.charges)
            prevdir = nextdir
            # TDDFT calculations
            if self.nroots > 0:
                nextdir = "is_tddft_"+self.solvstr(self.solvent)
                self.calc_all_excited_states(namelist,prevdir,nextdir,self.wrapper.excitations,
                                        self.calc_params,nroots=self.nroots,
                                        solvent=self.solvent,charges=self.charges,
                                        plot_homo=self.plot_homo,plot_lumo=self.plot_lumo,
                                        plot_trans_den=self.plot_trans_den)
            # Vibrational frequency calculations, if requested
            if self.vibrations:
                nextdir = "is_freq_"+self.solvstr(self.solvent)
                self.calc_vib_freq(namelist,prevdir,nextdir,self.wrapper.freq,
                              self.calc_params,solvent=self.solvent,
                              charges=self.charges)

    from os import makedirs, path

    # Download xyz files if they do not already exist
    def get_xyz_files(self,namelist,out_path):
        """
        Downloads initial geometries from the NCI's webserver cactus, based on their IUPAC names.

        These geometries are usually not great, but are a reasonable starting point for optimisation.

        Visit https://cactus.nci.nih.gov/chemical/structure to see what works and check your names before use.

        Uses ``wget``, so if the machine you are using does not have access to this command,
        this routine will fail, in which case put starting point geometries in the directory
        'xyz'.

        namelist: dict
            Keys are shortnames (eg "cate"), entries are full names (eg "catechol" or "1,2-dihydroxybenzene")
        out_path: str
            String for directory name where xyz files will be written (eg "xyz"). Created if not present
        """
        from os import path, makedirs
        import subprocess

        # Download sdf file from cactus.nci.nih.gov
        if not path.exists(out_path):
            makedirs(out_path)
        for seed in namelist:
            # TODO: use urllib here
            wget_str = ("wget -O \""+out_path+"/" + seed + ".xyz\" " +
                        "\"https://cactus.nci.nih.gov/chemical/structure/" + 
                        namelist[seed] + "/file?format=xyz\"")
            if not path.exists(out_path+"/"+seed+".xyz"):
                print(wget_str)
                errorcode = subprocess.call(wget_str, shell=True)
                if errorcode:
                    raise RuntimeError('{} returned an error: {}'
                                                .format('wget', errorcode))
            else:
                print("Skipping download: "+out_path+"/"+seed+".xyz already exists")

        # strip out long names and convert to list
        shortnames = [x for x in namelist]
        return shortnames

    from os import path, makedirs, getcwd, chdir
    from ase.io import read, write

    def solvstr(self,solvent):
        if isinstance(solvent,str):
            return solvent
        if isinstance(solvent,dict):
            return solvent['solvent']

    # Optimize geometries of solute selection
    def geom_opt_all(self,solute_names,in_path,out_path,geom_opt_func,calc_params,
                     driver_tol='default',solvent=None,charges={}):
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
            Geometry optimisation tolerance level (eg in NWChem)
        target: int
            Excited state index, or None for ground state
        solvent: str
            Implicit solvent name, or None for gas-phase
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule
        """
        from ase.io import read, write
        from os import path, makedirs, getcwd, chdir

        # Make directory for optimised structures
        if not path.exists(out_path):
            makedirs(out_path)
        sol_str = ''
        target = calc_params['target']
        if solvent is not None:
            sol_str = f'in {self.solvstr(solvent)} solvent '

        for seed in solute_names:
            if target is not None and target!=0:
                baseseed = seed
                seed = seed+"_es"+str(target)
            if seed in charges:
                charge = charges[seed]
            else:
                charge = 0
            infile = in_path+"/"+seed+".xyz"
            outfile = out_path+"/"+seed+".xyz"
            if not path.exists(infile):
                # Try basename without _esX
                if target is not None and target !=0:
                    infile = in_path+"/"+baseseed+".xyz"
                if not path.exists(infile):
                    print(f'Skipping geometry optimisation {sol_str}'+
                          f' for: {seed} - no input file')
                    continue
            solute_opt = read(infile)
            if  path.exists(outfile):
                print(f'Skipping geometry optimisation {sol_str}'+
                      f' for: {seed} - output file already present')
                continue
            print(f'Geometry optimization {sol_str}for: {seed}')
            label = seed
            origdir = getcwd()
            wdir = f'geom/{seed}'
            if not path.exists(wdir):
                makedirs(wdir)
            chdir(wdir)
            if solvent is not None:
                label = label+"_"+self.solvstr(solvent)
            try:
                geom_opt_func(solute_opt,label,calc_params,driver_tol,solvent,charge)
            except KeyboardInterrupt:
                raise Exception('Keyboard Interrupt')
            except SyntaxError:
                raise Exception('Syntax Error')

            chdir(origdir)
            print('Writing to ',outfile)
            if '' in solute_opt.info:
                del solute_opt.info['']
            write(outfile,solute_opt)

    # Attempt to find best rotamer for each solute

    def find_best_rotas(self,solute_names,in_path,out_path,singlepoint_func,
                        geom_opt_func,calc_params,solvent=None,charges={}):
        """
        Finds the lowest energy rotamer for each of a list of solutes.
        Proceeds by identifying -OH groups attached to C-C units, and tries 'flipping' the dihedral, then optimising
        the resulting geometry if it within a certain tolerance of the original energy. If any lower energy structure
        is found, this will be returned instead of the original one.

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
            Implicit solvent name, or None for gas-phase
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule
        """
        from os import path, makedirs
        from ase.io import read, write

        # Hard-coded logic for what constitutes a rotatable bond
        # Works OK for -OH groups in organic compounds but will need editing
        # for anything else. Assumes anything within 1.5A is a bond.
        rota_elem = ['H','O','C','C']
        rota_max_dist = [1.5,1.5,1.5]
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
            if target is not None and target!=0:
                seed = seed+"_es"+str(target)
            outfile = out_path+'/'+seed+'.xyz'
            infile = in_path+"/"+seed+".xyz"
            if not path.exists(infile):
                print('Skipping Rotamer Search for: ',seed,
                      ' - input structure not found')
                continue
            if path.exists(outfile):
                print('Skipping Rotamer Search for: ',seed,
                      ' - output file already present')
                continue

            print('Finding best rotamer for: ',seed)
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
                if sym[i]==elem[0]:
                    for j in range(len(sol_opt)):
                        if sym[j]==elem[1] and sol_opt.get_distance(i,j) < max_dist[0]:
                            for k in range(len(sol_opt)):
                                if sym[k]==elem[2] and sol_opt.get_distance(j,k) < max_dist[1]:
                                    for l in range(len(sol_opt)):
                                        if l!=k and sym[l]==elem[3] and sol_opt.get_distance(k,l) < max_dist[2]:
                                            dih = sol_opt.get_dihedral(l,k,j,i)
                                            if dih>180-dih_range and dih<180+dih_range:
                                                ijkl.append([i,j,k,l])
                                                nrot = nrot + 1
                                                break
            if nrot==0:
                print('No rotatable OH bonds found')
                if '' in sol_opt.info:
                    del sol_opt.info['']
                write(outfile,sol_opt)
            else:
                print(nrot, 'rotatable bonds found, generating all',2**nrot,
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
                    flip = [(rota&(2**(len(ijkl)-i-1)))>>(len(ijkl)-i-1) for i in range(len(ijkl))]
                    for oH in range(len(ijkl)):
                        i = ijkl[oH][0]; j = ijkl[oH][1]; k = ijkl[oH][2]; l = ijkl[oH][3];
                        if flip[oH]:
                            sol_opt_rota[rota].set_dihedral(l,k,j,i,0)
                    label = 'rota'+repr(rota).zfill(3)
                    driver_tol = 'loose'
                    opt = 0
                    try:
                        energy,_ = singlepoint_func(sol_opt_rota[rota],label,calc_params,charge)
                        if rota==0:
                            energy_rota.append(energy)
                        if rota>0:
                            if energy<energy_rota[0]+opt_thresh:
                                opt = 1
                                energy, forces, positions = geom_opt_func(sol_opt_rota[rota],
                                                label+"_geom",calc_params,driver_tol,charge)
                                write(label+'.xyz',sol_opt_rota[rota])
                            energy_rota.append(energy)
                        print(rota,':',flip,energy_rota[rota],'opt=',opt)
                    except KeyboardInterrupt:
                        raise Exception('Keyboard Interrupt')
                    except:
                        print('Rotamer energy calculation failed for: ',seed)
                chdir(origdir)

                min_energy, best_rota = min( (energy_rota[i],i) for i in xrange(len(energy_rota)) )
                write(outfile,sol_opt_rota[best_rota])

    # Attempt to rotate molecules to lie in xy-plane
    # with long axis along x, short axis along y
    # and central C(=O) atom put at (10,10,10)
    from esteem.tasks.clusters import rotate_and_center_solute

    def rotate_all_to_xy_plane(self,solute_names,in_path,out_path,target=None):
        """
        Rotates each of a list of solutes so that the longest two C-C distances lie in the xy plane.

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
        from ase.io import read,write

        # Make directory for rotated structures
        if not path.exists(out_path):
            makedirs(out_path)

        # Load optimised xyz files and write rotated/translated ones
        for seed in solute_names:
            if target is not None and target!=0:
                seed = seed+"_es"+str(target)
            infile = in_path+"/"+seed+".xyz"
            outfile = out_path+"/"+seed+".xyz"
            if not path.exists(infile):
                print('Skipping rotate and centre operation for: ',seed,
                      ' - no input file')
                continue
            try:
                rot = read(infile,0)
            except FileNotFoundError:
                print(f'Could not read file: {infile}')
                continue
            if rot.get_chemical_symbols().count('C') < 2:
                print('Skipping rotate and centre operation for: ',seed,
                      ' - not enough C atoms')
                write(outfile,rot)
                continue
            try:
                rot = rotate_and_center_solute(rot)
            except Exception as e:
                raise Exception(f'Rotate and centre failed for {seed} with exception: {e}')
            if '' in rot.info:
                del rot.info['']
            write(outfile,rot)

    import numpy as np
    from os import path,getcwd,chdir
    from ase.io import read

    def find_range_sep(self,solute_names,in_path,out_path,wrapper,
                       calc_params,solvent=None,charges={},rs_range=[0.1,0.2,0.3],all_readonly=False):
        '''
        Optimises the range separation parameter gamma in a range-separated Hybrid functional with
        Yukawa-switching.
        See 'Using optimally tuned range separated hybrid functionals in ground-state
        calculations: Consequences and caveats',  Andreas Karolewski, Leeor Kronik, and Stephan Kümmel
        J. Chem. Phys. 138, 204115 (2013)
        https://aip.scitation.org/doi/10.1063/1.4807325
        https://pubs.acs.org/doi/abs/10.1021/ct5000617
        and 'Electronic Band Shapes Calculated with Optimally Tuned Range-Separated Hybrid Functionals'
        B. Moore, et al, D. Jacquemin, J. Chem. Theory Comput. 2014, 10, 10, 4599
        https://pubs.acs.org/doi/10.1021/ct500712w

        Minimises J^2 = \Sum_i=0^1 [eps_H[N+i] + IP(N+i)))]^2
        where IP(N) = E(N-1) - E(N)
        so J^2 = \Sum_i=0^1 [eps_H[N+i] + E(N-1+i) - E(N+i)]^2
               = [eps_H[N] + E(N-1) - E(N)]^2 + [eps_H[N+1] + E(N) - E(N+1)]^2 
        '''
        wdir = f'{out_path}'
        if not path.exists(wdir):
            makedirs(wdir)

        for seed in solute_names:
            if seed in charges:
                charge = charges[seed]
            else:
                charge = 0

            # Find input and check if output already exists
            infile = in_path+"/"+seed+".xyz"
            if not path.exists(infile):
                print('Skipping optimal RS tuning for: ',seed,' - no input file')

            # Read the geometry
            solute=read(infile)

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
                for c in [charge,charge+1,charge-1]:
                    rs = f'{rsf:.2f}'
                    label = f"ot_pars_cam{rs}_Q{c}"
                    outfile = f'{label}'
                    print(f'Optimal RS tuning with rs={rs} for: {seed} (Q={c}): ({basis} {func} {self.solvstr(solvent)})')
                    if path.exists(outfile+".out") or path.exists(outfile+".nwo") or all_readonly:
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
                    calc_params['func'] = func+f':{rs}'
                    energy[rs,c], calc[rs,c] = wrapper.singlepoint(solute,label,calc_params,
                                               solvent=solvent,charge=c,spin=s,readonly=readonly)
                    print(rs,c,energy[rs,c]/Ha)
                    evals[rs,c] = calc[rs,c].calc.kpts[0].eps_n
                    occs[rs,c] = calc[rs,c].calc.kpts[0].f_n
                    if c != charge:
                        evals[rs,c,s] = calc[rs,c].calc.kpts[1].eps_n
                        occs[rs,c,s] = calc[rs,c].calc.kpts[1].f_n
            chdir(origdir)
            return energy,calc,evals,occs

    # Calculate Excited states of all molecules
    def calc_all_excited_states(self,solute_names,in_path,out_path,excit_func,
                                calc_params,nroots,solvent=None,charges={},
                                plot_homo=None,plot_lumo=None,plot_trans_den=None):
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
        target: int
            Excited state index, or None for ground state
        solvent: str
            Implicit solvent name, or None for gas-phase
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule
        """
        from ase.io import read
        from os import path, makedirs, getcwd, chdir

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
            if target is not None and target!=0:
                seed = seed+"_es"+str(target)

            # Find input and check if output already exists
            infile = f'{in_path}/{seed}.xyz'
            outfile = f'{out_path}/{seed}/{seed}_{method}'
            if not path.exists(infile):
                print(f'Skipping {method} excitations for: {seed} - no input file')
                continue
            readonly = False
            if path.exists(f'{outfile}.out') or path.exists(f'{outfile}.nwo'):
                print(f'Skipping recalculation of {method} excitations for: ',seed,
                      ',',self.solvstr(solvent),' - output file already present')
                readonly = True
            else:
                print(f'{method} excitations for: {seed}')

            # Read the geometry
            solute=read(infile)

            # Check if output directory exists, create it if not, and change to it
            origdir = getcwd()
            wdir = f'{out_path}/{seed}'
            if not path.exists(wdir):
                makedirs(wdir)
            chdir(wdir)

            # Now run the excitations function of the wrapper
            try:
                excit_func(solute,f'{seed}_{method}',calc_params,nroots,solvent,charge,readonly=readonly,
                           plot_homo=plot_homo,plot_lumo=plot_lumo,plot_trans_den=plot_trans_den)
            except KeyboardInterrupt:
                raise Exception('Keyboard Interrupt')
            except SyntaxError as e:
                raise Exception('Syntax Error:',e)
            except Exception as e:
                print('Excited state calculation failed for: ',seed)
                print('Exception was: ',e)
            chdir(origdir)

    # Calculate vibrational frequencies of all molecules


    def calc_vib_freq(self,solute_names,in_path,out_path,freq_func,
                      calc_params,solvent=None,charges={}):
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
        target: int
            Excited state index, or None for ground state
        solvent: str
            Implicit solvent name, or None for gas-phase
        charges: dict
            Keys are strings corresponding to some or all of the entries in solute names, entries are net charges on each molecule
        """
        from ase.io import read
        from os import path, makedirs, getcwd, chdir

        wdir = f'{out_path}'
        target = calc_params['target']
        if not path.exists(wdir):
            makedirs(wdir)
        for seed in solute_names:
            if seed in charges:
                charge = charges[seed]
            else:
                charge = 0
            if target is not None and target!=0:
                seed = seed+"_es"+str(target)
            infile = f'{in_path}/{seed}.xyz'
            outfile = f'{out_path}/{seed}/{seed}_freq'
            if not path.exists(infile):
                print('Skipping Vibrational Frequencies for: ',seed,
                      ' - no input file')
                continue
            if path.exists(outfile+".out") or path.exists(outfile+".nwo"):
                print('Skipping Vibrational Frequencies for: ',seed,
                      ' - output file already present')
                continue
            solute=read(infile)
            print('Vibrational Frequencies for: ',seed)
            origdir = getcwd()
            wdir = f'{out_path}/{seed}'
            if not path.exists(wdir):
                makedirs(wdir)
            chdir(wdir)
            try:
                freq_func(solute,f"{seed}_freq",calc_params,solvent,charge)
            except KeyboardInterrupt:
                raise Exception('Keyboard Interrupt')
            except SyntaxError:
                raise Exception('Keyboard Interrupt')
            except Exception as e:
                print('Vibrational Frequencies failed for: ',seed)
                print('Exception was: ',e)
            chdir(origdir)

    # Create a parser for the solutes program
    def make_parser(self):

        import argparse

        # Parse command line values
        main_help = ('Prepares DFT-optimised model structures for a list of solutes.\n'+
                     'for each solvent, loops over various tasks:\n'+
                     'download xyz, geom opt, TDDFT\n'+
                     'rotamer search, rotation to xy-plane,\n'+
                     'solvated geom opt and TDDFT.')
        epi_help = ('Names are read in from the file provided in the "namefile"\n' +
                    'argument. All other\n' +
                    'arguments are optional and turn on other tasks. To provide\n' +
                    'pre-optimised inputs, put their structures in a pre-existing\n' +
                    'directory named "xyz". If not present, an attempt will be\n' +
                    'made to download a structure from the NIH chemical name\n' +
                    'resolver (CACTUS).')
        parser = argparse.ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--namefile','-N',default='names',type=str,help='File from which to read a list of names of solvent or solute molecules. Each line should take the form "shortname: IUPACsystematicname"')
        parser.add_argument('--solvent','-S',default=None,type=str,help='Solvent for implicit solvent runs.')
        parser.add_argument('--nroots','-n',default=5,type=int,help='Number of TDDFT excitations to calculate.')
        parser.add_argument('--target','-t',default=None,type=int,help='Targetted TDDFT excitation for geom opt.)')
        parser.add_argument('--rotate','-r',default=True,type=bool,help='If True, attempt to rotate molecule so that the "flattest" part, defined by furthest-apart pairs of C atoms, lies in the xy-plane')
        parser.add_argument('--rotamers','-R',default=False,nargs='?',const=True,type=bool,help='If True, attempt to find the lowest-energy rotamer by switching positions of all rotatable -OH groups.')
        parser.add_argument('--vibrations','-V',default=False,nargs='?',const=True,type=bool,help='If True, Calculate all vibrational frequencies for final level of theory.')
        parser.add_argument('--charges','-Q',default={},nargs='?',type=dict,help='Charges on molecular species. Not for command-line use')
        parser.add_argument('--directory','-D',default=None,nargs='?',type=str,help=argparse.SUPPRESS)
        parser.add_argument('--calc_params','-Z',default={},nargs='?',type=str,help=argparse.SUPPRESS)
        # Wrapper Dependent
        parser.add_argument('--solvent_settings',default=None,type=str,help='Solvent settings for implicit solvent runs.')
        parser.add_argument('--plot_homo','-H',default=None,type=int,help='Number of orbitals from HOMO downwards to plot')
        parser.add_argument('--plot_lumo','-L',default=None,type=int,help='Number of orbitals from LUMO downwards to plot')
        parser.add_argument('--plot_trans_den','-T',default=None,type=int,help='Number of excitations for which to plot the transition density')
        parser.add_argument('--basis','-b',default='6-311++G**',type=str,help='Basis set string for geom opt and TDDFT.')
        parser.add_argument('--func','-f',default='PBE0',type=str,help='XC Functional string for geom opt and TDDFT.')
        parser.add_argument('--disp','-d',default=True,type=bool,help='Grimme D3 Dispersion correction (set to True to activate)')

        return parser

    def validate_args(self,args):
        default_args = make_parser().parse_args("")
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# In[ ]:


def get_parser():
    return SolutesTask().make_parser()

# Main program
if __name__ == '__main__':
    args = make_parser().parse_args()
    if (args.target==0):
        args.target = None
    print(args)
    sols = SolutesTask()

    # get list of molecules and sanitize
    with open(args.namefile) as f:
        namelist = f.readlines()
    namelist = [x.split(":") for x in namelist]
    namelist = [[y.strip() for y in x] for x in namelist]
    
    # duplicate if a shortname has not been assigned
    namelist = [[x[0],x[0]] if len(x)==1 else x for x in namelist]
    
    # make it into a dictionary
    namelist = {x[0]: x[1] for x in namelist}
    
    # download xyz files from cactus if not already present
    namelist = get_xyz_files(namelist,"xyz")
    
    # run main driver
    from esteem.wrappers import nwchem
    sols.wrapper = nwchem.NWChemWrapper()
    sols.run(namelist)

