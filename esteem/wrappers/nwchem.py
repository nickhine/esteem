#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines the NWChemWrapper Class"""

# Set up imports
import ase
from ase.calculators.nwchem import NWChem
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import Hartree
from os import environ, path, makedirs, remove
import numpy as np

class NWChemWrapper():
    """Sets up and runs the NWChem Calculator (via ASE) for DFT and TDDFT calculations"""
    
    def __init__(self):
        """Sets up instance attributes for NWChemWrapper """

        # Set up some defaults
        self.dftblock_singlepoint = {'maxiter': 120,
                                    'tolerances': 'tight',
                                    'grid': 'xfine'}
        self.dftblock_geom_loose = {'maxiter': 120}
        self.dftblock_geom_tight = {'maxiter': 120,
                                    'tolerances': 'tight',
                                    'grid': 'xfine'}
        self.driver_block = {str('default'): None,
                             'maxiter': 80,
                             'xyz': 1}
        self.dftblock_freq = {'maxiter': 120, 'tolerances': 'tight', 'grid': 'xfine'}
        self.freq_block = {'temp': f'1 300'}
        self.dftblock_excitations = {'maxiter': 120,'tolerances': 'tight', 'grid': 'xfine'}
        self.dftblock_qmd = {'maxiter': 120}

    def setup(self,nprocs=None,nwchem_cmd=None,nwchem_top=None):
        """Sets up the internal variables of the NWChemWrapper class, including run command"""
        try:
            nwchem_cmd = environ["ASE_NWCHEM_COMMAND"]
        except KeyError:
            if nwchem_cmd is None:
                nwchem_cmd = "nwchem"
            if nprocs is not None:
                nproc_cmd = f'-np {nprocs}'
            else:
                nproc_cmd = ''
            environ["ASE_NWCHEM_COMMAND"]=f'mpirun {nproc_cmd} {nwchem_cmd} PREFIX.nwi >> PREFIX.nwo 2> PREFIX.err'

    def get_restart_file_exts(self):
        return ['.qmdrst','.db']

    def get_completed_file_exts(self):
        return ['.nwo']

    def cleanup(self,seed):
        """Cleans up temporary files created by a NWChem run that are of no further use"""

        # Cleanup temporary files
        for dir in "./",seed+"/":
            for s in ("p","zmat","b","b^-1","c","dmat","err"):
                if path.isfile(dir+seed+"."+s):
                    remove(dir+seed+"."+s)
            for i in range(0,99):
                for z in range(1,3):
                    gridfile = dir+seed+".gridpts."+repr(i).zfill(z)
                    if path.isfile(gridfile):
                        remove(gridfile)

    def unpack_params(self,calc_params):
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
        return basis,xc,target,disp
                    
    def _cosmo_seed(self,solvent):
        # See here for a list of solvents:
        # http://www.nwchem-sw.org/index.php/Release66:SMD_Model
        if solvent=='meth':
            return 'methanol'
        elif solvent=='etoh':
            return 'ethanol'
        elif solvent=='acet':
            return 'acetntrl'
        elif solvent=='dich':
            return 'dcm'
        elif solvent=='cycl':
            return 'cychexan'
        elif solvent=='watr':
            return 'water'
        elif solvent=='diox':
            return 'dioxane'
        else:
            return solvent

    def check_func(self,func):

        fullfunc = func
        cam = None
        if ":" in func:
            cam = float(func.split(":")[1])
            func = func.split(":")[0]

        # Special settings for long-range corrected functionals (and others not accessible via a single command)
        if func=='BLYP':
            fullfunc  = 'becke88 lyp'
        if func=='CAM-B3LYP':
            if cam is None:
                cam = 0.33
            fullfunc  = f'xcamb88 1.00 lyp 0.81 vwn_5 0.19 HFexch 1.00\n  cam {cam:.2f} cam_alpha 0.19 cam_beta 0.46\n  direct'
        if func=='LC-BLYP':
            if cam is None:
                cam = 0.33
            fullfunc  = f'xcamb88 1.00 lyp 1.00            HFexch 1.00\n  cam {cam:.2f} cam_alpha 0.00 cam_beta 1.00\n  direct'
        if func=='LC-PBE':
            if cam is None:
                cam = 0.30
            fullfunc  = f'xcampbe96 1.0 cpbe96 1.0 HFexch 1.0\n  cam {cam:.2f} cam_alpha 0.00 cam_beta 1.00\n  direct'
        if func=='LC-PBE0':
            if cam is None:
                cam = 0.30
            fullfunc  = f'xcampbe96 1.0 cpbe96 1.0 HFexch 1.0\n  cam {cam:.2f} cam_alpha 0.25 cam_beta 0.75\n  direct'
        if func=='wLC-PBE':
            if cam is None:
                cam = 0.40
            fullfunc  = f'xwpbe 1.0     cpbe96 1.0 HFexch 1.0\n  cam {cam:.2f} cam_alpha 0.00 cam_beta 1.00\n  direct'
        if func=='LRC-wPBEh':
            if cam is None:
                cam = 0.20
            fullfunc  = f'xwpbe 0.8     cpbe96 1.0 HFexch 1.0\n  cam {cam:.2f} cam_alpha 0.20 cam_beta 0.80\n  direct'

        return fullfunc

    def _get_grid_str(self,grid):
        if grid is None:
            grid = [[-5.0,10.0,100],
                    [-5.0,10.0,100],
                    [-5.0,10.0,100]]
        grid_str = f'''
     {grid[0][0]} {grid[0][1]} {grid[0][2]}
     {grid[1][0]} {grid[1][1]} {grid[1][2]}
     {grid[2][0]} {grid[2][1]} {grid[2][2]}
    '''
        return grid_str
    
    # These dplot routines should only be used on a calculator that has already been run
    # successfully on the same system
    def _add_dplot_orb(self,calc,iorb,grid=None):

        dplot_block = {'TITLE': f'{calc.label}_ORBITAL{iorb}',
                       'vectors': f'{calc.label}.movecs',
                       'LimitXYZ': self._get_grid_str(grid),
                       'spin': 'total',
                       'orbitals': f'view; 1; {iorb}',
                       'gaussian': None,
                       'output': f'{calc.label}_orbital{iorb}.cube'}
        calc.set(dplot=dplot_block)
        calc.set(theory='')
        calc.set(restart_kw='restart')
        calc.set(task='dplot')

    def _add_dplot_den(self,calc,grid=None):
        dplot_block = {'TITLE': f'{calc.label}_DENSITY',
                       'vectors': f'{calc.label}.movecs',
                       'LimitXYZ': self._get_grid_str(grid),
                       'spin': 'total',
                       'gaussian': None,
                       'output': f'{calc.label}_density.cube'}
        calc.set(dplot=dplot_block)
        calc.set(theory='')
        calc.set(restart_kw='restart')
        calc.set(task='dplot')
    
    def _add_dplot_trans_den(self,calc,iexc,grid=None):

        dplot_block = {'TITLE': f'TRANS_DEN{iexc}',
                       'civecs': f'{calc.label}.civecs_singlet',
                       'LimitXYZ': self._get_grid_str(grid),
                       'spin': 'total',
                       'root': f'{iexc}',
                       'gaussian': None,
                       'output': f'{calc.label}_trans_den{iexc}.cube'}
        calc.set(dplot=dplot_block)
        calc.set(theory='')
        calc.set(restart_kw='restart')
        calc.set(task='dplot')

    def _add_cosmo(self,calc,solvent):
        cosmo_smd = False
        cosmo_vem = 0
        if isinstance(solvent,str):
            solv_str = self._cosmo_seed(solvent)
        if isinstance(solvent,dict):
            if 'solvent' in solvent:
                solv_str = self._cosmo_seed(solvent['solvent'])
            if 'cosmo_smd' in solvent:
                cosmo_smd = solvent['cosmo_smd']
            if 'cosmo_vem' in solvent:
                cosmo_vem = solvent['cosmo_vem']
        calc.set(cosmo={'solvent': solv_str,
                        'do_cosmo_smd':str(cosmo_smd),
                        'do_cosmo_vem':cosmo_vem})

    def _add_tddft(self,calc,nroots,target=None):

        tddft_block = {'nroots': nroots,
                       'civecs': None, # adds civecs directive
                       'notriplet': None}
        if target is not None and target!=0:
            tddft_block['target'] = target
            tddft_block['grad'] = {'root': target}

        calc.set(tddft=tddft_block)

    def singlepoint(self,model,label,calc_params={},solvent=None,charge=0,spin=0,
                    forces=False,dipole=True,continuation=False,readonly=False,calconly=False,
                    cleanup=True):
        """Runs a singlepoint calculation with the NWChem ASE calculator"""
        basis, xc, target, disp = self.unpack_params(calc_params)
        calc_nw = NWChem(label=label,basis=basis,xc=self.check_func(xc))
        if (target is not None) and (target != 0):
            if forces:
                self._add_tddft(calc_nw,target,target)
            else:
                self._add_tddft(calc_nw,target) # no need for a target, just use nroots=target
        if solvent is not None:
            self._add_cosmo(calc_nw,solvent)
        self.dftblock_singlepoint['mult'] = int(2*spin + 1)
        if disp:
            self.dftblock_singlepoint['disp'] = {'vdw': '3'}
        calc_nw.set(dft=self.dftblock_singlepoint)
        if continuation:
            calc_nw.set(restart_kw='restart')
        calc_nw.set(charge=charge)
        calc_nw.set(print='default')
        calc_nw.set(noprint='"final vectors analysis"')
        calc_nw.set(memory='2000 mb')
        model.calc = calc_nw
        if readonly:
            calc_nw.atoms = model
            calc_nw.read_results() # skip calculation
        if calconly:
            return calc_nw
        if not forces:
            energy = model.get_potential_energy()
            if cleanup:
                self.cleanup(label)
            return energy, calc_nw
        else:
            forces = model.get_forces()
            energy = model.get_potential_energy()
            if cleanup:
                self.cleanup(label)
            return energy, forces, calc_nw

    def geom_opt(self,model_opt,label,calc_params={},driver_tol='default',
                 solvent=None,continuation=False,charge=0,readonly=False,
                 calconly=False,cleanup=True):
        """Runs a Geometry Optimisation calculation with the NWChem ASE calculator"""
        basis, xc, target, disp = self.unpack_params(calc_params)
        calc_nw = NWChem(label=label,basis=basis,xc=self.check_func(xc))
        calc_nw.set(driver=self.driver_block)
        if driver_tol != 'default':
            del self.driver_block['default']
            self.driver_block[driver_tol] = None
        if (target is not None) and (target!=0):
            self._add_tddft(calc_nw,target,target)
        if solvent is not None:
            self._add_cosmo(calc_nw,solvent)
        if driver_tol=='tight':
            if disp:
                self.dftblock_geom_tight['disp'] = {'vdw': '3'}
            calc_nw.set(dft=self.dftblock_geom_tight)
        else:
            if disp:
                self.dftblock_geom_loose['disp'] = {'vdw': '3'}
            calc_nw.set(dft=self.dftblock_geom_loose)
        if continuation:
            calc_nw.set(restart_kw='restart')
        calc_nw.set(charge=charge)
        calc_nw.set(print='default')
        calc_nw.set(task='optimize')
        calc_nw.set(memory='2000 mb')
        model_opt.calc = calc_nw
        if calconly:
            return calc_nw    
        if readonly:
            calc_nw.atoms = model_opt
            calc_nw.read_results() # skip calculation
        forces = model_opt.get_forces()
        model_opt.positions = calc_nw.calc.atoms.positions
        energy = model_opt.get_potential_energy()
        if cleanup:
            self.cleanup(label)
        return energy, forces, model_opt.positions

    def freq(self,model_opt,label,calc_params={},solvent=None,charge=0,
             temp=300,writeonly=False,readonly=False,continuation=False,
             cleanup=True):
        """Runs a Vibrational Frequency calculation with the NWChem ASE calculator"""
        basis, xc, target, disp = self.unpack_params(calc_params)
        calc_nw = NWChem(label=label,basis=basis,xc=self.check_func(xc))
        self.freq_block['temp'] = f'1 {temp}'
        calc_nw.set(freq=self.freq_block)
        if (target is not None) and (target != 0):
            self._add_tddft(calc_nw,target,target)
        if solvent is not None:
            self._add_cosmo(calc_nw,solvent)
        if continuation:
            calc_nw.set(restart_kw='restart')
        if disp:
            self.dftblock_freq['disp'] = {'vdw': '3'}
        calc_nw.set(dft=self.dftblock_freq)

        calc_nw.set(charge=charge)
        calc_nw.set(print='default')
        calc_nw.set(memory='2000 mb')
        calc_nw.set(task='frequencies')
        if disp:
            calc_nw.set(disp={'vdw': '3'})

        model_opt.calc = calc_nw
        if readonly:
            calc_nw.atoms = model_opt
            print("Reading results")
            calc_nw.read_results() # skip calculation
        forces = model_opt.get_potential_energy() # Run frequencies
        self.read_freq(calc_nw)

        if cleanup:
            self.cleanup(label)

    def run_md(self,model,steplabel,calc_params,qmd_steps,qmd_timestep,superstep,temp,
                solvent=None,charge=0,continuation=False,readonly=False,
                constraints=None,dynamics=None,cleanup=True):
        """Runs a Quantum Molecular Dynamics calculation with the NWChem ASE calculator"""
        import random

        basis, xc, target, disp = self.unpack_params(calc_params)
        calc_nw = NWChem(label=steplabel,basis=basis,xc=self.check_func(xc))
        random.seed(steplabel)
        randseed = random.randint(0,100000)
        
        qmd_block = {'nstep_nucl': qmd_steps*(superstep+1),
                     'dt_nucl':    qmd_timestep/0.024188843313701584,
                     'targ_temp':  temp,
                     'com_step':   10,
                     'rand_seed': randseed,
                     'thermostat': 'svr 100.0',
                     'print_xyz':  1}
        
        calc_nw.set(qmd=qmd_block)
        if continuation:
            calc_nw.set(restart_kw='restart')
        if (target is not None) and (target != 0):
            self._add_tddft(calc_nw,target,target)
        if solvent is not None:
            self._add_cosmo(calc_nw,solvent)
        if disp:
            self.dftblock_qmd['disp'] = {'vdw': '3'}
        calc_nw.set(dft=self.dftblock_qmd)
        calc_nw.set(print='low')
        if constraints is not None:
            calc_nw.set(constraints=constraints)
        calc_nw.set(charge=charge)
        calc_nw.set(memory='2000 mb')
        calc_nw.set(task='qmd')
        if disp:
            calc_nw.set(disp={'vdw': '3'})
        model.calc = calc_nw
        if readonly:
            print("Reading results")
            calc_nw.atoms = model
            calc_nw.read_results()
            # these should not trigger re-runs
            energy = model.get_potential_energy()
            forces = model.get_forces()
        else:
            forces = model.get_forces()
            energy = model.calc.results["energy"]
        model.positions = calc_nw.calc.atoms.positions
        model.calc.atoms = calc_nw.calc.atoms
        # after call to QMD, atoms /= calc_nw.atoms due to MD motion
        #next_model = calc_nw.get_atoms()
        next_model = model

        if cleanup:
            self.cleanup(steplabel)

        print(superstep,calc_nw.label,energy,forces[0],model.positions[0])
        return energy,forces,model.positions

    def excitations(self,model,label,calc_params={},nroots=1,solvent=None,charge=0,
                    writeonly=False,readonly=False,continuation=False,cleanup=True,
                    plot_homo=None,plot_lumo=None,plot_trans_den=None):
        """Calculates TDDFT excitations with the NWChem ASE calculator"""
        # Set up calculator
        basis, xc, target, disp = self.unpack_params(calc_params)
        calc_nw = NWChem(label=label,basis=basis,xc=self.check_func(xc))

        self._add_tddft(calc_nw,nroots,target)
        if solvent is not None:
            self._add_cosmo(calc_nw,solvent)
        calc_nw.set(dft=self.dftblock_excitations)
        calc_nw.set(charge=charge)
        calc_nw.set(task='energy')
        calc_nw.set(memory='2000 mb')
        if disp:
            self.dftblock_excitations['disp'] = {'vdw': '3'}
        if continuation:
            calc_nw.set(restart_kw='restart')
        model.calc = calc_nw
        if writeonly:
            calc_nw.write_input(atoms=model)
            return 0
        if readonly:
            calc_nw.read_results()
            print("Reading excitations")        
            s_excit = self.read_excitations(calc_nw)
            #model.set_array('singlet_excitations',s_excit)
            energy = calc_nw.get_property('energy',atoms=model,allow_calculation=False)
        else:
            energy = model.get_potential_energy()
            print("Reading excitations")        
            s_excit = self.read_excitations(calc_nw)
        if plot_trans_den is not None:
            for iexc in range(min(plot_trans_den,nroots)):
                self._add_dplot_trans_den(calc_nw,iexc+1)
                calc_nw.calculate(model)
        if plot_homo is not None:
            if isinstance(plothomo,int):
                irange = range(plot_homo)
            elif isinstance(plothomo,list):
                irange = plot_homo
            else:
                raise Exception(f'Unrecognised type for plot_homo: {type(plot_homo)}')
            for ih in irange:
                self._add_dplot_orb(calc_nw,ih+1)
                calc_nw.calculate(model)
        if plot_lumo is not None:
            if isinstance(plotlumo,int):
                irange = range(plotlumo)
            elif isinstance(plotlumo,list):
                irange = plot_lumo
            else:
                raise Exception(f'Unrecognised type for plot_lumo: {type(plot_lumo)}')
            for il in range(plot_lumo):
                self._add_dplot_orb(calc_nw,il+1)
                calc_nw.calculate(model)
        if cleanup:
            self.cleanup(label)
        return s_excit, energy

    def read_excitations(self,calc):
        """Read Excitations from nwchem calculator."""

        filename = calc.label+'.nwo'
        file = open(filename, 'r')
        lines = file.readlines()
        file.close()

        s_excit = []
        trans_lines = []
        for i, line in enumerate(lines):
            if line.find('NWChem TDDFT Module') >= 0:
                # In case of multiple copies of TDDFT output in file,
                # return only the last one
                s_excit = []
                trans_lines = []
                for j, line2 in enumerate(lines[i:]):
                    if line2.find('No. of roots') >= 0:
                        word = line2.split()
                        nroots = word[4]
                    if line2.find('Alpha electrons') >= 0:
                        word = line2.split()
                        nalpha = int(word[3])
                    if line2.find('Root') >= 0 and line2.find('singlet a') >= 0:
                        word = line2.split()
                        root = int(word[1])
                        energy = float(word[6])
                        word = lines[i+j+5].split()
                        osc = float(word[3])
                        tl = []
                        for k, line3 in enumerate(lines[i+j+10:]):
                            if '----------' in line3 or len(line3.split())==0:
                                break
                            omo = int(line3.split()[1])
                            umo = int(line3.split()[5])
                            weight = float(line3.split()[7])
                            xy = line3.split()[8]
                            tl.append([nalpha-omo,umo-nalpha-1,weight,xy])
                        tl = sorted(tl,key = lambda x: x[2]**2,reverse=True)
                        tl.insert(0, (root,energy,osc))
                        s_excit.append((root,energy,osc))
                        trans_lines.append(tl)
                        if root == nroots:
                            break
        calc.results['excitations'] = np.array(s_excit,ndmin=2)
        calc.results['transition_origins'] = trans_lines
        return s_excit

    def read_freq(self,calc):
        """Read Vibrational Frequencies and Normal Modes from results of nwchem calculator."""
        file = open(calc.label+'.nwo', 'r')
        lines = file.readlines()
        file.close()
        nat = len(calc.atoms)
        nRa = 3*nat
        freq = []
        for i, line in enumerate(lines):
            if line.find('NORMAL MODE EIGENVECTORS IN CARTESIAN COORDINATES') >= 0:
                # In case of multiple copies of frequency output in file,
                # return only the last one
                freq = np.zeros(nRa)
                nmode = np.zeros((nRa,nRa))
                maxRa = 0
                minRa = 0
                Rb = -1
                for j, line2 in enumerate(lines[i:i+nRa*nRa/6+nRa/6*10]):
                    words = line2.split()
                    if len(words)>0:
                        if words==[str(e) for e in range(maxRa+1,maxRa+len(words)+1)]:
                            minRa = maxRa + 1
                            maxRa = minRa + len(words) - 1
                            Rb = 0
                        if words[0] ==str(Rb+1) and len(words)==maxRa-minRa+2:
                            nmode[Rb,minRa-1:maxRa] = [float(w) for w in words[1:]]
                            Rb = Rb + 1
                    if line2.find('Frequency') >= 0:
                        new_freqs = [float(s) for s in words[1:]]
                        freq[minRa-1:maxRa]= new_freqs
            if line.find('Derivative Dipole Moments') >= 0:
                ddip = np.zeros((nRa,3))
                Rb = 0
                for j, line2 in enumerate(lines[i:i+nRa+10]):
                    words = line2.split()
                    if len(words)>0:
                        if words[0]==str(Rb+1):
                            ddip[Rb,1:3] = [float(w) for w in words[4:6]]
                            Rb = Rb + 1
            if line.find('Infra Red Intensities') >= 0:
                intense = np.zeros((nRa))
                Rb = 0
                for j, line2 in enumerate(lines[i:i+nRa+10]):
                    words = line2.split()
                    if len(words)>0:
                        if words[0]==str(Rb+1):
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




