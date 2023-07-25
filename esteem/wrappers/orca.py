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
        self.origdir = None
        self.exts_to_link = ['out','err','gbw','cis','inp','engrad','_property.txt','mdrestart']
        self.exts_to_save = ['log','txt','xyz','hess','engrad']
        self.scf_block = None
    
    def setup(self,nprocs=None,orca_cmd=None,mpi_options=None,maxcore=None):
        """Sets up the internal variables of the ORCAWrapper class, including run command"""
        try:
            orca_cmd = environ["ASE_ORCA_COMMAND"]
        except KeyError:
            if orca_cmd is None:
                orca_cmd = "~/orca5/orca"
            if mpi_options is None:
                mpi_options="--oversubscribe --mca opal_warn_on_missing_libcuda 0"
            environ["ASE_ORCA_COMMAND"]=f'{orca_cmd} PREFIX.inp "{mpi_options}" >> PREFIX.out 2> PREFIX.err'
        if maxcore is not None:
            self.maxcore = maxcore
        if nprocs is not None:
            self.nprocs = nprocs

    def get_completed_file_exts(self):
        return ['.out','.engrad']

    def get_restart_file_exts(self):
        return ['.mdrestart','.gbw']

    def move_to_tempdir(self,seed):
        self.origdir = getcwd()
        self.tempdir = f'{self.temproot}/{seed}'
        if not path.exists(self.tempdir):
            #print(f'Creating temporary directory {self.tempdir}')
            makedirs(self.tempdir)
        #print(f'Changing to temporary directory {self.tempdir}')
        try:
            chdir(self.tempdir)
        except:
            print(f'Failed to change to {self.tempdir}')
        if self.origdir != self.tempdir:
            #print(f'Making symlinks for {self.exts_to_link} files back to {self.origdir}')
            for ext in self.exts_to_link:
                ext='.'+ext if not '.' in ext else ext
                if path.islink(f'./{seed}{ext}') or path.isfile(f'./{seed}{ext}'):
                    remove(f'./{seed}{ext}')
                symlink(f'{self.origdir}/{seed}{ext}',f'./{seed}{ext}')
        
    def return_from_tempdir(self,seed):
        import glob
        if self.origdir is None:
            raise Exception(f'Original directory has not been set!')
        #print(f'Copying {self.exts_to_save} files back to {self.origdir}')
        for ext in self.exts_to_save:
            ext='.'+ext if not '.' in ext else ext
            extfiles = glob.glob(f"{seed}*{ext}")
            for f in extfiles:
                if path.isfile(f) and not path.islink(f):
                    copyfile(f,f'{self.origdir}/{f}')
        # Delete temporary copies of files linked back
        if self.origdir != self.tempdir:
            for ext in self.exts_to_link:
                ext='.'+ext if not '.' in ext else ext
                if path.islink(f'{seed}{ext}'):
                    remove(f'{seed}{ext}')
        #print(f'Returning to {self.origdir}')
        chdir(self.origdir)

    def cleanup(self,seed):
        """Cleans up temporary files created by a ORCA run that are of no further use"""

        # Cleanup temporary files - TODO
        for dir in "./",seed+"/":
            for ext in ["densities","err"]:
                ext='.'+ext if not '.' in ext else ext
                if path.isfile(f'{dir}{seed}{ext}'):
                    remove(f'{dir}{seed}{ext}')

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
        # https://www.orcasoftware.de/tutorials_orca/prop/CPCM.html#cpcm-and-cosmo
        if solvent=='meth':
            return 'methanol'
        elif solvent=='etoh':
            return 'ethanol'
        elif solvent=='etgl': # ethylene glycol not recognised by orca - set in cpcm_block below
            return 'ethanol'
        elif solvent=='glyc': # glycerol not recognised by orca - set in cpcm_block below
            return 'ethanol'
        elif solvent=='acet':
            return 'acetonitrile'
        elif solvent=='dich':
            return 'CH2Cl2'
        elif solvent=='cycl':
            return 'cyclohexane'
        elif solvent=='watr':
            return 'water'
        else:
            return solvent

# data from
# https://www.engineeringtoolbox.com/liquid-dielectric-constants-d_1263.html#google_vignette
# https://www.engineeringtoolbox.com/refractive-index-d_1264.html
    def _add_cpcm_block(self,calc,solvent):
        if solvent=='etgl':
            self.cpcm_block = (f"\n%cpcm\n" + 
                               f"epsilon 37.0\n" +
                               f"refrac 1.43\nend\n")
        elif solvent=='glyc':
            self.cpcm_block = (f"\n%cpcm\n" + 
                               f"epsilon 46.5\n" +
                               f"refrac 1.47\nend\n")
        else:
            self.cpcm_block = None
            # for reference, ethanol: 25.3, 1.36
        if self.cpcm_block is not None:
            calc.parameters['orcablocks'] += self.cpcm_block

    def check_func(self,func):

        fullfunc = func
        return fullfunc

    def set_careful_scf(self,sthresh=4e-8):
        self.scf_block = f"\n%scf\n"
        self.scf_block += f"maxiter 50\n"
        self.scf_block += f"sthresh {sthresh}\n"
        #self.scf_block += f"ConvForced False\n"
        #self.scf_block += f"SOSCFstart 0.00005 SOSCFMaxIt 12\n"
        #self.scf_block += f"AutoTRAH False\n"
        self.scf_block += f"end\n"
    
    def _add_solvent(self,calc,solvent):

        calc.parameters['orcasimpleinput'] += f" CPCMC({self._cosmo_seed(solvent)})"
        self._add_cpcm_block(calc,solvent)

    def _add_tddft(self,calc,nroots,target=None):

        calc.parameters['orcablocks'] += f"\n%tddft\n  nroots {nroots}\n  tda false"
        if target is not None:
            calc.parameters['orcablocks'] += f"\n  Iroot {target}\n  end"
        else:
            calc.parameters['orcablocks'] += f"\n  end"

    def singlepoint(self,model,label,calc_params={},solvent=None,charge=0,spin=0,
                    forces=False,dipole=True,continuation=False,readonly=False,calconly=False,
                    cleanup=True):
        """Runs a singlepoint calculation with the ORCA ASE calculator"""
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        extra = "" # "defgrid2"
        calc_orca = ORCA(label=label,orcasimpleinput=f"{xc} {dispstr} {basis} {extra}",
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n')
        if (target is not None) and (target != 0):
            self._add_tddft(calc_orca,target,target)
        if solvent is not None:
            self._add_solvent(calc_orca,solvent)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        # Set up spin
        calc_orca.set(mult=int(2*spin+1))
        # Set up charge
        calc_orca.set(charge=charge)
        model.calc = calc_orca
        if readonly:
            calc_orca.atoms = model
            calc_orca.read_results() # skip calculation
        if calconly:
            return calc_orca
        if not readonly:
            self.move_to_tempdir(label)
        if forces:
            f_calc = model.get_forces()
            e_calc = model.calc.results["energy"]
        else:
            e_calc = model.get_potential_energy()
        if dipole:
            d_calc = model.get_dipole_moment()
        if cleanup and not readonly:
            self.cleanup(label)
        if not readonly:
            self.return_from_tempdir(label)

        if forces:
            if dipole:
                return e_calc, f_calc, d_calc, calc_orca
            else:
                return e_calc, f_calc, calc_orca
        else:
            if dipole:
                return e_calc, d_calc, calc_orca
            else:
                return e_calc, calc_orca

    def geom_opt(self,model_opt,label,calc_params={},driver_tol='default',
                 solvent=None,continuation=False,charge=0,spin=0,readonly=False,
                 calconly=False,cleanup=True):
        """Runs a Geometry Optimisation calculation with the ORCA ASE calculator"""
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        calc_orca = ORCA(label=label,orcasimpleinput=f"{xc} {dispstr} {basis}",
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n')
        if (target is not None) and (target!=0):
            self._add_tddft(calc_orca,target,target)
        if solvent is not None:
            self._add_solvent(calc_orca,solvent)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        if hasattr(self,"sym_thresh"):
            sym_block = "%sym SymThresh {sym_thresh} end\n"
            calc_orca.parameters['orcablocks'] += sym_block
        
        if len(model_opt.constraints)>0:
            constraint_str = "%geom Constraints\n"
            for constr in model.constraints:
                if isinstance(constr,FixAtoms):
                    constraint_str = ( constraint_str + 
                                      '{ C' + ' '.join([str(i) for i in constr.index]) + 'C }\n')
            constraint_str = constraint_str + "end\nend\n"
        # Set up spin
        calc_orca.set(mult=int(2*spin+1))
        # Set up charge
        calc_orca.set(charge=charge)
        # Set task
        calc_orca.set(task='opt')
        if driver_tol.lower().find('tight') > -1:
            calc_orca.set(task=driver_tol+'opt')
        model_opt.calc = calc_orca
        if calconly:
            return calc_orca    
        if readonly:
            calc_orca.atoms = model_opt
            calc_orca.read_results() # skip calculation
        if not readonly:
            self.move_to_tempdir(label)
        forces = model_opt.get_forces()
        energy = calc_orca.results['energy'] # avoid second calculation 
        #energy = model_opt.get_potential_energy()
        model_opt.positions = calc_orca.atoms.positions
        if cleanup and not readonly:
            self.cleanup(label)
        if not readonly:
            self.return_from_tempdir(label)
        return energy, forces, model_opt.positions

    def freq(self,model_opt,label,calc_params={},solvent=None,charge=0,
             temp=300,writeonly=False,readonly=False,continuation=False,
             cleanup=True,summary=False):
        """Runs a Vibrational Frequency calculation with the ORCA ASE calculator"""
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        calc_orca = ORCA(label=label,orcasimpleinput=f"{xc} {dispstr} {basis}",
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n')
        if (target is not None) and (target != 0):
            self._add_tddft(calc_orca,target,target)
        if solvent is not None:
            self._add_solvent(calc_orca,solvent)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        calc_orca.set(charge=charge)
        calc_orca.set(task='freq')

        model_opt.calc = calc_orca
        if readonly:
            calc_orca.atoms = model_opt
            #print("Reading results")
            try:
                calc_orca.read_results() # skip calculation
            except Exception as e:
                print(f'Failed reading results: exception was {e}')
        else:
            self.move_to_tempdir(label)
        try:
            forces = model_opt.get_potential_energy() # Run frequencies
        except Exception as e:
            print(f'Failed reading results: exception was {e}')
        #self.read_freq(calc_orca)
        if not readonly:
            self.return_from_tempdir(label)

        if cleanup:
            self.cleanup(label)

    def run_md(self,model,steplabel,calc_params,qmd_steps,qmd_timestep,superstep,temp,
                solvent=None,charge=0,continuation=False,readonly=False,
                constraints=None,dynamics=None,cleanup=True):
        """Runs a Quantum Molecular Dynamics calculation with the ORCA ASE calculator"""
        import random

        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        calc_orca = ORCA(label=steplabel,orcasimpleinput=f"{xc} {dispstr} {basis}",
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n')
        timecon = 10.0
        random.seed(steplabel)
        randseed = random.randint(0,100000)
        md_block = f'''
%md
timestep {qmd_timestep/fs}_fs
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
            self._add_tddft(calc_orca,target,target)
        if solvent is not None:
            self._add_solvent(calc_orca,solvent)
        if constraints is not None:
            calc_orca.set(constraints=constraints)
        if self.scf_block is not None:
            calc_orca.parameters['orcablocks'] += self.scf_block
        calc_orca.set(charge=charge)
        calc_orca.set(task='md')
        model.calc = calc_orca
        if readonly:
            print("Reading results")
            calc_orca.atoms = model
            calc_orca.read_results()
            # these should not trigger re-runs
            energy = model.get_potential_energy()
            forces = model.get_forces()
        else:
            self.move_to_tempdir(steplabel)
            forces = model.get_forces()
            energy = model.calc.results["energy"]
            self.return_from_tempdir(steplabel)
        model.positions = calc_orca.atoms.positions
        next_model = model

        if cleanup and not readonly:
            self.cleanup(steplabel)

        print(superstep,calc_orca.label,energy,forces[0],model.positions[0])
        return None

    def excitations(self,model,label,calc_params={},nroots=1,solvent=None,charge=0,
                    writeonly=False,readonly=False,continuation=False,cleanup=True,
                    plot_homo=None,plot_lumo=None,plot_trans_den=None):
        """Calculates TDDFT excitations with the ORCA ASE calculator"""
        # Set up calculator
        basis, xc, target, disp = self.unpack_params(calc_params)
        dispstr = 'D3BJ' if disp else ''
        calc_orca = ORCA(label=label,orcasimpleinput=f"{xc} {dispstr} {basis}",
                         orcablocks=f'%pal nprocs {self.nprocs} end\n%maxcore {self.maxcore}\n')

        self._add_tddft(calc_orca,nroots,target=None)
        if solvent is not None:
            self._add_solvent(calc_orca,solvent)
        calc_orca.set(charge=charge)
        calc_orca.set(task='energy')
        if self.scf_block is not None:
            calc.parameters['orcablocks'] += self.scf_block
        model.calc = calc_orca
        if writeonly:
            calc_orca.write_input(atoms=model)
            return 0
        if readonly:
            calc_orca.read_results()
            print("Reading excitations")        
            s_excit = self.read_excitations(calc_orca)
            energy = calc_orca.get_property('energy',atoms=model,allow_calculation=False)
        else:
            self.move_to_tempdir(label)
            energy = model.get_potential_energy()
            print("Reading excitations")        
            s_excit = self.read_excitations(calc_orca)
            self.return_from_tempdir(label)
        if cleanup:
            self.cleanup(label)
        return s_excit, energy

    def read_excitations(self,calc):
        """Read Excitations from ORCA calculator."""

        filename = calc.label+'.out'
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
            if getexcit and len(line.split())>0:
                if line.split()[0].isdigit():
                    #print(line)
                    root = int(line.split()[0])
                    energy = float(line.split()[1])*invcm
                    osc = float(line.split()[3])
                    s_excit.append((root,energy,osc))
        calc.results['excitations'] = np.array(s_excit)
        calc.results['transition_origins'] = trans_lines
        return s_excit

    def read_freq(self,calc):
        """Read Vibrational Frequencies and Normal Modes from results of ORCA calculator."""
        file = open(calc.label+'.out', 'r')
        lines = file.readlines()
        file.close()
        nat = len(calc.atoms)
        nRa = 3*nat
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
                f = float(line.split()[1])*invcm
                nmode = np.zeros((nRa,nRa))
                maxRa = 0
                minRa = 0
                Rb = -1
                for j, line2 in enumerate(lines[i:i+int(nRa*nRa/6+nRa/6*10)]):
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




