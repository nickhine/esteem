#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Defines the OnetepWrapper Class"""

from ase.calculators.onetep import Onetep
from os.path import isfile, dirname, abspath, join
from os import environ
import numpy as np

class OnetepWrapper():
    """Sets up and runs the ONETEP Calculator (via ASE) for DFT and TDDFT calculations"""
    
    def __init__(self):
        """Sets up instance attributes for OnetepWrapper """
        
        self.dielectric = {}
        self.dielectric.update(dict.fromkeys(['wat', 'wate', 'water', 'watr'], ["Water",80.4,1.33]))
        self.dielectric.update(dict.fromkeys(['meth', 'methanol', 'meoh'], ["Methanol",32.63,1.329]))
        self.dielectric["wate"] = ["Water",80.4,1.33]
        self.dielectric["acet"] = ["Acetonitrile",36.6,1.344]
        self.dielectric["actn"] = ["Acetone",20.7,1.359]
        self.dielectric["ammo"] = ["Ammonia",22.4,1.33]
        self.dielectric["etoh"] = ["Ethanol",24.3,1.361]
        self.dielectric["dich"] = ["CH2Cl2",9.08,1.424]
        self.dielectric["tech"] = ["CCl4",2.24,1.466]
        self.dielectric["dmfu"] = ["DMF",38.3,1.43]
        self.dielectric["dmso"] = ["DMSO",47.2,1.479]
        self.dielectric["pyri"] = ["Pyridine",12.5,1.51]
        self.dielectric["thff"] = ["THF",7.25,1.407]
        self.dielectric["chlf"] = ["Chloroform",4.9,1.45]
        self.dielectric["hexa"] = ["Hexane",1.89,1.375]
        self.dielectric["tolu"] = ["Toluene",2.4,1.497]
        self.dielectric["cycl"] = ["Cyclohexane",2.02,1.4262]

        self.singlepoint_params = {
            'task': 'SINGLEPOINT',
            'ngwf_threshold_orig': 1e-6,
            'fftbox_batch_size': 12,
            'write_forces': True
        }
        self.geom_opt_params = {
            'task': 'GEOMETRYOPTIMIZATION',
            'ngwf_threshold_orig': 1e-6,
            'geom_continuation': False,
            'geom_reuse_dkngwfs': True,
            'fftbox_batch_size': 12,
            'write_forces': True
        }

        self.excitations_params = {
            'ngwf_threshold_orig': 1.4e-6, 
            'lnv_threshold_orig': 2e-7, 
            'maxit_palser_mano': 600, 
            'do_properties': False, 
            'fftbox_batch_size': 14, 
            'write_forces': False, 
            'write_denskern': True, 
            'write_tightbox_ngwfs': True, 
            'read_denskern': False, 
            'read_tightbox_ngwfs': False, 
            'output_detail': "NORMAL", 
            'cond_energy_gap': '0.1 eV', 
            'cond_num_states': 20, 
            'cond_num_extra_its': 4, 
            'cube_format': False, 
            'lr_tddft_analysis': True, 
            'lr_tddft_RPA': True, 
            'lr_tddft_init_random': False, 
            'lr_tddft_write_kernels': True, 
            'lr_tddft_write_densities': False, 
            'lr_tddft_restart': False, 
            'lr_tddft_num_conv_states': 0,
            'lr_tddft_homo_num': 60, 
            'lr_tddft_lumo_num': 60, 
            'lr_tddft_maxit_cg': 120 }

        self.solvent_params = {
            'is_implicit_solvent': True,
            'is_auto_solvation' : True,
            'is_smeared_ion_rep': True }
        
        self.paw = False
        self.pseudo_path = ""
        self.pseudo_suffix = ""

    def setup(self,nprocs=None,mthreads=None,onetep_cmd=None,mpirun=None,
              set_pseudo_path=None,set_pseudo_suffix=None):
        """Sets up the internal variables of the OnetepWrapper Class"""
        
        # Set up pseudopotential/PAW information
        if set_pseudo_path is None:
            self.pseudo_path = '~/JTH_PBE/'
        else:
            self.pseudo_path = set_pseudo_path
        if set_pseudo_suffix is None:
            self.pseudo_suffix = '.PBE-paw.abinit'
        else:
            self.pseudo_suffix = set_pseudo_suffix
        if self.pseudo_suffix == '.PBE-paw.abinit':
            self.paw = True
        else:
            self.paw = False

        # Set the environment variable for the ONETEP Calculator
        if onetep_cmd is None:
            onetep_cmd = "onetep"
        if mpirun is None:
            mpirun = "mpirun"
        mthread_cmd = ''
        nproc_cmd = ''
        if mpirun == "mpirun":
            if nprocs is not None:
                nproc_cmd = f'-n {nprocs}'
            if mthreads is not None:
                mthread_cmd = f'export OMP_NUM_THREADS={mthreads}; '
        if mpirun == "aprun":
            mthread_cmd = '' # already set in script
            try:
                nproc_cmd = environ["APRUN_ARGS"]
            except KeyError:
                nproc_cmd = ""
        environ["ASE_ONETEP_COMMAND"] = f'{mthread_cmd}{mpirun} {nproc_cmd} {onetep_cmd} PREFIX.dat >> PREFIX.out 2> PREFIX.err'
   
    # Adjust this function to define complex behaviour of NGWF counts
    def _set_cond_ngwfs(self,model,calc_on):
        tags = ["" if i==0 else str(i) for i in model.get_tags()]
        all_species = set(zip(model.get_chemical_symbols(), tags))
        species_ngwf_number_cond = {}
        for s in all_species:
            cond_ngwf_count = -1
            if s[1]=="1":
                if s[0]=='H':
                    cond_ngwf_count = -1
                if s[0]=='C' or s[0]=='O' or s[0]=='N':
                    cond_ngwf_count = -1
            if s[1]=="":
                cond_ngwf_count = -1
            if cond_ngwf_count >= 0:
                species_ngwf_number_cond[s[0]+s[1]] = cond_ngwf_count
        calc_on.set(species_ngwf_number_cond=species_ngwf_number_cond)

    def unpack_params(self,calc_params):
        if 'ngwf_rad' in calc_params:
            ngwf_rad = calc_params['ngwf_rad']
        else:
            raise Exception("Basis not specified in calc_params")
        if 'cutoff' in calc_params:
            cutoff = calc_params['cutoff']
        else:
            raise Exception("Cutoff Energy not specified in calc_params")
        if 'func' in calc_params:
            xc = calc_params['func']
        else:
            raise Exception("XC Functional not specified in calc_params")
        if 'target' in calc_params:
            target = calc_params['target']
        else:
            raise Exception("Target excitation not specified in calc_params")
        if 'energy_range' in calc_params:
            energy_range = calc_params['energy_range']
        else:
            raise Exception("Target excitation not specified in energy_range")
        return ngwf_rad,cutoff,xc,energy_range,target

    def singlepoint(self,model,label,calc_params,solvent=None,charge=0,
                    forces=False,restart=False,readonly=False,writeonly=False,calconly=False):
        """Runs a singlepoint calculation with the ONETEP ASE calculator"""
        calc_on = Onetep(label=label)
        ngwf_rad,cutoff,func,energy_range,target = self.unpack_params(calc_params)
        if (model.cell==0.0).all():
            model.center(10)
            print('Added cell:\n',model.cell)
        if self.pseudo_path is not None:
            calc_on.set(pseudo_path=self.pseudo_path)
        if self.pseudo_suffix is not None:
            calc_on.set(pseudo_suffix=self.pseudo_suffix)
        calc_on.set(paw=self.paw)
        calc_on.set(xc=func)
        calc_on.set(charge=charge)
        calc_on.set(cutoff_energy=cutoff)
        calc_on.set(ngwf_radius=ngwf_rad)
        calc_on.set(**self.singlepoint_params)
        if not forces:
            calc_on.set(write_forces=False)
            model_forces = None
        model.calc = calc_on
        if writeonly:
            calc_on.write_input(atoms=model)
            return 0
        if calconly:
            return calc_on
        if readonly:
            calc_on.read_results()
            energy = calc_on.get_property('energy',atoms=model,allow_calculation=False)
            if forces:
                model_forces = calc_on.get_property('forces',atoms=model,allow_calculation=False)
        else:
            if forces:
                model_forces = model.get_forces()
            energy = model.get_potential_energy()

        return energy, model_forces
    
    def geom_opt(self,model_opt,label,calc_params,driver_tol='default',
                 solvent=None,charge=0,readonly=False,writeonly=False,calconly=False):
        """Runs a Geometry Optimisation calculation with the ONETEP ASE calculator"""
        calc_on = Onetep(label=label)
        ngwf_rad,cutoff,func,energy_range,target = self.unpack_params(calc_params)
        if (model_opt.cell==0.0).all():
            model_opt.center(10)
            print('Added cell:\n',model_opt.cell)
        if self.pseudo_path is not None:
            calc_on.set(pseudo_path=self.pseudo_path)
        if self.pseudo_suffix is not None:
            calc_on.set(pseudo_suffix=self.pseudo_suffix)
        calc_on.set(paw=self.paw)
        calc_on.set(xc=func)
        calc_on.set(charge=charge)
        calc_on.set(cutoff_energy=cutoff)
        calc_on.set(ngwf_radius=ngwf_rad)
        calc_on.set(**self.geom_opt_params)
        # TODO: Handle driver_tol
        model_opt.calc = calc_on
        if writeonly:
            calc_on.write_input(atoms=model_opt)
            return 0
        if calconly:
            return calc_on
        if readonly:
            calc_on.read_results()
            energy = calc_on.get_property('energy',atoms=model_opt,allow_calculation=False)
            model_forces = calc_on.get_property('forces',atoms=model_opt,allow_calculation=False)
        else:
            model_forces = model_opt.get_forces()
            energy = model_opt.get_potential_energy()

        return energy, model_forces, model_opt.positions

    def excitations(self,model,label,calc_params={},nroots=1,solvent=None,charge=0,writeonly=False,
                    readonly=False,continuation=False,cleanup=False,
                    plot_homo=None,plot_lumo=None,plot_trans_den=None):
        """Calculates TDDFT excitations with the ONETEP ASE calculator"""
        calc_on = Onetep(label=label)
        ngwf_rad,cutoff,func,energy_range,target = self.unpack_params(calc_params)
        if (model.cell==0.0).all():
            model.center(10)
            print('Added cell:\n',model.cell)
        if self.pseudo_path is not None:
            calc_on.set(pseudo_path=self.pseudo_path)
        if self.pseudo_suffix is not None:
            calc_on.set(pseudo_suffix=self.pseudo_suffix)
        if not continuation:
            calc_on.set(task='SINGLEPOINT COND LR_TDDFT')
        else:
            calc_on.set(task='LR_TDDFT')
        calc_on.set(paw=self.paw)
        calc_on.set(xc=func)
        calc_on.set(charge=charge)
        calc_on.set(cutoff_energy=cutoff)
        calc_on.set(ngwf_radius=ngwf_rad)
        calc_on.set(ngwf_radius_cond=ngwf_rad)
        calc_on.set(lr_tddft_num_states=nroots)
        calc_on.set(cond_energy_range=energy_range)
        calc_on.set(**self.excitations_params)
        # Setup tags on kernel species, if supplied
        if any(model.get_tags()>0):
            tags = ["" if i==0 else str(i) for i in model.get_tags()]
            all_species = set(zip(model.get_chemical_symbols(), tags))
            tddft_kernel_species = [s[0]+s[1] for s in all_species if s[1]!=""]
            tddft_kernel_str = [('%s' % sp) for sp in tddft_kernel_species]
            calc_on.set(species_tddft_kernel=[tddft_kernel_str])
            # TODO: setup LDOS groups
            self._set_cond_ngwfs(model,calc_on)
        if solvent is not None:
            calc_on.set(**self.solvent_params)
            if solvent not in self.dielectric.keys():
                raise Exception("Solvent not found in database: {}".format(solvent))
            calc_on.set(is_bulk_permittivity=self.dielectric[solvent][1])
            calc_on.set(lr_optical_permittivity=self.dielectric[solvent][2]**2)
        model.calc = calc_on
        if writeonly:
            calc_on.write_input(atoms=model)
            return 0
        if readonly:
            calc_on.read_results()
            gs_energy = calc_on.get_property('energy',atoms=model,
                                             allow_calculation=False)
        else:
            gs_energy = model.get_potential_energy()

        excitations = self.read_excitations(calc_on)
        energies = np.array([gs_energy]*(len(excitations)+1))
        energies[1:] = energies[1:] + excitations[:,1]

        if (cleanup):
            self.cleanup(label)

        return excitations,energies

    def read_excitations(self,calc,out=None):
        
        import numpy as np
        from ase.units import Hartree

        if out is None:
            onetep_file = calc.label + '.out'
            try:
                out = open(onetep_file, 'r')
            except IOError:
                raise Exception('Could not open output file "%s"' % onetep_file)
        line = out.readline()
        excitations = []
        while line:
            line = out.readline()
            if '|Excitation|    Energy (in Ha)   |     Oscillator Str' in line:
                nE = 0
                excitations = []
                line = out.readline()
                while line:
                    words = line.split()
                    if len(words)==0 or 'Transition:' in words[0]:
                        break
                    nE = nE + 1
                    excitations.append([nE,float(words[1])*Hartree,float(words[2])])
                    line = out.readline()
        excitations = np.array(excitations)
        calc.results['excitations'] = excitations
        return excitations

    from os import path
    def cleanup(self,seed):
        # Cleanup temporary files from a ONETEP job
        for b in ("tightbox_ngwfs","dkn"):
            for p in ("","vacuum_"):
                for q in ("","_cond"):
                    s = p+b+q
                    print(seed+"."+s)
                    if path.isfile(seed+"."+s):
                        remove(seed+"."+s)
        for s in ["_dl_mg_log.txt"]:
            print(seed+s)
            if path.isfile(seed+"."+s):
                remove(seed+"."+s) 

