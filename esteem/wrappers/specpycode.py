#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Defines the SpecPyCodeWrapper Class"""

import numpy as np
import sys

from esteem.tasks.spectra import wavelength_eV_conv as hc

class SpecPyCodeWrapper():
    """
    Writes input files for Spectroscopy Python Code, runs them, and reads results
    
    :ivar input_filename: default value: "input_file"
    :ivar rootname: name given to all trajX.dat files - default value: "rootname"
    :ivar rootname: default "rootname"
    :ivar input_filename: default "input_file"
    :ivar third_order_cumulant: default TRUE
    :ivar num_steps: default 1000
    :ivar max_t: default 300.0
    :ivar correlation_length_3rd: default 300
    :ivar decay_length: default 500
    :ivar num_trajs: default 0
    :ivar task: default "ABSORPTION"
    :ivar temperature: default 300.0
    :ivar method: default "CUMULANT"
    :ivar spectral_window: default 1.5
    :ivar md_step: default 0.5
    :ivar chromophore_model: default "MD"
    :ivar exec_path: default "/storage/nanosim/Spectroscopy_python_code/"
    :ivar ncores: default 4
    :ivar solvent_model: default "NONE"
    :ivar solvent_reorg: default 0.0
    """

    def __init__(self,
                 solvent_model="NONE",
                 solvent_reorg="0.0",
                 rootname="rootname",
                 input_filename="input_file",
                 third_order_cumulant="TRUE",
                 num_steps=1000,
                 max_t=300.0,
                 correlation_length_3rd=300,
                 decay_length=500,
                 num_trajs=0,
                 task="ABSORPTION",
                 temperature=300.0,
                 method="CUMULANT",
                 spectral_window=1.5,
                 md_step=0.5,
                 chromophore_model="MD",
                 exec_path="/storage/nanosim/Spectroscopy_python_code/",
                 ncores=4):
        
        self.input_filename = input_filename 
        self.rootname = rootname 
        self.ncores = ncores
        self.num_trajs = 0
        self.solvent_model = solvent_model
        self.solvent_reorg = solvent_reorg
        self.rootname = rootname
        self.third_order_cumulant = third_order_cumulant
        self.num_steps = num_steps
        self.max_t = max_t
        self.correlation_length_3rd = correlation_length_3rd
        self.decay_length = decay_length
        self.task = task
        self.temperature = temperature
        self.num_trajs = num_trajs
        self.method = method
        self.spectral_window = spectral_window
        self.md_step = md_step
        self.chromophore_model = chromophore_model

        sys.path.append(exec_path)

        # Initialise data to empty list
        self.data = []
    
    def write_input_file(self):
        inpf = open(self.input_filename,"w")
        
        self.args = {"SOLVENT_MODEL": str(self.solvent_model).upper(),
                     "SOLVENT_REORG": self.solvent_reorg,
                     "MD_ROOTNAME": self.rootname+"_",
                     "THIRD_ORDER_CUMULANT": str(self.third_order_cumulant).upper(),
                     "NUM_STEPS": self.num_steps,
                     "MAX_T": self.max_t,
                     "CORRELATION_LENGTH_3RD": self.correlation_length_3rd,
                     "DECAY_LENGTH": self.decay_length,
                     "TASK": str(self.task).upper(),
                     "TEMPERATURE": self.temperature,
                     "NUM_TRAJS": self.num_trajs,
                     "METHOD": str(self.method).upper(),
                     "SPECTRAL_WINDOW": self.spectral_window,
                     "MD_STEP": self.md_step,
                     "CHROMOPHORE_MODEL": str(self.chromophore_model).upper()}
        for arg in self.args:
            inpf.write(f"{arg} {self.args[arg]}\n")
        inpf.close()
    
    def run(self):
        import importlib
        store_argv = sys.argv        
        sys.argv = ['generate_spectra.py',self.input_filename,self.ncores]
        print(f'# Executing Spectroscopy Python Code from input file {self.input_filename} with parameters:')
        print('#',self.args)
        reload = True if "generate_spectra" in sys.modules else False
        import generate_spectra
        if reload:
            print("Attempting to reload...")
            importlib.reload(generate_spectra) 
        sys.argv = store_argv

    def trajs_exist(self):
        from os import path
        all_trajs_exist = True
        for i in range(self.num_trajs):
            filename = f'{self.rootname}_traj{str(i+1)}.dat'
            if not path.exists(filename):
                all_trajs_exist = False
                print(f'# No file found: {filename}')
        return all_trajs_exist

    def results_exist(self):
        from os import path
        if self.method=="CUMULANT":
            results_file = f"{self.rootname}_MD_cumulant_spectrum.dat"
        else:
            results_file = f"{self.rootname}_MD_ensemble_spectrum.dat"
        return True if path.exists(results_file) else False
    
    def read(self):
        if self.method=="CUMULANT":
            results_file = f"{self.rootname}_MD_cumulant_spectrum.dat"
        else:
            results_file = f"{self.rootname}_MD_ensemble_spectrum.dat"
        if not self.results_exist():
            print(f"Results file {results_file} not found")
            #raise Exception(f"Results file {results_file} not found")
        try:
            spec = np.loadtxt(results_file,usecols=(0,1))
            spec[:,0] = hc/spec[:,0]
            if spec.shape==(2,):
                spec = np.array([spec],ndmin=2)
        except:
            spec = None
        return spec
    
    def write_excitations(self,excitations):
        filename = f'{self.rootname}_traj{str(self.num_trajs+1)}.dat'
        print(f'# Writing to excitations to {filename}')
        excf = open(filename,"w")
        for e in excitations:
            excf.write(f'{e[0][1]} {e[0][2]}\n')
        self.num_trajs += 1
        excf.close()


# In[ ]:




