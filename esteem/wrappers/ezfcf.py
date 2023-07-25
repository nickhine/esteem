#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Defines the ezFCFWrapper Class"""

import numpy as np

class ezFCFWrapper():
    """
    Writes input xml files for ezFCF, runs them, and reads results
    """

    def __init__(self,
                 temp=300,
                 threshold=0.001,
                 max_vib_init=1,
                 max_vib_target=4,
                 initial_state_thresh=0.1,
                 target_state_thresh=0.4,
                 xml_filename="ezfcf.xml",
                 ez_exec="~/ezSpectrum_2021/bin/ezFCF_linux.exe"):

        self.temp = temp
        self.threshold = threshold
        self.max_vib_init = max_vib_init
        self.max_vib_target = max_vib_target
        self.initial_state_thresh = initial_state_thresh
        self.target_state_thresh = target_state_thresh
        self.exec = ez_exec
        self.xml_filename = xml_filename

        # Initialise data to empty list
        self.data = []
    
    def write_job_params(self,xmlF):
        
        job_params_str = f"""<input
  job = "harmonic_pes">

<job_parameters
        temperature                              = "{self.temp}"
        spectrum_intensity_threshold             = "{self.threshold}" >
</job_parameters>

<!--
  ______________________________________________________________________

    Tags which start with "OPT_" will be ignored.
    To include these optional keywords please "uncomment" by removing
    "OPT_" from the start and the corresponding end tag (if present)
  ______________________________________________________________________

 -->
 """
        xmlF.write(job_params_str)
        
    def write_parallel_approximation(self,xmlF,opt=True):
        optstr = "OPT_" if opt else ""
        paral_str = f"""
<{optstr}parallel_approximation
        max_vibr_excitations_in_initial_el_state = "{self.max_vib_init}"
        max_vibr_excitations_in_target_el_state  = "{self.max_vib_target}"
        combination_bands                        = "true"
        use_normal_coordinates_of_target_states  = "true"
 >

  <OPT_do_not_excite_subspace size = "0" normal_modes = " " >
  </OPT_do_not_excite_subspace>

  <OPT_energy_thresholds  units="eV, K, cm-1">
    <initial_state   units="eV">      {self.initial_state_thresh}    </initial_state>
    <target_state   units="eV">      {self.target_state_thresh}    </target_state>
  </OPT_energy_thresholds>

  <OPT_print_franck_condon_matrices flag="true">
  </OPT_print_franck_condon_matrices>

</{optstr}parallel_approximation>

<!--
  ______________________________________________________________________

 -->
"""
        xmlF.write(paral_str)

    def write_dushinsky_rotations(self,xmlF,opt=False):
        optstr = "OPT_" if opt else ""
        dush_str = f"""<{optstr}dushinsky_rotations target_state="1"
      max_vibr_excitations_in_initial_el_state = "{self.max_vib_init}"
      max_vibr_excitations_in_target_el_state  = "{self.max_vib_target}"
      >
  <OPT_max_vibr_to_store  target_el_state="4">
  </OPT_max_vibr_to_store>

  <OPT_do_not_excite_subspace size = "2" normal_modes = "0 1">
  </OPT_do_not_excite_subspace>

  <energy_thresholds  units="eV, K, cm-1">
    <initial_state   units="eV">      {self.initial_state_thresh}    </initial_state>
    <target_state   units="eV">      {self.target_state_thresh}    </target_state>
  </energy_thresholds>

  <OPT_single_excitation
       ini="0"
       targ="1v1">
  </OPT_single_excitation>

</{optstr}dushinsky_rotations>

<!--
  ______________________________________________________________________

 -->\n\n"""
        xmlF.write(dush_str)
    
    def run(self):
        import subprocess
        ezfcf_command = (f"{self.exec} {self.xml_filename} > {self.xml_filename}.out")
        #print(ezfcf_command)
        errorcode = subprocess.call(ezfcf_command, shell=True)
        if errorcode:
            print(f'ezFCF process failed with errorcode: {errorcode}')
             #aise Exception(f'ezFCF process failed with errorcode: {errorcode}')
    
    def results_exist(self):
        from os import path
        return True if path.exists(f"{self.xml_filename}.spectrum_dushinsky") else False
    
    def read(self):
        results_file = f"{self.xml_filename}.spectrum_dushinsky"
        if not self.results_exist():
            print(f"Results file {results_file} not found")
            #raise Exception(f"Results file {results_file} not found")
        try:
            spec = np.loadtxt(results_file,usecols=(0,1))
            if spec.shape==(2,):
                spec = np.array([spec],ndmin=2)
        except:
            spec = None #[[0.0,-1.0]]
        return spec
    
    def add_model_data(self,model,linear=False):
        data = {}
        nat = np.count_nonzero(model.get_tags()==1)
        data['NAtoms'] = nat
        data['ifLinear'] = linear
        data['Geometry'] = "".join([at.symbol + " " + ' '.join(map(str, at.position)) + "\n" 
                                    for at in model if at.tag==1])
        data['geometry_units'] = "angstr"
        data['if_normal_modes_weighted'] = False
        data['atoms_list'] = " ".join([at.symbol for at in model if at.tag==1])
        nmodes = len(model.info['modes'])
        nmstr = ""
        for k in range(0,nmodes,3):
            for iat in range(nat):
                for dk in range(0,min(3,nmodes-k)):
                    kk = k + dk
                    mode = model.info['modes'][kk][iat*3:iat*3+3]
                    nmstr += " ".join(map(str,mode)) + " "
                nmstr += "\n"
            nmstr += "\n"
        data['NormalModes'] = nmstr
        freqs = np.real(model.info['freqs'])[0:]
        data['Frequencies'] = "\n".join(map(str,freqs))

        self.data.append(data)

    def write_xml(self):
        xmlF = open(self.xml_filename, 'w')

        # Write default job parameters
        self.write_job_params(xmlF)
        self.write_parallel_approximation(xmlF)
        self.write_dushinsky_rotations(xmlF)
        xmlF.write('<initial_state>\n')
        xmlF.write(f'  <!-- THIS INITIAL STATE IS FROM data[0] -->\n\n')
        self.write_state_xml_file(xmlF, self.data[0], "initial")
        xmlF.write('</initial_state>\n\n')
        xmlF.write("""<!--
  ______________________________________________________________________

 -->\n\n""")
    
        state_n = 0
        for dat in self.data[1:]:
            state_n += 1
            xmlF.write('<target_state>\n\n')
            xmlF.write(f'  <excitation_energy units="eV"> 0 </excitation_energy>\n\n')
            xmlF.write(f'  <!-- THIS TARGET STATE IS FROM data[{state_n}] -->\n')
            self.write_state_xml_file(xmlF, dat, "initial")
            xmlF.write('</target_state>\n\n')
            xmlF.write("""<!--
  ______________________________________________________________________

 -->\n\n""")
        xmlF.write('</input>\n')

    # Adapted from make_xml.py in the ezFCF distrbution
    def write_state_xml_file(self,xmlF, data: dict, which_state: str):
        """ Write the state to the xml file. """
        # To improve readability, each item of the data dictionary is unpacked to a variable.

        xmlF.write('  <geometry\n')
        no_atoms = data["NAtoms"]
        xmlF.write(f'    number_of_atoms = "{str(no_atoms)}"\n')
        # linear?
        is_linear = data['ifLinear']
        if is_linear:
            xmlF.write('    linear = "true"\n')
        else:
            xmlF.write('    linear = "false"\n')

        geometry_units = data["geometry_units"]
        xmlF.write(f'    units   = "{geometry_units}"\n')

        xmlF.write('    text   = "\n')
        geometry = data['Geometry']
        xmlF.write(geometry)
        xmlF.write('             ">\n')
        xmlF.write('  </geometry>\n\n')

        atoms_order = " ".join([f"{str(nm)}" for nm in range(no_atoms)])

        xmlF.write('  <OPT_manual_atoms_reordering\n')
        xmlF.write(f'     new_order="{atoms_order}">\n')
        xmlF.write('  </OPT_manual_atoms_reordering>\n\n')

        xmlF.write('  <normal_modes\n')
        if_normal_modes_weighted = data["if_normal_modes_weighted"]
        xmlF.write(f'    if_mass_weighted="{if_normal_modes_weighted}"\n')
        xmlF.write('    text = "\n')
        normal_modes = data['NormalModes']
        xmlF.write(normal_modes)
        xmlF.write('           "\n')
        xmlF.write('   atoms = "')
        atoms_list = data['atoms_list']
        xmlF.write(atoms_list)
        xmlF.write('           ">\n')
        xmlF.write('  </normal_modes>\n\n')

        if is_linear:
            normal_modes = " ".join([f"{str(nm)}" for nm in range(3*no_atoms - 5)])
        else:
            normal_modes = " ".join([f"{str(nm)}" for nm in range(3*no_atoms - 6)])

        if which_state == "target":
            xmlF.write('  <OPT_manual_normal_modes_reordering\n')
            xmlF.write(f'     new_order="{normal_modes}">\n')
            xmlF.write('  </OPT_manual_normal_modes_reordering>\n\n')

        xmlF.write('  <frequencies\n')
        xmlF.write('    text = "\n')
        frequencies = data['Frequencies']
        xmlF.write(frequencies)
        xmlF.write('             ">\n')
        xmlF.write('  </frequencies>\n\n')


# In[ ]:




