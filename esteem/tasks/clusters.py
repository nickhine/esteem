#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Performs processing of Molecular Dynamics Trajectories to enable use of the results in further tasks
   such as extraction of cluster models centered on the solute molecule, including all solvent molecules
   within a given range, for the purpose of ML training or explicit solvent spectroscopy"""


# In[ ]:


class ClustersTask:

    def __init__(self,**kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'clusters'
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self,arg,getattr(args,arg))

    def run(self,dryrun=False):
        """
        Main routine for processing clusters and running them for excitations.

        The steps involved are:

            1. Load the trajectory file of MD snapshots, which for a given solvent solute pair 
            it expects to find at '{solute}_{solvent}_{md_suffix}.traj' in the current directory

            2. 'Carve' spheres out of the trajectory, that is to say:

                a. Delete counterions (checking they are not within the sphere first)

                b. Delete any whole solvent molecules for which no atoms of the solvent molecule
                are within ``self.radius`` of any atom in the solute molecule.

                c. Label solvent molecules with a tag if they are within ``self.kernel`` by the
                criteria above.

                d. Rotate and center the cluster in the simulation cell, using the two most-distant
                pairs of carbon atoms in the solute to find a common 'plane' for the solute snapshots.

            3. Calculate energies or electronic excitations for each cluster, using the supplied wrapper

        *Arguments*

        self: Argument list for the task, with attributes including:

            ``solute``, ``solvent``, ``radius``, ``output``, ``task_id``, ``counterions``, ``charges``
            ``kernel``, ``basis``, ``func``, ``boxsize``, ``impsolv``,``nroots``, ``cleanup``, ``continuation``

        wrapper: class
            Wrapper to use in the Clusters task

        *Output*:

            If ``self.task_id`` is None, excited state calculations for the whole trajectory.
            If has a int value, then an excited state calculation for just that frame in the trajectory. 
        """
        
        from ase.io import Trajectory
        from esteem.drivers import get_solu_solv_names
        from esteem.trajectories import targstr,merge_traj
        import os

        if (self.output=='dat' or self.output=='onetep') and self.basis=='6-31g*':
            # set default basis for ONETEP calculations:
            # 10 a0 NGWFs, 800 eV cutoff, 6 eV cond NGWF energy range
            calc_params = {'ngwf_rad': 10.0,
                           'cutoff': '800 eV',
                           'energy_range': '6 eV',
                           'target': 0,
                           'func': self.func}
            if self.calc_params == {}:
                self.calc_params = calc_params
        if ('nwo' in self.output or 'nwchem' in self.output or 'orca' in self.output):
            # set default basis for NWChem/ORCA calculations:
            calc_params = {'basis': self.basis,
                           'func': self.func,
                           'disp': self.disp,
                           'target': None}
        if any(out in self.output.lower() for out in ['physnet','mace']):
            if self.calc_params == {}:
                calc_params = {'calc_seed': self.calc_seed,
                               'calc_suffix': self.calc_suffix,
                               'calc_prefix': f'../../{self.calc_prefix}', # MD will be run from subdirectory
                               'target': self.target}
            else:
                calc_params = self.calc_params

        # implicit solvent name defaults to solvent name but should usually be specified separately
        if self.impsolv is None:
            self.impsolv = self.solvent
        # Determine charge on molecule (TODO: spin)
        charge = 0
        spin = 0
        if isinstance(self.charges,dict):
            if self.solute in self.charges:
                charge  = self.charges[self.solute]
        if isinstance(self.charges,int) or isinstance(self.charges,str):
            charge  = int(self.charges)
        
        # Setup snapshot range
        if self.max_snapshots is None:
            self.max_snapshots = 1e6
        if self.min_snapshots is None:
            self.min_snapshots = 0

        if self.task_id is not None and self.radius is not None:
            input_suffix = f'{self.carved_suffix}_{self.task_id:04}'
            delete_traj_carved_file = True
        else:
            input_suffix = f'{self.carved_suffix}'
            delete_traj_carved_file = False

        # Check if the carved trajectory already exists
        which_targstr = targstr(self.which_target)
        solvstr = f'_{self.solvent}' if self.solvent is not None else ''
        traj_carved_file = f'{self.solute}{solvstr}_{which_targstr}_{self.which_traj}_{input_suffix}.traj'
        # In the case where we have carved a trajectory already, skip the rest
        if os.path.exists(traj_carved_file) and self.radius is not None:
            if os.path.getsize(traj_carved_file)>0:
                input_traj = self.get_input_traj(self.solute,self.solvent,self.md_suffix)
                traj_max = min(len(input_traj),self.max_snapshots)
                traj_min = max(0,self.min_snapshots)
                input_traj.close()
                print(f'# Loading existing carved trajectory: {traj_carved_file}')
                traj_carved = list(enumerate(Trajectory(traj_carved_file)))
                #if len(traj_carved) != traj_max - traj_min:
                #    raise Exception(f"Error: pre-existing carved trajectory {traj_carved_file} is the wrong length:\n" +
                #                    f"Expected {traj_max-traj_min}, got {len(traj_carved)}\n" +
                #                    "Delete it or adjust range to match.")
                delete_traj_carved_file = False # do not delete the file if it already existed
            else:
                print(f'# Existing carved trajectory: {traj_carved_file} is empty, writing new one')
                os.remove(traj_carved_file)
        # Case where some form of carving, merging or linking is required:
        traj_carved_present = os.path.exists(traj_carved_file) and os.path.getsize(traj_carved_file)>0
        if (not traj_carved_present) and self.radius is not None:
            traj_carved, traj_carved_file = self.carve_spheres(
                          self.counterions,self.radius,self.kernel,self.max_solvent_mols,
                          self.boxsize,self.task_id)
            print(f'# {len(list(traj_carved))} clusters carved from full snapshots with radius {self.radius}.')
        traj_carved_present = os.path.exists(traj_carved_file) and os.path.getsize(traj_carved_file)>0
        if (not traj_carved_present) and self.radius is None and isinstance(self.md_suffix,list):
            # merging required
            input_traj_names = [f'{self.solute}{solvstr}_{mds}.traj' for mds in self.md_suffix]
            print(f'# Merging {input_traj_names} and writing to {traj_carved_file}')
            merge_traj(input_traj_names,traj_carved_file)
        traj_carved_present = os.path.exists(traj_carved_file) and os.path.getsize(traj_carved_file)>0
        if (not traj_carved_present) and self.radius is None:
            # No carving or merging required, just make link to input
            input_traj_name = f'{self.solute}{solvstr}_{self.md_suffix}.traj'
            print(f'# Looking for {traj_carved_file}')
            if not (os.path.isfile(f'{traj_carved_file}') or os.path.islink(f'{traj_carved_file}')):
                print(f'# Creating link from {input_traj_name} to {traj_carved_file}')
                os.symlink(input_traj_name,traj_carved_file)
        # Case where a subset is to be selected
        if self.subset_selection_method is not None and self.radius is None:
            input_suffix = self.selected_suffix
            if self.subset_selection_which_traj is not None:
                subset_selection_traj = self.subset_selection_which_traj
            else:
                subset_selection_traj = self.which_traj
            traj_subset_file = f'{self.solute}{solvstr}_{which_targstr}_{subset_selection_traj}_{input_suffix}.traj'
            traj_subset_present = os.path.exists(traj_subset_file) and os.path.getsize(traj_subset_file)>0
            if not traj_subset_present:
                #if self.task_id is not None:
                #    raise Exception(f"# Please run trajectory selection by running with no task_id before running individual task_id's")
                write_subset_trajectory(traj_carved_file,traj_subset_file,nmax=self.subset_selection_nmax,
                                        method=self.subset_selection_method,
                                        min_spacing=self.subset_selection_min_spacing,
                                        bias_beta=self.subset_selection_bias_beta)
            else:
                print(f'# Loading existing subset trajectory: {traj_subset_file}')
                delete_traj_carved_file = False
            traj_carved_file = traj_subset_file
        # Case where a slice must be taken of a pre-existing uncarved trajectory
        if os.path.exists(traj_carved_file) and self.radius is None:
            # Open trajectory via link, then slice if required
            input_traj = Trajectory(traj_carved_file)
            traj_max = min(len(input_traj),self.max_snapshots)
            traj_min = max(0,self.min_snapshots)
            traj_carved = list(enumerate(input_traj[traj_min:traj_max]))
            if self.task_id is not None:
                traj_carved = [traj_carved[self.task_id]]
            print(f'# {len(list(traj_carved))} frames loaded')
            
        writeonly = (self.output=="nw") or (self.output=="dat") or (self.output=="xyz")
        if self.calc_forces:
            mintarget = 0
            if hasattr(self,'target'):
                if isinstance(self.target,list):
                    target = self.target
                else:
                    mintarget = self.target
                    target = list(range(mintarget,self.nroots+1))
            else:
                target = list(range(0,self.nroots+1))
        else:
            target = self.nroots
        if dryrun:
            print(f'# Cluster setup completed, halting before calculation.')
            return
        if writeonly:
            print(f'# Writing input files in {self.output} format.')
        else:
            outputs = "energies"
            targets_str = f"ground state and {self.nroots} excited states"
            if self.calc_forces:
                outputs = outputs + " and forces"
                if len(target)>1:
                    targets_str = f"states {target}"
                else:
                    targets_str = f"state {target}"
            print(f'# Calculating {outputs} for {targets_str} in {self.output}.')
        # Main block for calculation of cluster energies and forces
        if any(out in self.output.lower() for out in ['nw','orca','physnet','mace']):
            from esteem.trajectories import recalculate_trajectory
            seed = f'{self.solute}_{self.solvent}' if self.solvent is not None else self.solute
            traj_label = self.which_traj
            traj_suffix = self.output
            input_target = self.which_target
            # If we are processing the whole trajectory, set the full range and no offset
            input_traj_label = self.which_traj
            if self.task_id is None:
                input_traj_range = None
                output_traj_offset = 0
                # If we are selecting a subset, choose appropriate range
                if self.subset_selection_method is not None and self.radius is None:
                    input_traj_range = range(self.min_snapshots,self.max_snapshots)
                    input_traj_label = subset_selection_traj
                    output_traj_offset = -self.min_snapshots
            else:
                # If we are processing just one carved frame, set range to 1 and offset to task_id
                input_traj_range = range(0,1)
                output_traj_offset = self.task_id
                # If we are processing one frame directly from source, set range to task_id and offset to 0
                if self.radius is None:
                    input_traj_range = range(self.task_id,self.task_id+1)
                    output_traj_offset = 0
                    if self.subset_selection_method is not None:
                        input_traj_label = subset_selection_traj
                        input_traj_range = range(self.min_snapshots+self.task_id,self.min_snapshots+self.task_id+1)
                        output_traj_offset = -self.min_snapshots
            # Now run through the trajectory, calculating singlepoint energy for each frame
            recalculate_trajectory(seed,target,traj_label,traj_suffix,input_target,input_suffix,
                                   self.wrapper,calc_params=calc_params,
                                   input_traj_range=input_traj_range,input_traj_label=input_traj_label,
                                   output_traj_offset=output_traj_offset,
                                   charge=charge,solvent=self.impsolv,calc_forces=self.calc_forces,
                                   geom_opt_kernel=self.geom_opt_kernel,vibfreq_kernel=self.vibfreq_kernel)
            # If we are processing whole trajectory, do sanity checking of result now
            if self.task_id is None and self.ref_mol_dir is not None:
                ref_solu, ref_solv = get_solu_solv_names(seed)
                for targ in target[0:]:
                    if targ==0:
                        ref_solu_t = ref_solu
                    else:
                        ref_solu_t = f'{ref_solu}_{targstr(targ)}'
                    ref_mol_dir = self.ref_mol_dir
                    ref_mol_dir = ref_mol_dir.replace("{target}",targstr(targ))
                    ref_mol_dir = ref_mol_dir.replace("{ref_solv}",ref_solv)
                    ref_solu_dir = f'../{ref_mol_dir}'
                    ref_mol_dir = self.ref_mol_dir
                    ref_mol_dir = ref_mol_dir.replace("{target}","gs")
                    ref_mol_dir = ref_mol_dir.replace("{ref_solv}",ref_solv)
                    ref_solv_dir = f'../{ref_mol_dir}'
                    calc_params['target'] = targ
                    trajname = f"{seed}_{targstr(targ)}_{traj_label}_{traj_suffix}.traj"
                    fails = sanity_check(trajname, self.wrapper, calc_params, ref_solu_dir, ref_solu_t,
                                         ref_solv_dir, ref_solv)
            if self.repeat_without_solute:
                print('#\n# Repeating calculation with solute removed\n#')
                traj_suffix = f'{self.output}_nosolu'
                if self.task_id is not None:
                    input_suffix = f'{self.carved_suffix}_nosolu_{self.task_id:04}'
                else:
                    input_suffix = f'{self.carved_suffix}_nosolu'
                traj_carved_nosolu_file = self.remove_solute(traj_carved,self.solute,self.solvent,seed,traj_label,traj_suffix)
                # Now run through the trajectory, calculating singlepoint energy for each frame
                recalculate_trajectory(seed,target,traj_label,traj_suffix,input_target,input_suffix,
                                       self.wrapper,calc_params=calc_params,
                                       input_traj_range=input_traj_range,output_traj_offset=output_traj_offset,
                                       charge=0,solvent=self.impsolv,calc_forces=self.calc_forces,
                                       geom_opt_kernel=False,vibfreq_kernel=False)
                # Always remove nosolu traj carved, as it is very quick to recreate it if needed
                os.remove(traj_carved_nosolu_file)


        elif self.output=="dat" or self.output=="onetep":
            self.calc_all_excitations(self.solute,self.solvent,traj_carved,self.wrapper.excitations,charge,
                                 self.calc_params,self.impsolv,self.nroots,writeonly,
                                 self.continuation,self.cleanup)
        elif self.output=="xyz":
            self.write_traj_xyz(f'{self.solute}_{self.solvent}_{self.md_suffix}',traj_carved)
        elif self.output=="amber":
            self.write_amber_minimised(self.solute,self.solvent,traj_carved)
        else:
            raise Exception('Error: Unrecognised output format. The value of self.output was: {}'.format(self.output))

        # Delete any temporary trajectory files made during a task-parallel run
        if self.task_id is not None and traj_carved_file is not None and delete_traj_carved_file:
            os.remove(traj_carved_file)
            traj_carved_file = None

    def get_input_traj(self,solute,solvent,md_suffix):

        from os import path
        from esteem.trajectories import merge_traj
        from ase.io import Trajectory
        
        # Check if we are merging trajectories before use
        if isinstance(md_suffix,list):
            all_trajs = [f'{solute}_{solvent}_{suffix}.traj' for suffix in md_suffix]
            merged_traj_name = f'{solute}_{solvent}_{md_suffix[0]}_merged.traj'
            if self.task_id is not None:
                if (not path.isfile(merged_traj_name)) or path.getsize(merged_traj_name)==0:
                    print(f'File not found (or file is empty): {merged_traj_name}')
                    raise Exception(f"Please merge trajectories by running with no task_id before running individual task_id's")
            else:
                # Only do the merging if run interactively - prevents collision in array jobs
                merge_traj(all_trajs,merged_traj_name)
            print(f'# Reading from merged trajectory {merged_traj_name}')
            traj = Trajectory(merged_traj_name)
        else:
            input_traj_name = f'{solute}_{solvent}_{md_suffix}.traj'
            print(f'# Reading from input trajectory {input_traj_name}')
            traj = Trajectory(input_traj_name)

        return traj

    def carve_spheres(self,counterions=None,solvent_radius=0.0,kernel_radius=0.0,
                      nmol_solvent_targ=None,boxsize=None,task_id=None):
        """Carves out spheres from a periodic solvent model, centered on the solute"""
        
        from ase.io import Trajectory, read
        from esteem.trajectories import targstr
        from os import path

        # Determine trajectory names and open trajectory
        input_traj = self.get_input_traj(self.solute,self.solvent,self.md_suffix)
        which_targstr = targstr(self.which_target)
        
        if self.max_snapshots is None:
            self.max_snapshots = 1e10
        traj_max = min(len(input_traj),self.max_snapshots)
        traj_min = max(0,self.min_snapshots)

        if task_id is not None:
            if task_id >= traj_max or task_id < 0:
                raise Exception(f"Invalid task_id: {task_id} outside range 0:{traj_max}")

        if task_id is not None:
            input_traj = [input_traj[traj_min+task_id]]
            print(f'# Input trajectory frame {traj_min+task_id}')
            traj_carved_file = f'{self.solute}_{self.solvent}_{which_targstr}_{self.which_traj}_{self.carved_suffix}_{task_id:04}.traj'
            traj_min = 0
            traj_max = 1
        else:
            input_traj = input_traj[traj_min:traj_max]
            print(f'# Input trajectory frame range {traj_min}:{traj_max}')
            traj_carved_file = f'{self.solute}_{self.solvent}_{which_targstr}_{self.which_traj}_{self.carved_suffix}.traj'
            task_id = 0
        print(f'# Writing carved snapshots to {traj_carved_file}')
        traj_carved = Trajectory(traj_carved_file,'w')

        # Find counterions, if set
        counterion_atoms = Atoms()
        if isinstance(counterions,dict):
            if self.solute in counterions:
                counterion_atoms = Atoms(counterions[self.solute])
        if isinstance(counterions,str):
            counterion_atoms = Atoms(counterions)

        nat_solute = len(read(self.solute+".xyz"))
        nat_solvent = len(read(self.solvent+".xyz"))
        nat_counterions = len(counterion_atoms)
        
        if self.max_atoms is not None and nmol_solvent_targ is None:
            nmol_solvent_targ = int((self.max_atoms - nat_solute)/nat_solvent)

        for t in input_traj[0:traj_max-traj_min]:
            nat_tot = len(t)
            nmol_solvent = int((nat_tot-nat_solute-nat_counterions)/nat_solvent)
            carve_sphere(t,solvent_radius,kernel_radius,nat_solute,nat_solvent,
                         nat_counterions,nmol_solvent,nmol_solvent_targ,
                         rotate=self.rotate,boxsize=boxsize)
            # Write model to trajectory
            traj_carved.write(t)

        traj_carved.close()
        traj_carved = Trajectory(traj_carved_file)
        return list(enumerate(traj_carved,start=task_id)),traj_carved_file
    
    def remove_solute(self,traj_carved,soluseed,solvseed,seed,traj_label,traj_suffix,task_id=None):
        """Removes solute molecules from a previously-carved trajectory"""

        from ase.io import Trajectory, read
        from esteem.trajectories import targstr

        which_targstr = targstr(self.which_target)
        if task_id is not None:
            traj_nosolu_name = f'{soluseed}_{solvseed}_{which_targstr}_{self.which_traj}_carved_nosolu_{task_id:04}.traj'
            traj_min = 0
            traj_max = 1
            traj_carved = traj_carved[traj_min:traj_max]
        else:
            traj_nosolu_name = f'{soluseed}_{solvseed}_{which_targstr}_{self.which_traj}_carved_nosolu.traj'
            task_id = 0
        traj_nosolu = Trajectory(traj_nosolu_name,"w")
        nat_solu = len(read(soluseed+".xyz"))
        for i,f in traj_carved:
            if len(f)==nat_solu:
                print(f'No solvent found in frame {i}, writing solute')
                traj_nosolu.write(f)
            else:
                traj_nosolu.write(f[nat_solu:])
        traj_nosolu.close()
        #traj_nosolu = Trajectory(traj_nosolu_name)
        #return list(enumerate(traj_nosolu,start=task_id)),traj_nosolu_name
        return traj_nosolu_name

    def calc_all_excitations(self,soluseed,solvseed,traj_carved,excit_func,
                             charge=0,calc_params={},impsolv=None,nroots=1,
                             writeonly=False,continuation=False,cleanup=False):
        """Loop over trajectory frames and do an excited states calculation for each one"""
        for i,t in traj_carved:
            frameseed = f'{soluseed}_{solvseed}_{self.md_suffix}{i:04}'
            excitations, energies = excit_func(t,frameseed,nroots=nroots,
                                               calc_params=calc_params,solvent=impsolv,charge=charge,
                                               writeonly=writeonly,continuation=continuation,
                                               cleanup=cleanup)
            print(f'Excitations for {frameseed}: {excitations}')

    def calc_all_vibrations(self,soluseed,solvseed,traj_carved,geom_opt_func,freq_func,
                            charge=0,calc_params={},impsolv=None,nroots=1,
                            writeonly=False,continuation=False,cleanup=False):

        """Loop over trajectory frames and do a vibrational frequency calculation for each one"""
        for i,t in traj_carved:
            frameseed = f'{soluseed}_{solvseed}_{self.md_suffix}{i:04}'
            from ase.constraints import FixAtoms
            c = FixAtoms(indices=[atom.index for atom in t if atom.tag == 0])
            t.set_constraint(c)
            energy,force,positions = geom_opt_func(t,frameseed,
                                                  calc_params=calc_params,solvent=impsolv,charge=charge,
                                                  writeonly=writeonly,continuation=continuation,
                                                  cleanup=cleanup)
            freq_func(t,frameseed,calc_params=calc_params,solvent=impsolv,charge=charge,
                      writeonly=writeonly,continuation=continuation,
                      cleanup=cleanup)
            print(f'Vibrational Frequencies for {frameseed} completed')


    def write_traj_xyz(self,seed,traj_carved):
        for i,t in traj_carved:
            frameseed = f'{seed}{i:04}'
            write(frameseed+".xyz",t)

    # Load routines from Amber.py
    def write_amber_minimised(self,soluseed,solvseed,traj_carved):

        from os import path, makedirs, getcwd, chdir
        from shutil import copyfile
        from ase.io import read, write

        seed = f'{self.solute}_{self.solvent}'
        target = None
        traj_label = self.which_traj
        traj_suffix = self.output
        input_target = 0
        if self.task_id is None:
            input_suffix = 'carved'
        else:
            input_suffix = f'carved_{self.task_id:04}'

        # Create directory for output files and change to it
        wdir = traj_suffix
        origdir = getcwd()
        if not path.exists(wdir):
            print(f'# Creating directory {wdir}')
            try:
                makedirs(wdir)
            except FileExistsError:
                print(f'# Possible mid-air collision between jobs - directory {wdir} exists')
        copyfile(f'{soluseed}.xyz',f'{wdir}/{soluseed}.xyz')
        copyfile(f'{solvseed}.xyz',f'{wdir}/{solvseed}.xyz')
        chdir(wdir)

        nat_solute = len(read(f'{soluseed}.xyz'))
        nat_solvent = len(read(f'{solvseed}.xyz'))
        
        # Set large Ewald cutoff radius
        self.wrapper.cut = 40

        # Prepare Solute inputs
        if not path.exists(f'{soluseed}.prmtop'):
            print(f'Preparing Amber inputs for {soluseed} solute')
            self.wrapper.prepare_input(soluseed,netcharge=0,offset=0)
        else:
            print(f'Using pre-existing Amber inputs for {soluseed} solute')
        solute = read(f'{soluseed}.pdb')
        e0_am_solu = self.wrapper.singlepoint(solute,soluseed)
        print(f'Amber solute ground state energy: {e0_am_solu}')

        # Prepare Solvent inputs
        if not path.exists(f'{solvseed}.prmtop'):
            print(f'Preparing Amber inputs for {solvseed} solvent')
            if solvseed == soluseed:
                offset = 0
            else:
                offset = 99
            self.wrapper.prepare_input(solvseed,netcharge=0,offset=offset)
        else:
            print(f'Using pre-existing Amber inputs for {solvseed} solvent')
        solvent = read(f'{solvseed}.pdb')
        e0_am_solv = self.wrapper.singlepoint(solvent,solvseed)
        print(f'Amber solvent ground state energy: {e0_am_solv}')

        # Write parameter and topology file for this frame via Amber wrapper
        print(f'Creating frame prmtop files for {soluseed} in {solvseed} clusters')
        for i,t in traj_carved:
            nat_tot = len(t)
            nmol_solvent = int((nat_tot-nat_solute)/nat_solvent)
            label = f'{seed}_gs_{traj_label}_{traj_suffix}{i:04}'
            write(f'{label}.pdb',t)
            if True: # set to false to deactivate writing prmtop and use existing versions
                self.wrapper.make_frame_prmtop(label,t,soluseed,solvseed,nat_solute,nat_solvent,nmol_solvent)
        chdir(origdir) # go back to parent directory as recalculate_trajectory expects to start there

        # Recalculate trajectory energies with Amber wrapper
        from esteem.trajectories import recalculate_trajectory
        calc_params = {}
        recalculate_trajectory(seed,target,traj_label,traj_suffix,input_target,input_suffix,
                               self.wrapper.singlepoint,calc_params=calc_params)

    def make_parser(self):

        # Parse command line values
        main_help = ('Given a trajectory containing MD snapshots of a solvated \n'+
                     'molecule, carve out spheres of a given radius to produce \n'+
                     'inputs for explicit solvent calculations.')
        epi_help = ('Note: Expects the input trajectory to be named in the format \n'+
                    '<solute>_<solvent>_<md_suffix>.traj\n'+
                    'Writes output to trajectory <solute>_<solvent>_carved.traj \n'+
                    'and to <solute>_<solvent>_solvXXX.<ext> where XXX is the \n'+
                    'index of the snapshot and <ext> is the file extension of the \n'+
                    'input file for the code being used (specified by -o).')
        from argparse import ArgumentParser, SUPPRESS
        parser = ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--solute','-u',required=False,type=str,help='Solute name')
        parser.add_argument('--solvent','-v',required=False,type=str,help='Solvent name')
        parser.add_argument('--radius','-r',default=5.0,type=float,help='Maximum distance from any atom of a solvent molecule to atom of a solute molecule for it to be included in cluster')
        parser.add_argument('--kernel','-k',default=0.0,type=float,help='Maximum distance from any atom of a solvent molecule to atom of a solute molecule for it to be included in tddft kernel')
        parser.add_argument('--max_solvent_mols','-M',default=None,type=int,help='Maximum number of solvent molecules to include (chosen randomly)')
        parser.add_argument('--max_atoms','-H',default=None,type=int,help='Maximum number of atoms to include in carved sphere')
        parser.add_argument('--boxsize','-B',default='50.0',type=float,help='Size of box in which electronic excitation calculation is performed')
        parser.add_argument('--rotate','-O',default=True,type=bool,help='If True, rotate the molecule to align with xy plane')
        parser.add_argument('--output','-o',default='nw',type=str,help='Format of output: takes values nw, dat, nwchem and onetep (former 2 write input files only, latter 2 perform calculation)')
        parser.add_argument('--nroots','-n',default=5,type=int,help='Number of excitations to calculate')
        parser.add_argument('--calc_forces','-F',default=True,type=bool,help='Whether to calculate forces on each snapshot')
        parser.add_argument('--task_id','-t',default=None,type=int,help='Task ID of the current job, inherited from driver')
        parser.add_argument('--cleanup','-c',default=False,action='store_true',help='Whether to delete all temporary files after the job')
        parser.add_argument('--continuation','-N',default=False,action='store_true',help='Whether to continue from a previous run of this file')
        parser.add_argument('--counterions','-C',default={},type=str,help='Counterion(s) to add, eg Na')
        parser.add_argument('--charges','-Q',default={},nargs='?',type=dict,help='Charges on molecular species. Not for command-line use')
        parser.add_argument('--max_snapshots','-S',default=None,type=int,help='Maximum snapshot to process')
        parser.add_argument('--min_snapshots','-I',default=None,type=int,help='Minimum snapshot to process')
        parser.add_argument('--valid_snapshots','-D',default=None,type=int,help='Number of validation snapshots to process')
        parser.add_argument('--subset_selection_method','-Y',default=None,type=str,help='Method for selecting subset from input trajectory')
        parser.add_argument('--subset_selection_nmax','-y',default=None,type=int,help='Number of frames to select in subset from input trajectory')
        parser.add_argument('--subset_selection_min_spacing','-g',default=1,type=int,help='Minimum frame spacing for selecting subset from input trajectory')
        parser.add_argument('--subset_selection_bias_beta','-a',default=20,type=int,help='Effective inverse temperature 1/kBT (in eV-1) for bias on U trajectories')
        parser.add_argument('--subset_selection_which_traj','-J',default=None,type=str,help='Which trajectory name to read/write subset selection to')
        parser.add_argument('--repeat_without_solute','-W',default=False,type=bool,help='Repeat calculation with no solute present')
        parser.add_argument('--geom_opt_kernel','-G',default=False,type=bool,help='Geometry Optimise the kernel region')
        parser.add_argument('--vibfreq_kernel','-X',default=False,type=bool,help='Calculate vibrational frequencies of kernel region')
        parser.add_argument('--which_traj','-w',default='A',type=str,help='Which trajectory set to use (default is A, active learning uses higher)')
        parser.add_argument('--which_target','-T',default='0',type=int,help='Which target to use in trajectory names')
        parser.add_argument('--md_prefix','-p',default="{solu}_{solv}_md",nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--md_suffix','-m',default="solv",nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--carved_suffix','-V',default="carved",nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--selected_suffix','-U',default="selected",nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--exc_suffix','-e',default="exc",nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--ref_mol_dir','-l',default="{target}_PBE0",type=str,help='Location of output of solutes run from which to find reference energies')
        # Wrapper Dependent
        parser.add_argument('--calc_params','-q',default={},nargs='?',type=str,help=SUPPRESS)
        parser.add_argument('--calc_seed','-Z',default=None,type=str,help='Seed for the calculator')
        parser.add_argument('--calc_suffix','-K',default='',type=str,help='Suffix for the calculator (often specifies ML hyperparameters)')
        parser.add_argument('--calc_prefix','-P',default='',type=str,help='Prefix for the calculator (often specifies directory)')
        parser.add_argument('--basis','-b',default='6-311++G**',nargs='*',type=str,help='Basis string or tuple')
        parser.add_argument('--func','-f',default='PBE0',type=str,help='Functional for electronic excitation calculation')
        parser.add_argument('--disp','-d',default=True,type=bool,help='Dispersion correction (set to True to activate)')
        parser.add_argument('--impsolv','-i',default=None,type=str,help='Implicit solvent string (may differ from solvent name)')

        return parser

    def validate_args(args):
        default_args = make_parser().parse_args(['--solute','a','--solvent','b'])
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Helper Functions to delete/label molecules and rotate clusters

# In[ ]:


import numpy as np
from ase import Atoms

def carve_sphere(t,solvent_radius,kernel_radius,nat_solute,nat_solvent,
                 nat_counterions,nmol_solvent,nmol_solvent_targ,boxsize=None,
                 rotate=True,deleted_atoms=None):

    nat_tot = len(t)

    # Delete counterions further away than solvent_radius from
    # any solute atom
    if (nat_counterions>0):
        t = delete_counterions(t,solvent_radius,nat_tot,nat_solute,nat_counterions,
                               deleted_atoms)

    # Delete solvent molecules further away than solvent_radius from
    # any solute atom
    t = delete_distant_molecules(t,solvent_radius,nat_tot-nat_counterions,
                                 nat_solvent,nmol_solvent,nat_solute,deleted_atoms)
    
    if nmol_solvent_targ is not None:
        #t = delete_excess_molecules(t,nat_tot-nat_counterions,nat_solvent,nat_solute,
        #                            nmol_solvent_targ,deleted_atoms)
        t = delete_furthest_molecules(t,nat_tot-nat_counterions,nat_solvent,nat_solute,
                                      nmol_solvent_targ,deleted_atoms)

    # Set a tag of 2 on the nearby solvent molecules within kernel_radius
    # from a solute molecule. Also gives tag 1 to solute molecule atoms.
    t = label_nearby_molecules(t,kernel_radius,nat_tot-nat_counterions,
                               nat_solvent,nmol_solvent,nat_solute)

    # Reimage cluster so that everything is in MIC relative to solute
    if not (t.cell==0.0).all():
        t = reimage_cluster(t,nat_solvent,nat_solute)

    # Rotates model so longest axes of solute are in common plane for
    # ease of viewing, and adds a box
    if deleted_atoms==None and rotate:
        t = rotate_and_center_solute(t,nat_solute,boxsize)

def delete_counterions(t,solvent_radius,nat_tot,nat_solute,nat_counterions,
                       deleted_atoms=None):
    """
    Deletes counterions from an Atoms model.
    Assumes they appear in the range [nat_solute:nat_solute+nat_counterions]
    """

    Rij = np.zeros((nat_counterions,nat_solute))

    # Get entries in matrix of distances between all solute atoms and counterions
    for i in range(nat_solute):
        Rij[:,i] = np.array(t.get_distances(i,range(nat_solute,nat_solute+nat_counterions),mic=True))
    # Reshape to (number of counterions) x (atoms in solute)
    Rij = Rij.reshape((nat_counterions,nat_solute))
    # Turn into Boolean values of whether distance is less than threshold
    flags = (Rij<solvent_radius)
    # Reduce over solute atoms
    flags = np.any(flags,axis=1)
    # Delete all the atoms we don't want
    del_idx = [i for i in range(nat_solute,nat_solute+nat_counterions) if flags[i-nat_solute]==0]
    if len(del_idx)!=nat_counterions:
        print('WARNING: counterion(s) not deleted! Too close to solute')
    if deleted_atoms is not None:
        deleted_atoms = deleted_atoms + t[del_idx]
    del t[del_idx]

    return t


def delete_excess_molecules(t,nat_tot,nat_solvent,nat_solute,nmol_solvent_targ,
                            deleted_atoms=None):
    """
    Deletes solvent molecules
    """
    import random
    nat_tot_cluster = len(t)
    nmol_solvent = int((nat_tot_cluster-nat_solute)/nat_solvent)
    solvent_mols_to_keep = random.sample(range(nmol_solvent),nmol_solvent_targ)
    del_idx = [i for i in range(nat_solute,nat_tot_cluster) 
               if (i-nat_solute)//nat_solvent not in solvent_mols_to_keep]
    if deleted_atoms is not None:
        deleted_atoms = deleted_atoms + t[del_idx]
    del t[del_idx]
    return t

def delete_furthest_molecules(t,nat_tot,nat_solvent,nat_solute,nmol_solvent_targ,
                              deleted_atoms=None):
    nat_tot_cluster = len(t)
    nmol_solvent = int((nat_tot_cluster-nat_solute)/nat_solvent)
    Rij = np.zeros((nat_solvent*nmol_solvent,nat_solute))
    # Get entries in matrix of distances between all solute and solvent atoms
    if nat_tot==nat_solute:
        for i in range(nat_solute):
            Rij[:,i] = np.zeros((0))
    else:
        for i in range(nat_solute):
            Rij[:,i] = np.array(t.get_distances(i,range(nat_solute,nat_tot_cluster),mic=True))
    # Reshape to (number of solvent molecules) x (atoms in each solvent molecule) x (atoms in solute)
    Rij = Rij.reshape((nmol_solvent,nat_solvent,nat_solute))
    # Find lowest distance between any solute atom and any solvent atom in each molecule
    dist_to_mol = np.min(np.min(Rij,axis=1),axis=1)
    # Sort distances lowest to highest and make list of those beyond nmol_solvent_targ
    mols_to_del = np.argsort(dist_to_mol)[nmol_solvent_targ:]
    # Delete each molecule in this list
    del_idx = [i for i in range(nat_solute,nat_tot_cluster) 
               if (i-nat_solute)//nat_solvent in mols_to_del]
    if deleted_atoms is not None:
        deleted_atoms = deleted_atoms + t[del_idx]
    del t[del_idx]
    # return trimmed cluster
    return t

def reimage_cluster(t,nat_solvent,nat_solute):
    """
    Translates whole solvent molecules by lattice vectors to minimise distance to solute
    """
    from ase.geometry.geometry import wrap_positions

    solu_av_pos_frac = np.linalg.solve(t.cell.T,np.asarray(np.mean(t[0:nat_solute].get_positions(),axis=0)).T).T
    nmol_solvent = int((len(t)-nat_solute)/nat_solvent)
    solv_av_pos = np.zeros((nmol_solvent,3))
    for i in range(nmol_solvent):
        solv_av_pos[i] = np.mean(t[i*nat_solvent+nat_solute:(i+1)*nat_solvent+nat_solute].get_positions(),axis=0)
    solv_av_pos_wrapped = wrap_positions(solv_av_pos, t.cell, center=solu_av_pos_frac)
    shift = np.repeat(solv_av_pos_wrapped-solv_av_pos,nat_solvent,axis=0)
    t.positions[nat_solute:] += shift

    return t

def delete_distant_molecules(t,solvent_radius,nat_tot,nat_solvent,nmol_solvent,nat_solute,
                             deleted_atoms=None,keep_idx=None):
    """
    Deletes solvent molecules beyond a certain radius from the solute from an Atoms model
    Assumes that the first nat_solute atoms are the solute, and after that solvent
    atoms are arranged in nmol_solvent groups of size nat_solvent
    """

    Rij = np.zeros((nat_solvent*nmol_solvent,nat_solute))
    # Get entries in matrix of distances between all solute and solvent atoms
    if nat_tot==nat_solute:
        for i in range(nat_solute):
            Rij[:,i] = np.zeros((0))
    else:
        for i in range(nat_solute):
            Rij[:,i] = np.array(t.get_distances(i,range(nat_solute,nat_tot),mic=True))
    # Reshape to (number of solvent molecules) x (atoms in each solvent molecule) x (atoms in solute)
    Rij = Rij.reshape((nmol_solvent,nat_solvent,nat_solute))
    # Turn into Boolean values of whether distance is less than threshold
    flags = (Rij<solvent_radius)
    # Reduce over solute atoms each solvent atom could be in range of
    flags = np.any(flags,axis=2)
    # Reduce over solvent atoms in the same molecule as a solvent atom which is in range of a solute atom
    flags = np.any(flags,axis=1)
    # Expand out so there is the same entry for each atom in each molecule
    flags = np.repeat(flags,nat_solvent)
    # Delete all the atoms we don't want
    del_idx = [i for i in range(nat_solute,nat_tot) if flags[i-nat_solute]==0]
    if deleted_atoms is not None:
        deleted_atoms = deleted_atoms + t[del_idx]
    if keep_idx is not None:
        keep_idx.extend([i for i in range(nat_solute)])
        keep_idx.extend([i for i in range(nat_solute,nat_tot) if flags[i-nat_solute]!=0])
    del t[del_idx]

    return t

def label_nearby_molecules(t,kernel_radius,nat_tot,nat_solvent,nmol_solvent,nat_solute):
    """
    Adds a tag to solvent molecules within a certain radius from the solute to an Atoms model
    Assumes that the first nat_solute atoms are the solute, and after that solvent
    atoms are arranged in nmol_solvent groups of size nat_solvent
    """

    if kernel_radius is None:
        return t
    nat_tot_cluster = len(t)
    nmol_solvent_cluster = int((nat_tot_cluster-nat_solute)/nat_solvent)
    Rij = np.zeros((nat_solvent*nmol_solvent_cluster,nat_solute))
    # Get entries in matrix of distances between all solute and solvent atoms
    if nmol_solvent_cluster>0:
        for i in range(nat_solute):
            Rij[:,i] = np.array(t.get_distances(i,range(nat_solute,nat_tot_cluster),mic=True))
    # Reshape to (number of solvent molecules) x (atoms in each solvent molecule) x (atoms in solute)
    Rij = Rij.reshape((nmol_solvent_cluster,nat_solvent,nat_solute))
    # Turn into Boolean values of whether distance is less than threshold
    flags = (Rij<kernel_radius)
    # Reduce over solute atoms each solvent atom could be in range of
    flags = np.any(flags,axis=2)
    # Reduce over solvent atoms in the same molecule as a solvent atom which is in range of a solute atom
    flags = np.any(flags,axis=1)
    # Expand out so there is the same entry for each atom in each molecule
    flags = np.repeat(flags,nat_solvent)
    # Set all tags to zero
    tags = np.zeros(nat_tot_cluster)
    # Tag the solute molecule as "1"
    tags[0:nat_solute] = 1
    # Tag the rest as "2"
    tags[nat_solute:nat_tot_cluster] = 2
    # Now find those in range kernel_radius and tag them as "0"
    flag_idx = [i for i in range(nat_solute,nat_tot_cluster) if flags[i-nat_solute]==0]
    for i in flag_idx:
        tags[i] = 0
    t.set_tags(tags)

    return t

def rotate_and_center_solute(t,nat_solute=None,boxsize=None):
    """Rotates a cluster model so the solute is centered and lies in the xy plane"""

    # Extract solute atoms as separate atoms object
    if nat_solute is None:
        nat_solute = len(t)
    solute = t[0:nat_solute]
    
    # any molecule of less than 10 atoms is not rotated
    rotate_threshold = 10
    if nat_solute>rotate_threshold:
        # Choose atoms along long and short axes
        d=solute.get_all_distances()
        for i in range(len(solute)):
            for j in range(len(solute)):
                if not (solute.get_chemical_symbols()[i]=='C' and 
                        solute.get_chemical_symbols()[j]=='C'):
                    d[i,j]=0
        # Find furthest-apart pair of carbon atoms, these define '1' and '4' atoms
        ind = np.unravel_index(np.argmax(d, axis=None), d.shape)
        p1=ind[0]; p4=ind[1]
        # Find next-furthest-apart pair of carbon atoms, defining '2' and '3' atoms
        d[p1,p4]=0; d[p4,p1]=0
        ind2 = np.unravel_index(np.argmax(d, axis=None), d.shape)
        # Ensure consistent orientation by choosing lowest overall index as '1'
        # and set others based on ensuring |p1-p2| < |p1-p3|
        p1 = min(min(ind,ind2)); p4 = max(min(ind,ind2));
        p2 = min(max(ind,ind2)); p3 = max(max(ind,ind2));
        if solute.get_distance(p1,p2)<solute.get_distance(p1,p3):
            p3 = min(max(ind,ind2)); p2 = max(max(ind,ind2))            
        # Rotate so average of p1-p2 and p3-p4 points along x
        vector1 = (t.positions[p1] - t.positions[p2] + 
                  t.positions[p3] - t.positions[p4])
        t.rotate(vector1,(1,0,0))
        # Rotate so average of p1-p3 and p2-p4 points along y
        vector2 = (t.positions[p3] - t.positions[p1] +
                  t.positions[p4] - t.positions[p2])
        t.rotate(vector2,(0,1,0))
        # Rotate so cross product points along z
        vector1 = (t.positions[p1] - t.positions[p2] + 
                  t.positions[p3] - t.positions[p4])
        vector2 = (t.positions[p3] - t.positions[p1] +
                  t.positions[p4] - t.positions[p2])
        t.rotate(np.cross(vector1,vector2),(0,0,1))

    # Translate so center is at middle of box and set cell
    if boxsize is not None:
        # use average of the positions of the carbon atoms defining the longest
        # lengths in the molecule to define the "center"
        if nat_solute>rotate_threshold:
            vector = ((boxsize*0.5,boxsize*0.5,boxsize*0.5) -
                (t.positions[p1]+t.positions[p2]+t.positions[p3]+t.positions[p4])/4)
        else:
            vector = (boxsize*0.5,boxsize*0.5,boxsize*0.5) - solute.get_center_of_mass()
        t.translate(vector)
        t.set_cell(boxsize * np.identity(3))
    else:
        # center molecule on origin
        if nat_solute>rotate_threshold:
            vector = -(t.positions[p1]+t.positions[p2]+t.positions[p3]+t.positions[p4])/4
        else:
            vector = -solute.get_center_of_mass()
        t.translate(vector)
        t.pbc = False
        t.set_cell(None)

    return t

def dist_test(mol):
    all_dist = mol.get_all_distances()
    if (all_dist>5).any() or np.logical_and((all_dist<0.82),(all_dist>0)).any():
        print(all_dist)
        return True
    else:
        return False

def get_ref_mol_energy(wrapper,ref_mol,solv,calc_params,ref_mol_xyz,ref_mol_dir,silent=True):

    from os import getcwd, chdir
    from ase.io import read
    from esteem.trajectories import atom_energy

    # Load reference molecule calculation from Solutes calculation
    if not silent:
        print(f'# Reading reference molecule from {ref_mol_xyz}')
    ref_mol_model = read(ref_mol_xyz)
    ref_mol_calc_dir = f'{ref_mol_dir}/geom/{ref_mol}'
    ref_mol_seed = f'{ref_mol}_{solv}'
    if not silent:
        print(f'# Reading reference molecule calculation from {ref_mol_calc_dir}/{ref_mol_seed}')
    orig_dir = getcwd()
    chdir(f'{ref_mol_calc_dir}')
    if hasattr(wrapper,'atom_energies'):
        if wrapper.atom_energies != {}:
            wrapper.atom_e = atom_energy(ref_mol_model,wrapper.atom_energies)
    if hasattr(wrapper,'calc'):
        if hasattr(wrapper.calc,'use_neighborlist'):
            if wrapper.calc.use_neighborlist:
                if ref_mol_model.cell.volume == 0.0:
                    from ase.geometry import Cell
                    ref_mol_model.cell = Cell([[40,0,0],[0,40,0],[0,0,40]])
    ref_mol_energy,calc = wrapper.singlepoint(ref_mol_model,
                ref_mol_seed,calc_params,forces=False,dipole=False,readonly=True)
    chdir(orig_dir)
    return ref_mol_energy, ref_mol_model

def write_subset_trajectory(trajin_file,trajout_file,nmax,method='R',min_spacing=1,bias_beta=20.0):
    from ase.io import Trajectory
    t=Trajectory(trajin_file)
    to=Trajectory(trajout_file,"w")
    t0_en = t[0].get_potential_energy()
    if not isinstance(t0_en,np.ndarray):
        raise Exception(f"# Expected a trajectory with energies of type np.ndarray, got {type(t0_en)}")
    if nmax > len(t):
        raise Exception(f"# Not enough frames to write {nmax}: found {len(t)} in trajectory")
    fullmethod = {'R':'(R) Random',
                  'E':'(E) Highest Energy Std Dev',
                  'D':'(D) Highest Dipole Std Dev',
                  'U':f'(U) Biased by Energy Std Dev with beta={bias_beta}'}
    fullmethod = fullmethod[method]
    print(f'# Attempting to choose {nmax} frames out of {len(t)}, using method: {fullmethod} with min_spacing = {min_spacing}')
    stde=np.zeros(len(t))
    for i,f in enumerate(t):
        stde[i]=np.std(f.get_potential_energy())/len(f)
    # energy standard deviation-based sorting
    if method=='E':
        args=np.argsort(stde)
    # dipole-based sorting
    if method=='D':
        stdd=np.zeros(len(t))
        std_dip=np.zeros((len(t),3))
        for i,f in enumerate(t):
            std_dip[i]=np.std(f.get_dipole_moment(),axis=0)
            stdd[i] = np.sqrt(np.sum(std_dip[i]*std_dip[i]))
        args=np.argsort(stdd)
    # energy standard deviation-based biasing, or random choice
    if method=='U' or method=='R':
        from numpy.random import default_rng
        rng = default_rng()
        p = np.exp(stde*bias_beta) if method=='U' else np.ones(len(t))
        p /= np.sum(p)
        args = rng.choice(range(len(t)),size=len(t),replace=False,p=p)
        args = args[::-1] if method=='U' else args
    framelist = []
    for i in range(len(t)-1,-1,-1):
        if all([abs(args[i]-k) >= min_spacing for k in framelist]):
            #print(f'added frame {args[i]} as entry {len(framelist)}')
            framelist.append(args[i])
        if len(framelist)==nmax:
            break
    if len(framelist)!=nmax:
        raise Exception(f"Could not find enough frames to pick {nmax} with min_spacing={min_spacing} from {len(t)}")
    #framelist = list(args[(len(t)-nmax):len(t)])
    print(f'# Chosen frames:\n# {framelist}')
    print(f'# Chosen frames sorted ascending:\n# {sorted(framelist)}')
    stdelist = stde[framelist]
    print(f'# Standard deviations of E for chosen frames:\n# {stdelist}')
    print(f'# Average standard deviation of E for chosen frames: {np.mean(stdelist)}')
    print(f'# Average standard deviation of E for all frames: {np.mean(stde)}')
    for i in range(nmax):
        to.write(t[args[-i-1]])

def sanity_check(trajname='', wrapper=None, calc_params = {},
                 ref_solu_dir='', ref_solu='', ref_solv_dir='', ref_solv=''):

    """Function to check a trajectory has reasonable dipole moments, energies, and forces.
       return the index of suspected failed calculations (or empty list if none)"""

    from os import path, getcwd, chdir
    from ase.io import Trajectory
    from esteem.trajectories import atom_energy

    # Read in Reference E, f, p
    ref_mol_xyz = f'{ref_solu_dir}/is_opt_{ref_solv}/{ref_solu}.xyz'
    solu_energy,solu_model = get_ref_mol_energy(wrapper,ref_solu,ref_solv,calc_params,ref_mol_xyz,ref_solu_dir)
    if isinstance(solu_energy,np.ndarray):
        solu_energy = np.mean(solu_energy)
    ref_mol_xyz = f'{ref_solv_dir}/is_opt_{ref_solv}/{ref_solv}.xyz'
    solv_energy,solv_model = get_ref_mol_energy(wrapper,ref_solv,ref_solv,calc_params,ref_mol_xyz,ref_solv_dir)
    if isinstance(solv_energy,np.ndarray):
        solv_energy = np.mean(solv_energy)
    ref_solu_d = np.linalg.norm(solu_model.get_dipole_moment())
    ref_solv_d = np.linalg.norm(solv_model.get_dipole_moment())
    print('# Solute reference energy, dipole: ',solu_energy,ref_solu_d)
    print('# Solvent reference energy, dipole: ',solv_energy,ref_solv_d)

    # Read in the trajectory
    traj = Trajectory(trajname)
    targ = calc_params['target']

    # Loop over trajectory making a list of frames that fail the sanity check
    fails = []
    for i,frame in enumerate(traj):
        read_success = False
        #try:
        if True:
            n = int((len(frame)-len(solu_model))/len(solv_model))
            e = frame.get_potential_energy()
            #if hasattr(wrapper,'atom_energies'):
            #    if isinstance(e,np.ndarray):
            #        e = np.float64(e[0])
            #    e = e + atom_energy(frame,wrapper.atom_energies)
            if isinstance(e,np.ndarray):
                e = np.mean(e)
            f = frame.get_forces()
            d = frame.get_dipole_moment()
            read_success = True
        #except Exception as excp:
        #    print(excp)
        #    fails.append(i)
        eref = solu_energy + n*solv_energy
        try:
            if n>0:
                de_per_solv = (e-eref)/n
            else:
                de_per_solv = e-eref
            dnorm = np.linalg.norm(d)
            fnorm = np.linalg.norm(f)/len(frame)
            refdnorm = ref_solv_d*n+ref_solu_d
        except:
            de_per_solv = 0
            dnorm = 0
            fnorm = 0
            refdnorm = 0
        if not read_success:
            print(f'{i:04} {targ:03} {n:03} (JOB EXECUTION FAILED)')
        # Empirical thresholds currently
        elif de_per_solv>0.85 or de_per_solv<0.0 or dnorm>refdnorm*2 or fnorm>1:
            fails.append(i)
            print(f'{i:04} {targ:03} {n:03} {e:16.8f} {eref:16.8f} {de_per_solv:16.8f} {fnorm:16.8f} {dnorm:16.8f} {refdnorm:16.8f}')

    if len(fails)==0:
        howmany = 'No'
    else:
        howmany = len(fails)
    print('# {howmany} frames in the trajectory were found to have possibly alarming energy, force or dipole values')
    print('# Thresholds used: energy deviation < 0.85eV per solv molecule, force norm < 1eV/A/atom, dipole < 2 ref value)')
    
    def energy_check():
        # Energy Check
        return fails
    
    def forces_check():
        # Forces check
        return fails
    
    def dipole_moment_check():
        # Dipole moment check
        # Should never be greater then n times the magnitude of the dipole of a single solvent molecule
        # where n is the number of solvent molecules
        p_tot = np.linalg.norm(np.sum(p, axis=0))
        if p_tot > ref_solv_p:
            raise Exception(f"Possible dipole error at frame {frame} of {trajname} found")
        #else:
        #    print(f'Total dipole within reasonable range.')
    
    def gbw_read_check():
        # Make note of issue where TDDFT run reads existing .gbw file and thinks calculation
        # is unconverged and converges to a new energy
        return fails
    
    return fails


# # Handle inputs

# In[ ]:


def get_parser():
    return ClustersTask().make_parser()

if __name__ == '__main__':
    from esteem.wrappers import nwchem
    clus = ClustersTask()
    args = clus.make_parser().parse_args()
    clus.wrapper = nwchem.NWChemWrapper()
    clus.run()

