#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Functions to help generate and manipulate trajectories
"""


# # Generic MD driver

# In[ ]:


# Generate training (or test) data
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.units import Bohr
from ase import units
import numpy as np
from os import path, makedirs, getcwd, chdir, remove
import shutil

def targstr(targ):
    if targ == 0 or targ is None:
        return "gs"
    else:
        return "es"+str(targ)

def output_header(model,prefix):
    if (model.cell!=0.0).all():
        print(f"{prefix}: Step Targ     Potential_Energy      Kinetic_Energy       Temperature_K     Stress_[0:2]                   Volume")
    else:
        print(f"{prefix}: Step Targ     Potential_Energy      Kinetic_Energy       Temperature_K")

def report_outputs(model,step,targ,prefix):
    from ase.units import kB
    pe = model.get_potential_energy()
    if isinstance(pe,list) or isinstance(pe,np.ndarray):
        pe = float(pe[0])
    ke = model.get_kinetic_energy()
    temp = 2*ke/(3*len(model)*kB)
    if (model.cell!=0.0).all():
        stress = model.get_stress()[0:3]
        vol = model.cell.volume
        print(f"{prefix}: {step:4} {targ:4} {pe:20.10f} {ke:20.10f} {temp:20.10f} {stress} {vol:20.8f}",flush=True)
    else:
        print(f"{prefix}: {step:4} {targ:4} {pe:20.10f} {ke:20.10f} {temp:20.10f}",flush=True)

def get_param(md_param,label):

    if isinstance(md_param,dict):
        if label in md_param:
            param = md_param[label]
        else:
            raise Exception(f"# Error: dictionary provided has no entry for key '{label}'")
    else:
        param = md_param

    return param

def generate_md_trajectory(model,seed,target,traj_label,traj_suffix,wrapper,
                           count_snaps,count_equil,md_steps,md_timestep,md_friction,temp,
                           calc_params,charge=0,solvent=None,constraints=None,dynamics=None,
                           snap_wrapper=None,snap_calc_params=None,
                           continuation=False,store_full_traj=False,debugger=None):
    """
    Runs equilibration followed by statistics generation MD for a given model.
    
    This generic function gets called by both AIMD and by MLMD.
    
    model: ASE Atoms
        Initial geometry for the MD trajectory
    seed: str
        String containing name of molecule and excited state index
    target: int or None
        Excited state index
    traj_label: character or str
        String labelling trajectory (usually A,B,C..)
    seed: str
        String containing name of molecule and excited state index
    traj_suffix: str
        String appended to seed and target state to give full trajectory filename
    wrapper: ESTEEM wrapper class
        Wrapper containing run_md and singlepoint functions
    count_snaps: int
        Number of snapshot runs
    count_equil: int
        Number of equilibration runs
    md_steps: int
        Number of molecular dynamics timesteps in each run above
    md_timestep: float
        Molecular dynamics timestep within the runs
    store_full_traj: bool
        Determines whether the full step-by-step trajectory is written, or just the snapshots each md_steps
    temp: float
        Thermostat temperature (NVT ensemble)
    calc_params: 
        Control-parameters for the wrapper (for QM this is the basis, functional, and target;
        for ML this is the calculator seed, suffix, prefix and target)
    solvent:
        Implicit solvent description
    constraints: wrapper-dependent type
        Variables controlling the constraints
    dynamics: wrapper-dependent type
        Variables controlling the dynamics
    """
    from ase.units import fs
    
    if isinstance(target,list):
        all_targets = target
    else:
        all_targets = [target]        

    # Setup file extensions for restart files
    exts = []
    snap_exts = []
    if snap_wrapper is None:
        snap_wrapper = wrapper
    if snap_calc_params is None:
        snap_calc_params = calc_params
    if hasattr(wrapper,'get_restart_file_exts'):
        exts = wrapper.get_restart_file_exts()
    if hasattr(snap_wrapper,'get_restart_file_exts'):
        snap_exts = snap_wrapper.get_restart_file_exts()

    equil_dir = f'{traj_suffix}_equil'
    md_dir = f'{traj_suffix}_md'
    snaps_dir = f'{traj_suffix}_snaps'
    
    # Create equilibration directory
    if not path.exists(equil_dir):
        print(f"# Creating output directory {equil_dir} for equilibration MD runs")
        try:
            makedirs(equil_dir)
        except FileExistsError:
            print(f"# Possible mid-air collision between jobs - directory {equil_dir} exists")
    # Create MD directory
    if not path.exists(md_dir):
        print(f"# Creating output directory {md_dir} for MD runs")
        try:
            makedirs(md_dir)
        except FileExistsError:
            print(f"# Possible mid-air collision between jobs - directory {md_dir} exists")
    # Create snapshots directory
    if not path.exists(snaps_dir):
        print(f"# Creating output directory {snaps_dir} for snapshots")
        try:
            makedirs(snaps_dir)
        except FileExistsError:
            print(f"# Possible mid-air collision between jobs - directory {snaps_dir} exists")
    origdir = getcwd()
    
    # Ensure the system has a cell if we are doing NPT dynamics
    if hasattr(dynamics,'type'):
        if dynamics.type=="NPT" and (model.cell==0.0).all():
            print('# Setting 10A padding')
            model.center(10)

    # Run Equilibration MD
    if not continuation:
        chdir(equil_dir)
        timestep = get_param(md_timestep,'EQ')
        friction = get_param(md_friction,'EQ')
        output_header(model,'EQ')
        if dynamics is None:
            from types import SimpleNamespace
            dynamics = SimpleNamespace()
            dynamics.type = "LANG"
            dynamics.friction = friction
        dynamics.new_traj = store_full_traj
        targ = all_targets[0]
        print(f"# Starting EQ runs 0 to {count_equil} (each {md_steps} steps of {dynamics.type} dynamics with timestep {timestep/fs} fs)")
        for step in range(0,count_equil):
            steplabel, readonly, cont = cycle_step_labels_and_restarts(
                                         seed,traj_label,
                                         None,equil_dir,md_dir,
                                         targ,targ,targ,step-1,step,step+1,
                                         count_equil,0,exts)
            dynamics = wrapper.run_md(model,steplabel,calc_params,md_steps,timestep,step,
                    temp,solvent=solvent,charge=charge,constraints=constraints,dynamics=dynamics,
                    continuation=cont,readonly=readonly)
            if debugger is not None:
                debugger.check(model)
            report_outputs(model,step,targ,'EQ')

        # Return to base directory
        chdir(origdir)
        print("# Finished Equilibration MD runs")
    else:
        print("# Skipping Equilibration MD runs")

    outtraj = {}
    snaps_done = 0
    for targ in all_targets:
        # Check if output trajectory already exists
        output_traj = f"{seed}_{targstr(targ)}_{traj_label}_{traj_suffix}.traj"
        if path.isfile(output_traj) and not continuation:
            # Delete it and start afresh
            try:
                remove(output_traj)
            except:
                print('# Unable to remove {output_traj} - mid-air collision between jobs? Continuing')

        # Create output trajectory
        if continuation:
            print(f"# Opening output trajectory: {output_traj} for appending")
            outtraj[targ] = Trajectory(output_traj)
            snaps_done = len(outtraj[targ])
            outtraj[targ].close()
            outtraj[targ] = Trajectory(output_traj,"a")
        else:
            print(f"# Opening output trajectory: {output_traj} for writing")
            outtraj[targ] = Trajectory(output_traj,"w")

    # Setup main MD
    timestep = get_param(md_timestep,'MD')
    friction = get_param(md_friction,'MD')
    if dynamics is None:
        from types import SimpleNamespace
        dynamics = SimpleNamespace()
        dynamics.type = "LANG"
        dynamics.friction = friction
        dynamics.new_traj = False
    if snaps_done<count_snaps:
        print(f"# Starting Snapshot MD runs {snaps_done} to {count_snaps} (each {md_steps} steps of {dynamics.type} dynamics with timestep {timestep/fs} fs)")
        output_header(model,'MD')
    else:
        print(f'# Snapshot MD runs already completed - {count_snaps} snapshots found')
    if (count_equil > 0  or continuation) and store_full_traj:
        if continuation:
            print(f"# Warning - continuation functionality for store_full_traj not finished - overwriting")
        dynamics.new_traj = True
    for step in range(snaps_done,count_snaps):
        chdir(md_dir)
        targ = all_targets[0]
        calc_params['target'] = targ
        
        # Run main MD between snapshots
        steplabel, readonly, cont = cycle_step_labels_and_restarts(
                                     seed,traj_label,
                                     equil_dir,md_dir,None,
                                     targ,targ,targ,step-1,step,step+1,
                                     count_snaps,count_equil,exts)
        dynamics = wrapper.run_md(model,steplabel,calc_params,md_steps,timestep,count_equil+step,
                temp,solvent=solvent,charge=charge,constraints=constraints,dynamics=dynamics,
                continuation=cont,readonly=readonly)
        report_outputs(model,step,targ,'MD')

        # Now do singlepoints for (ground and) excited state snapshots, if applicable
        chdir(origdir)
        chdir(snaps_dir)
        prev_targ = targ
        for i,targ in enumerate(all_targets[0:]):
            if targ == all_targets[-1]:
                next_targ = all_targets[0]
                next_step = step + 1
            else:
                next_targ = all_targets[i+1]
                next_step = step
            steplabel, readonly, cont = cycle_step_labels_and_restarts(
                                         seed,traj_label,
                                         md_dir,snaps_dir,None,
                                         prev_targ,targ,next_targ,step,step,next_step,
                                         count_snaps,count_equil,snap_exts)
            readonly=False
            if path.isfile(steplabel+'.out') or path.isfile(steplabel+'.nwo'):
                readonly=True
            energy = None; forces = None;
            calc_params['target'] = targ
            calc_forces = True
            calc_dipole = True
            results = snap_wrapper.singlepoint(model,steplabel,
                                snap_calc_params,
                                solvent=solvent,charge=charge,
                                forces=calc_forces,dipole=calc_dipole,
                                continuation=cont,readonly=readonly)
            if calc_forces and calc_dipole:
                energy, forces, dipole, calc = results
            elif calc_forces and not calc_dipole:
                energy, forces, calc = results
            else:
                energy, calc = results
            energy = model.get_potential_energy()
            if isinstance(energy,list) or isinstance(energy,np.ndarray):
                energy = energy[0]
            if len(all_targets[0:])>1:
                 print("SP: ",step,i,targ,energy,model.positions[0])
            outtraj[targ].write(model)
            prev_targ = targ
        chdir(origdir)

    # Return to base directory and close trajectories
    chdir(origdir)
    for targ in all_targets:
        outtraj[targ].close()
    print("# Finished Snapshot MD runs")


# Helper functions

def cycle_step_labels_and_restarts(seed,traj_label,
                                   prevdir,currdir,nextdir,
                                   prevtarg,currtarg,nexttarg,
                                   prevstep,currstep,nextstep,
                                   count,prevcount,exts,use_subdirs=False):
    """
    Move restart files around so that each step continues from the previous one
    """

    # Find name of next restart files
    if (nextstep==count) and (nextdir is not None): # case for final step of non-final part
        next_steplabel = f"{seed}_{targstr(nexttarg)}_{traj_label}_{nextdir}{0:04}"
        next_dir = f"../{nextdir}/"
        if use_subdirs:
            next_dir += "{next_steplabel}/"
    else:
        next_steplabel = f"{seed}_{targstr(nexttarg)}_{traj_label}_{currdir}{nextstep:04}"
        next_dir = f""
        if use_subdirs:
            next_dir += "{next_steplabel}/"
    next_file = []
    for ext in exts:
        next_file.append(f"{next_dir}{next_steplabel}{ext}")

    # Find name of previous step restart files
    if (currstep==0) and (prevdir != currdir): # case for first step of new non-initial part
        prev_steplabel = f"{seed}_{targstr(prevtarg)}_{traj_label}_{prevdir}{prevcount-1:04}"
        prev_dir = f"../{prevdir}/"
        if use_subdirs:
            prev_dir += "{prev_steplabel}/"
    else:
        prev_steplabel = f"{seed}_{targstr(prevtarg)}_{traj_label}_{currdir}{prevstep:04}"
        prev_dir = f""
        if use_subdirs:
            prev_dir += "{prev_steplabel}/"
    prev_file = []
    for ext in exts:
        prev_file.append(f"{prev_dir}{prev_steplabel}{ext}")

    # Find name of current step restart file
    curr_steplabel = f"{seed}_{targstr(currtarg)}_{traj_label}_{currdir}{currstep:04}"
    curr_dir = f""
    if use_subdirs:
        curr_dir += "{curr_steplabel}/"
    curr_file = []
    for ext in exts:
        curr_file.append(f"{curr_dir}{curr_steplabel}{ext}")

    # Skip this step if the next one had previously started
    readonly = False
    if len(next_file)==0:
        return curr_steplabel, False, False
    
    if path.isfile(next_file[0]):
        readonly = True
    # On final overall step, assume if there is an output file for the last step then
    # we must have finished
    if ((path.isfile(f"{curr_steplabel}.nwo") or path.isfile(f"{curr_steplabel}.out")) and
         nextdir==None and currstep==count-1):
        readonly = True

    # If this step has not previously started, copy in restart file
    cont = False
    prev_rst_present = path.isfile(prev_file[0])
    curr_rst_present = path.isfile(curr_file[0])
    next_rst_present = path.isfile(next_file[0])
    if (prev_rst_present and not curr_rst_present):
        if not path.exists(curr_dir) and len(exts)>0 and curr_dir!="":
            makedirs(curr_dir)
        for i,ext in enumerate(exts):
            print(f"Copying from {prev_file[i]} to {curr_file[i]}")
            shutil.copyfile(prev_file[i],curr_file[i])
        cont = True
    elif (prev_rst_present and curr_rst_present and not next_rst_present):
        print(f"Resuming from data in {curr_file[0]}")
        cont = True

    # Debug printing of whether restart files are present
    if False:
        print(prevstep,targstr(prevtarg),prev_file[0],prev_rst_present,'cont=',cont)
        print(currstep,targstr(currtarg),curr_file[0],curr_rst_present,'read=',readonly)
        print(nextstep,targstr(nexttarg),next_file[0],next_rst_present)
    #print(step,curr_rst,path.isfile(curr_rst),cont)
    return curr_steplabel, readonly, cont


# # Obtain a starting geometry

# In[ ]:


def find_initial_geometry(seed,geom_opt_func=None,calc_params={},which_traj=None,ntraj=-1):
    """
    Obtains a suitable initial geometry for the current seed and state.
    Optimises it if not present.
    
    seed: str
        String indicating name of molecule, used to find xyz file
    geom_opt_func: function
        Wrapper function that runs a geometry optimisation
    calc_params: 
        Control-parameters for the wrapper (for QM this is the basis, functional, and target;
        for ML this is the calculator seed, suffix, prefix and target)
    """

    # Construct name for input geometries
    if 'target' in calc_params:
        targ = calc_params['target']
        seed_state_str = f"{seed}_{targstr(targ)}"
    else: 
        seed_state_str = seed

    xyzfile_opt = seed_state_str+'.xyz'
    model_init = None
    curr_dir = getcwd()

    # Look for trajectory of starting points
    if which_traj is not None:
        md_init_traj = f'{seed_state_str}_md_init.traj'
        try:
            traj = Trajectory(md_init_traj)
            print(f'# MD starting positions trajectory {md_init_traj} found: length {len(traj)}')
            import random
            if ntraj==len(traj):
                traj_list = get_trajectory_list(ntraj)
                i = traj_list.index(which_traj)
                print(f'# MD starting positions taken from snapshot {i} based on traj index {i} of {ntraj}==length')
            else:
                random.seed(which_traj)
                i = random.randrange(len(traj))
                print(f'# MD starting positions taken from snapshot {i} based on random seed {which_traj}')
            model_init = traj[i]
            optimised = True
        except:
            print(f'# MD starting positions trajectory {md_init_traj} not found.')

    # Look for existing xyz file with optimised geometry
    if model_init is None:
        try:
            model_init = read(xyzfile_opt)
            print(f"# Optimised geometry found in file: {curr_dir}/{xyzfile_opt}")
            optimised = True
        except:
            print(f"# Optimised geometry not found, no file: {curr_dir}/{xyzfile_opt}")
            xyzfile_unopt = seed+'.xyz'
            print(f"# Reading geometry from file: {curr_dir}/{xyzfile_unopt}")
            model_init = read(xyzfile_unopt)
            optimised = False
    print(f"# Loaded {len(model_init)} atoms")

    # Relax to find optimized geometry for this state if not already done
    model = model_init.copy()
    if not optimised and geom_opt_func is not None:
        print("# Optimising geometry for "+seed)
        energy, forces, positions = geom_opt_func(model,seed,calc_params,'default')
        write(xyzfile_opt,model)
    
    return model


# # Recalculate energies based on an existing input trajectory

# In[ ]:


def recalculate_trajectory(seed,target,traj_label,traj_suffix,input_target,input_suffix,
                           wrapper,calc_params,input_traj_range=None,input_traj_label=None,
                           output_traj_offset=0,charge=0,solvent=None,
                           calc_forces=True,calc_dipole=True,
                           geom_opt_kernel=False,vibfreq_kernel=False):
    """
    Loads snapshots from a trajectory and recalculates the energy and forces with the current settings
    seed: str
        String containing name of molecule and excited state index
    target: int or None
        Excited state index
    traj_label: character or str
        String labelling trajectory (usually A,B,C..)
    """
    
    from glob import glob
    
    if False:
        db_ext = '.db'
    else:
        db_ext = '.gbw'

    # Open input trajectory
    if input_traj_label is None:
        input_traj_label = traj_label
    input_traj = f"{seed}_{targstr(input_target)}_{input_traj_label}_{input_suffix}.traj"
    if not path.isfile(input_traj):
        raise Exception("Input trajectory not found: ",input_traj)
    print(f"# Reading from input trajectory {input_traj}")
    intraj = Trajectory(input_traj)

    if isinstance(target,list):
        all_targets = target
    else:
        all_targets = [target]
    
    if input_traj_range is None:
        input_traj_range = range(0,len(intraj))

    outtraj = {}
    for targ in all_targets:
        # Check if output trajectory already exists
        output_traj = f"{seed}_{targstr(targ)}_{traj_label}_{traj_suffix}.traj"
        if output_traj_offset>0 or (len(input_traj_range)==1 and output_traj_offset==0):
            print(f"# Not writing to output trajectory as range is subset of trajectory.")
            print(f"# Post-process to combine whole trajectory.")
            outtraj[targ] = None
        elif path.isfile(output_traj):
            # Delete it and start afresh
            print(f"# Output trajectory {output_traj} already exists: Overwriting!")
            try:
                remove(output_traj)
            except:
                print('# Unable to remove {output_traj} - mid-air collision between jobs? Continuing')
            outtraj[targ] = Trajectory(output_traj,"w")
        else:
            # Create new file for output trajectory
            print(f"# Opening output trajectory {output_traj} for writing")
            outtraj[targ] = Trajectory(output_traj,"w")
    
    # Create directory for output files and change to it
    wdir = traj_suffix
    origdir = getcwd()
    if not path.exists(wdir):
        try:
            makedirs(wdir)
        except FileExistsError:
            print(f"# Possible mid-air collision between jobs - directory {wdir} exists")
    chdir(wdir)

    # Loop over and recalculate each trajectory point
    for i in input_traj_range:
        frame = intraj[i].copy()
        frame.calc = intraj[i].calc
        iout = i + output_traj_offset
        try:
            energy_in, forces_in = (frame.get_potential_energy(),frame.get_forces())
        except:
            # existing trajectory may have no calculator
            energy_in = 0; forces_in = np.array([[0,0,0]]*len(frame))
        cont = False
        for targ in all_targets:
            label = f"{seed}_{targstr(targ)}_{traj_label}_{traj_suffix}{iout:04}"
            # Default to readonly for multiple-frame trajectories
            readonly = len(input_traj_range)!=1
            # If output files already exist, read them
            if hasattr(wrapper,'get_completed_file_exts'):
                completed_files = [path.isfile(label+f) for f in wrapper.get_completed_file_exts()]
            else:
                completed_files = [False]
            if all(completed_files):
                readonly = True
            elif readonly and any(completed_files):
                print(f'# Some files not present for {label} - cleanup may be required')
                main_outfile = f'{label}{wrapper.get_completed_file_exts()[0]}'
                try:
                    print(f'# Final 30 lines of {main_outfile}:')
                    final_lines = open(main_outfile, "r").readlines()[-30:]
                    for line in final_lines:
                        print(line, end="")
                except:
                    print(f'# Not found: {main_outfile}')
                all_files = glob(f"{label}*")
                print(f'# Removing {label}*: {all_files}')
                for f in all_files:
                    remove(f)
            energy = None;
            forces = None;
            if cont and not readonly:
                cycle_restarts(seed,traj_label,traj_suffix,prevtarg,targ,iout,iout,db_ext)
            calc_params['target'] = targ
            if geom_opt_kernel or vibfreq_kernel:
                from ase.constraints import FixAtoms
                c = FixAtoms(mask=[atom.tag!=1 for atom in frame])
                frame.set_constraint(c)
            if geom_opt_kernel:
                results = wrapper.geom_opt(frame,label,calc_params,
                                        solvent=solvent,charge=charge,
                                        continuation=cont,
                                        readonly=readonly)
                energy, forces, pos = results
                dip = frame.calc.results['dipole']
            if vibfreq_kernel:
                ir = wrapper.freq(frame,label,calc_params,
                                  solvent=solvent,charge=charge,
                                  continuation=cont,readonly=readonly,summary=True)
                #print('getting freqs')
                freqs = ir.get_frequencies()
                modes = ir.modes
                intensities = ir.intensities
                ir.clean()
            # Try to run or read the energy, forces and dipole for this frame
            success = False
            try:
                results = wrapper.singlepoint(frame,label,calc_params,
                                              solvent=solvent,charge=charge,
                                              forces=calc_forces,dipole=calc_dipole,
                                              continuation=cont,readonly=readonly)
                if calc_forces and calc_dipole:
                    energy, forces, dipole, calc = results
                elif calc_forces and not calc_dipole:
                    energy, forces, calc = results
                else:
                    energy, calc = results
                if len(frame) != len(forces) and not(hasattr(energy,"__len__")):
                    print(f'# ERROR: length of frame {i} ({len(frame)}) does not match length of forces array ({len(forces)})')
                    raise Exception('Length matching failure')
                frame.calc = calc
                success = True
            except KeyboardInterrupt:
                # Always exit if Ctrl-C pressed
                raise Exception('Keyboard Interrupt')
            except Exception as e:
                # Any other error should be logged as a FAIL and zeros logged.
                energy = 0.0
                frame.calc.results['dipole'] = np.zeros(3)
                dipole = np.zeros(3)
                forces = np.zeros([len(frame),3])
                print(e)
                print('FAIL: ',label+'.out',iout,targ,len(frame))
                #raise Exception(e)
            finally:
                chdir(origdir)
                chdir(wdir)
            cont = True # After first excitation, assume subsequently can restart
            prevtarg = targ
            # Prepare formatted strings for positions, forces and dipole
            pos = frame.positions
            pos_str = f'[ {pos[0,0]:10.6f} {pos[0,1]:10.6f} {pos[0,2]:10.6f} ]'
            if calc_forces:
                if forces.ndim==3:
                    forces = np.mean(forces,axis=0)
                force_str = f'[ {forces[0,0]:12.9f} {forces[0,1]:12.9f} {forces[0,2]:12.9f} ]'
            else:
                force_str = ''
            if calc_dipole:
                if dipole.ndim==3:
                    dipole = np.mean(dipole,axis=0)[0]
                elif dipole.ndim==2 and dipole.shape[-1]==3:
                    dipole = np.mean(dipole,axis=0)
                dip_str = f'[ {dipole[0]:10.6f} {dipole[1]:10.6f} {dipole[2]:10.6f} ]'
            if isinstance(energy,np.ndarray):
                energy = np.mean(energy)
            freq_str = ''
            # Write line to stdout
            print(f'{iout:04} {targ:2} {energy:16.8f} {len(frame):5} {pos_str} {force_str} {dip_str}')
            # Supply keyword dipole explicitly to ensure it gets written or fails
            if outtraj[targ] is not None:
                if vibfreq_kernel:
                    frame.info = {}
                    frame.info['freqs'] = freqs
                    frame.info['modes'] = modes
                    frame.info['intensities'] = intensities
                if isinstance(frame.calc,list):
                    frame.calc = frame.calc[-1]
                if True: #if success:
                    outtraj[targ].write(frame)

    # Return to base directory and close trajectories
    chdir(origdir)
    for targ in all_targets:
        if outtraj[targ] is not None:
            outtraj[targ].close()


def cycle_restarts(seed,traj_label,traj_suffix,prevtarg,currtarg,prevstep,currstep,db_ext):
    """
    Move db files around so that each step continues from the previous one
    """

    # Find name of previous step file
    prev_steplabel = f"{seed}_{targstr(prevtarg)}_{traj_label}_{traj_suffix}{prevstep:04}"
    prev_dir = f"{prev_steplabel}/"  if False else ""
    prev_db = f"{prev_dir}{prev_steplabel}{db_ext}"

    # Find name of current step restart file
    curr_steplabel = f"{seed}_{targstr(currtarg)}_{traj_label}_{traj_suffix}{currstep:04}"
    curr_dir = f"{curr_steplabel}/" if False else ""
    curr_db = f"{curr_dir}{curr_steplabel}{db_ext}"

    # Copy in db file
    prev_db_present = path.isfile(prev_db)
    curr_db_present = path.isfile(curr_db)
    if (prev_db_present and not curr_db_present):
        if curr_dir!="" and not path.exists(curr_dir) and db_ext is not None:
            makedirs(curr_dir)
        print(f"# Copying from {prev_db} to {curr_db}")
        shutil.copyfile(prev_db,curr_db)


# # Other tools to manage trajectories

# In[ ]:


def get_trajectory_list(ntraj):
    """
    Returns a list of characters to be used as trajectory labels
    Currently ABCDEDFGHIJKLMNOPQRSTUVWXYZ (26)
    then uses AA,AB,...,ZZ (26^2)
    """
    import itertools, string
    extras = []
    if ntraj>26*27:
        raise Exception('get_trajectory_list cannot yet handle more than 26*27 trajectories')
    if ntraj>26:
        extras = [''.join(i) for i in list(itertools.product(string.ascii_uppercase,string.ascii_uppercase))]
    return (list(string.ascii_uppercase)+extras)[0:ntraj]

# Merge trajectories (if generated separately)
def merge_traj(trajnames,trajfile):
    """
    Merges a list of trajectories supplied as a list of filenames,
    and writes the result to another trajectory supplied as a filename
    """

    fulltraj = Trajectory(trajfile,'w')

    for tr in trajnames:
        read_traj = Trajectory(tr)
        for frames in read_traj:
            fulltraj.write(frames)

    print("# Merged ",len(fulltraj)," frames: trajectory written to ",trajfile)

# Difference of two trajectories (generated separately)
def diff_traj(itrajfile,jtrajfile,outtrajfile):
    """
    Takes two trajectory filenames and finds the energy and force difference between
    them, outputting the result to a trajectory name outtrajfile
    """

    outtraj = Trajectory(outtrajfile,'w')
    itraj = Trajectory(itrajfile)
    jtraj = Trajectory(jtrajfile)
    assert(len(itraj)==len(jtraj))

    for i in range(len(itraj)):
        iframe = itraj[i].copy()
        jframe = itraj[j].copy()
        iframe.results["energy"] -= jframe.results["energy"]
        iframe.results["forces"] -= jframe.results["forces"]
        outtraj.write(iframe)

    print("# Took difference of energy and forces. ",len(outtraj)," frames: trajectory written to ",outtrajfile)
    outtraj.close()

def split_traj(input_traj_file,output_trajs=None,ntrajs=None,randomise=False,start=0,end=-1):
    """
    Takes in a trajectory filename, and splits it into sections, randomly or sequentially
    """
    import numpy as np
    if output_trajs is not None:
        if ntrajs is not None:
            if ntrajs!=len(output_trajs):
                raise Exception('Contradictory inputs to split_traj')
        else:
            ntrajs = len(output_trajs)
    if ntrajs is None and output_trajs is None:
        raise Exception('No outputs specified')
    
    if output_trajs is None:
        output_trajs = [input_traj_file.replace('.traj','_')+str(i)+'.traj' for i in range(ntrajs)]
    
    intraj = Trajectory(input_traj_file)
    outtraj = [Trajectory(output_trajs[i],"w") for i in range(ntrajs)]
    total = len(intraj)
    if end > total:
        raise Exception(f'In split_traj, end was {end}, but trajectory was only size {total}')
    if end<0:
        end = max(total - end - 1,0)
    start = max(start,0)
    total = end-start
    if randomise:
        order = np.random.permutation(total)
    for j in range(start,end):
        k = int((j-start)*ntrajs/total)
        jp = j
        if randomise:
            jp = order[j]
        print(end,start,j,k)
        outtraj[k].write(intraj[j])
    for i in range(ntrajs):
        outtraj[i].close()
    
def atom_energies(atom_traj):
    e_at = {}
    for t in atom_traj:
        if (len(t)>1):
            raise Exception(f'This is not a trajectory of single-atom frames: len(t)={len(t)}')
        at = t[0]
        j = at.symbol
        e_at[j] = t.get_potential_energy()
    return e_at
    
def atom_energy(atoms,e_at):
    e = 0
    for at in atoms.get_chemical_symbols():
        e = e + e_at[at]
    return e

def subtract_atom_energies_from_traj(traj,atom_traj,trajout):
    """
    Subtracts the energies associated with isolated atoms from the total energies
    in a trajectory
    """
    e_at = atom_energies(atom_traj)
    for i,t in enumerate(traj):
        try:
            e_raw = t.get_potential_energy()
        except Exception as e:
            print(f'Error while subtracting atoms energies for frame {i}:')
            raise e
        t.calc.results["energy"] = e_raw - atom_energy(t,e_at)        
        trajout.write(t)

# Scatter Plot
def plot_diff(e_x,e_y,rms_fd,xlabel=None,ylabel=None,clabel=None,stats={},plot_file='show',align_axes=True):

    from matplotlib import pyplot

    # Optionally, plot to file or screen
    if plot_file is None:
        return
    # Set up
    fig, ax = pyplot.subplots()
    # Plot data
    im = ax.scatter(e_x,e_y,c=rms_fd)
    cb = fig.colorbar(im)
    cb.set_label(clabel)
    # Plot y=x for comparison
    if align_axes:
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    stat_str=""
    for s in stats:
        stat_str += f'{s}: {stats[s]:10.4f}\n'
    ax.text(0.02,0.98,stat_str,transform=ax.transAxes,
            horizontalalignment='left',verticalalignment='top',)
    if plot_file=='show':
        fig.show()
    else:
        fig.savefig(plot_file)

# Compare two trajectories and plot the difference
def compare_traj_to_traj(trajx,trajy,plot_func=plot_diff,plot_file=None,xlabel=None,ylabel=None,clabel=None):
    """
    Compares two trajectories to each other and calculates statistics for how much they differ in energy and force
    
    trajx: ASE Trajectory
        Trajectory whose energy is plotted along x-axis
    trajy: ASE Trajectory
        Trajectory whose energy is plotted along y-axis
    plot_file:
        Filename to write plot image to.
    xlabel, ylabel: str
        Axis labels for plots.
    """
    import numpy as np

    e_x = []
    e_y = []
    f_x = []
    f_y = []
    d_x = []
    d_y = []
    rms_fd = []
    mae_fd = []
    max_fd = []
    nat = []
    
    if plot_func is None:
        plot_func = plot_diff
    
    for i,framex in enumerate(trajx):
        
        # Read in total energy and forces from trajectories
        e_x.append(framex.get_potential_energy())
        f_x.append(framex.get_forces())
        d_x.append(framex.get_dipole_moment())
        framey = trajy[i]
        e_y.append(framey.get_potential_energy())
        f_y.append(framey.get_forces())
        new_d = framey.get_dipole_moment()
        if new_d.shape==(1,3):
            new_d = new_d[0]
        d_y.append(new_d)
        nat.append(len(framex))

        # Calculate RMS and Max force errors
        rms_fd.append(np.sqrt(np.mean((f_x[-1]-f_y[-1])**2)))
        mae_fd.append(np.mean(np.abs(f_x[-1]-f_y[-1])))
        max_fd.append(np.max(np.sqrt((f_x[-1]-f_y[-1])**2)))

    # Calculate Root Mean Square energy error
    e_x = np.array(e_x); e_y = np.array(e_y); nat = np.array(nat)
    if e_y.ndim==2:
        mean_e_y = np.mean(e_y,axis=1)
        std_e_y = np.std(e_y,axis=1)
    else:
        mean_e_y = e_y
        std_e_y = np.zeros_like(e_y)
    diff = e_x - mean_e_y
    rms_e_err = np.sqrt((1./len(diff))*np.sum(diff**2))
    mae_e_err = np.mean(np.abs(diff))
    max_e_err = np.max(np.abs(diff))
    print(f"# MAE_dE,RMS_dE,MAX_dE {mae_e_err:16.8f} {rms_e_err:16.8f} {max_e_err:16.8f}")

    # Calculate Root Mean Square of RMS force difference
    rms_fd = np.array(rms_fd)
    rms_fd_err = np.sqrt((1./len(rms_fd))*np.sum(rms_fd**2))
    mae_fd_err = np.mean(mae_fd)
    max_fd_err = np.max(np.abs(rms_fd))
    print(f"# MAE_dF,RMS_dF,MAX_dF {mae_fd_err:16.8f} {rms_fd_err:16.8f} {max_fd_err:16.8f}")

    # Calculate Root Mean Square of and MAE of dipole difference
    diff = np.array(d_x)-np.array(d_y)
    rms_d_err = np.sqrt(np.sum((1./len(diff))*(np.sum(diff**2,axis=1))))
    mae_d_err = np.mean(np.sqrt(np.sum(diff**2,axis=1)))
    max_d_err = np.max(np.sqrt(np.sum(diff**2,axis=1)))
    print(f"# MAE_dd,RMS_dd,MAX_dd {mae_d_err:16.8f} {rms_d_err:16.8f} {max_d_err:16.8f}")
    
    # Calculate Mean number of atoms
    mean_nat = np.mean(nat)
    print(f"# Mean Number of atoms: {mean_nat:10.4f}")

    stats = {"MAE dE (eV)  ": mae_e_err,
             "MAE dF (eV/A)": mae_fd_err,
             "MAE dd (e.A) ": mae_d_err,
             "RMS dE (eV)  ": rms_e_err,
             "RMS dF (eV/A)": rms_fd_err,
             "RMS dd (e.A) ": rms_d_err,
             "MAX dE (eV)  ": max_e_err,
             "MAX dF (eV/A)": max_fd_err,
             "MAX dd (e.A) ": max_d_err,
             "Mean No.atoms": mean_nat}
    plot_func(e_x,mean_e_y,rms_fd,xlabel,ylabel,clabel,stats,plot_file)

    return rms_e_err,max_e_err,rms_fd_err,max_fd_err,e_x,e_y,rms_fd

