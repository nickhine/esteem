#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Drivers to run ESTEEM tasks in serial or parallel on a range of inputs.

These inputs will consist of solutes, solvents or pairs of solutes and solvent, depending on the task.
"""


# # Top-level Driver

# In[ ]:


import argparse
from esteem import parallel
from esteem.trajectories import get_trajectory_list

def main(all_solutes,all_solvents,all_solutes_tasks={},all_solvate_tasks={},
         all_clusters_tasks={},all_spectra_tasks={},all_qmd_tasks={},
         all_mltrain_tasks={},all_mltest_tasks={},all_mltraj_tasks={},
         make_script=parallel.make_sbatch):
    """
    Main driver routine which chooses other tasks to run as appropriate.
        
    Control of what driver is actually called depends on command-line arguments with which the script
    was invoked: 'python <seed>.py <task> <seed> <target>'

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents. Entries are the full names.
    all_solutes_tasks: namespace or class
        Argument list for the Solutes task - see solutes_driver documentation below for more detail.
    all_solvate_tasks: namespace or class
        Argument list for the Solvate task - see solvate_driver documentation below for more detail.
    all_clusters_tasks: namespace or class
        Argument list for the Clusters task - see clusters_driver documentation below for more detail.
    all_spectra_tasks: namespace or class
        Argument list for the Spectra task - see spectra_driver documentation below for more detail.
    make_script: dict of strings
        Routine that can write a job submission script. Usually parallel.make_sbatch.
    
    """
    
    import sys
    import time
    from datetime import datetime  
    
    time_stamp = time.time()
    date_time = datetime.fromtimestamp(time_stamp)
    str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    print(f'# ESTEEM version 1.0.0 started at {str_date_time}\n')

    # Parse arguments to script
    parser = argparse.ArgumentParser(description=f'ESTEEM Main driver script')
    parser.add_argument('task')
    parser.add_argument('seed',default=None,type=str)
    parser.add_argument('target',nargs='?',default=None,type=str)
    args = parser.parse_args()
    task = args.task
    scriptname = sys.argv[0].replace(".py","")
    if args.seed is None:
        import sys
        args.seed = scriptname
    seed = args.seed
    target = args.target

    # Special task: set up scripts
    if ('scripts' in task.split()):
        # case where many SolutesTask have been defined
        if isinstance(all_solutes_tasks,dict):
            for targ in all_solutes_tasks:
                if target is not None and targ!=target:
                    continue
                all_solutes_tasks[targ].script_settings['scriptname'] = scriptname
                solutes_script_settings = all_solutes_tasks[targ].script_settings
                make_script(seed=seed,task='solutes',target=targ,**solutes_script_settings)
                make_script(seed=seed,task='solvents',target=targ,**solutes_script_settings)
        else: # just one SolutesTask
            all_solutes_tasks.script_settings['scriptname'] = scriptname
            solutes_script_settings = all_solutes_tasks.script_settings
            make_script(seed=seed,task='solutes',target=target,**solutes_script_settings)
            make_script(seed=seed,task='solvents',target=target,**solutes_script_settings)

        for tasks in [all_solvate_tasks,all_clusters_tasks,all_qmd_tasks,all_spectra_tasks,
                      all_mltrain_tasks,all_mltest_tasks,all_mltraj_tasks]:
            if tasks is None or tasks=={}:
                continue
            if isinstance(tasks,dict):
                for targ in tasks:
                    if target is not None and targ!=target:
                        continue
                    print(tasks[targ].script_settings)
                    tasks[targ].script_settings['scriptname'] = scriptname
                    make_script(seed=seed,task=tasks[targ].task_command,target=targ,**tasks[targ].script_settings)
            else:
                tasks.script_settings['scriptname'] = scriptname
                make_script(seed=seed,task=tasks.task_command,target=target,**tasks.script_settings)

        return

    # Now run actual task
    if ('solutes' in task.split()):
        solutes_task = get_actual_args(all_solutes_tasks,target,'solutes')
        solutes_driver(all_solutes,all_solvents,solutes_task)
    if ('solvents' in task.split()):
        solutes_task = get_actual_args(all_solutes_tasks,target,'solvents')
        solvents_driver(all_solvents,solutes_task)
    if ('solvate' in task.split()):
        solvate_task = get_actual_args(all_solvate_tasks,target,'solvate')
        solvate_task.script_settings['target'] = target
        solvate_task.script_settings['scriptname'] = scriptname
        solvate_driver(all_solutes,all_solvents,seed,solvate_task,make_script)
    if (any('clusters' in t for t in task.split()) or any('mlclus' in t for t in task.split())): 
        clusters_task = get_actual_args(all_clusters_tasks,target,'clusters')
        clusters_task.script_settings['target'] = target
        clusters_task.script_settings['scriptname'] = scriptname
        if any('setup' in t for t in task.split()):
            dryrun = True
        else:
            dryrun = False
        clusters_driver(all_solutes,all_solvents,seed,clusters_task,make_script,dryrun=dryrun)
    if 'atoms' in task.split():
        atomen_task = get_actual_args(all_qmd_tasks,target,'qmd')
        atomen_task.seed = seed
        atomen_task.target = target
        atom_energies_driver(atomen_task)
    if 'qmd' in task.split():
        qmd_task = get_actual_args(all_qmd_tasks,target,'qmd')
        qmd_task.seed = seed
        qmd_driver(qmd_task,all_solutes,all_solvents)
    if ('train' in task.split()) or ('mltrain' in task.split()):
        mltrain_task = get_actual_args(all_mltrain_tasks,target,'mltrain')
        mltrain_task.seed = seed
        mltrain_driver(mltrain_task,all_solutes,all_solvents)
    if ('test' in task.split()) or ('mltest' in task.split()):
        mltest_task = get_actual_args(all_mltest_tasks,target,'mltest')
        mltest_task.seed = seed
        mltest_driver(mltest_task,all_solutes,all_solvents)
    if ('traj' in task):
        cleanup_only = True if 'cleanup' in task else False
        mltraj_task = get_actual_args(all_mltraj_tasks,target,'mltraj')
        mltraj_task.seed = seed
        mltraj_driver(mltraj_task,all_solutes,all_solvents,cleanup_only=cleanup_only)
    if ('spectra' in task.split()):
        spectra_task = get_actual_args(all_spectra_tasks,target,'spectra')
        #warp_params = spectral_warp_driver(all_solutes,all_solvents,spectra_task)
        spectra_task.seed = seed
        spectra_driver(all_solutes,all_solvents,spectra_task,warp_params=None)

    time_stamp_end = time.time()
    date_time = datetime.fromtimestamp(time_stamp_end)
    str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    print(f'\n# ESTEEM finished at {str_date_time} ({time_stamp_end-time_stamp:12.3f} seconds elapsed)')

# Helper function that extracts the appropriate element from a dictionary    
def get_actual_args(all_args,target,task):
    if isinstance(all_args,dict):
        try:
            args = all_args[target]
        except KeyError as e:
            print(f"all_{task}_args[{target}] not found. Possible values:")
            all_keys = list(all_args.keys())
            all_keys.sort()

            # First take "which_traj" off the end of they key, if it is present
            many_trajs = get_trajectory_list(26*27)
            prev_stem = ""
            key_suffix = ""
            for key in all_keys:
                key_parts = key.split('_')
                print(key,key_parts)
                prev_key_suffix = key_suffix
                if key_parts[-1] in many_trajs:
                    key_stem = '_'.join(key_parts[0:-1])
                    key_suffix = key_parts[-1]
                else:
                    key_stem = key
                    key_suffix = ""
                if key_stem!=prev_stem:
                    if prev_key_suffix!="":
                        print("], ",end="")
                    print(key_stem,end="")
                    if key_suffix!="":
                        print("_[",end="")
                    prev_stem = key_stem
                print(key_suffix+",",end="")
            if prev_key_suffix!="":
                print("]\n")
            else:
                print("\n")
            raise e
            
    else:
        args = all_args
    return args


# In[ ]:





# # Run calculations on individual Solute molecules

# In[ ]:


"""
Drivers to run ESTEEM tasks in serial or parallel on a range of inputs
These inputs will consist of solutes, solvents or pairs of solutes and solvent, depending on the task
"""

from esteem.tasks import solutes
from esteem import parallel
from os import path,chdir,makedirs
from shutil import copytree
  
def solutes_driver(all_solutes,all_solvents,task):
    """
    Driver to run a range of DFT/TDDFT calculations on a range of solutes.
    If the script is run as an array task, it performs the task for the
    requested solute only.

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents (implicit only, here). Entries are the full names.
    task: SolutesTask class
        Argument list for the whole Solutes job - see Solutes module documentation for more detail.
        
        Arguments used only within the driver routine include:

        ``task.directory``: Directory prefix for where output of this particular 'target' of the Solutes
        calculation will be written
    """

    base_path = path.basename(getcwd())
    if task.directory is not None and base_path != task.directory:
        if not path.exists(task.directory):
            print(f'Creating directory {task.directory}')
            try:
                makedirs(task.directory)
            except FileExistsError:
                print(f"# Possible mid-air collision between jobs - directory {task.directory} exists")
        if not path.exists(task.directory+'/xyz'):
            if path.exists('xyz'):
                copytree('xyz', task.directory+'/xyz')
        chdir(task.directory)

    # Determine if we are in a job array, in which case just run one pair
    task_id = parallel.get_array_task_id()
    if task_id is None:
        solutes_to_run = all_solutes.copy() # run everything
    else:
        solute = list(all_solutes)[task_id]
        solutes_to_run = {solute: all_solutes[solute]} # run just one solute

    # Geom opts and TDDFT for solutes, then implicit solvent versions in each solvent
    if not all_solvents:
        all_solvents_to_run = [None]
    else:
        all_solvents_to_run = all_solvents.copy()

    for solvent in all_solvents_to_run:
        task.solvent = solvent
        if task.solvent_settings is not None and task.solvent is not None:
            task.solvent = task.solvent_settings
            task.solvent['solvent'] = solvent
        # if the solute is also a solvent, do not run this solute in any other solvents
        solute_namelist = task.get_xyz_files(solutes_to_run,"xyz")
        solute_namelist = [solu2 for solu2 in solute_namelist if (solu2==solvent or solu2 not in all_solvents)]
        task.run(solute_namelist)
    if task.directory is not None:
        chdir("..")


# # Run calculations on individual Solvent molecules

# In[ ]:


from esteem.tasks import solutes

def solvents_driver(all_solvents,task):
    """
    Driver to run a range of DFT/TDDFT calculations on a range of solvent molecules.
    See the Solutes module documentation for more info on what tasks are run.
    
    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents. Entries are the full names.
    task: SolutesTask class
        Argument list for the whole Solutes job - see Solutes module documentation for more detail.
        
        Arguments used only within the driver routine include:

        ``task.directory``: Directory prefix for where output of this particular 'target' of the Solutes
        calculation will be written

    wrapper: namespace or class
        Functions that will be used in the Solutes task - see Solutes module documentation for more detail.
    """

    if task.directory is not None:
        if not path.exists(task.directory):
            print(f'# Creating directory {task.directory}')
            try:
                makedirs(task.directory)
            except FileExistsError:
                print(f"# Possible mid-air collision between jobs - directory {task.directory} exists")
        if not path.exists(task.directory+'/xyz'):
            print(f'# Copying xyz directory from current path to {task.directory}')
            if path.exists('xyz'):
                copytree('xyz', task.directory+'/xyz')
        print(f'# Changing directory to {task.directory}')
        chdir(task.directory)

    solvent_namelist = task.get_xyz_files(all_solvents,"xyz")

    # Geom opts and TDDFT for solvents (in implicit solvent of self)
    for solvent in all_solvents:
        task.solvent = solvent
        if task.solvent_settings is not None:
            task.solvent = task.solvent_settings
            task.solvent['solvent'] = solvent
        task.run([solvent])

    if task.directory is not None:
        print(f'# Returning to parent directory')
        chdir("..")


# # Calculate Range Separation Parameters

# In[ ]:


# Temporarily here till moved elsewhere
def range_sep_driver(all_solutes,all_solvents,task,wrapper):
    import os
    from ase.units import Hartree as Ha
    
    if task.directory is not None:
        if not path.exists(task.directory):
            raise Exception(f"# Directory not present: {task.directory} (must run solutes task first)")
        chdir(task.directory)

    #from esteem.tasks.solutes import find_range_sep

    solutes = ["AQS"]
    solvent = {'solvent': 'watr', 'cosmo_smd': True, 'cosmo_vem': 0}

    func = "LC-PBE0"
    basis = "6-311++G**"
    charges={"AH2QS": -1, "AQS": -1}
    rs_range = [0.05,0.1,0.15,0.2,0.25,0.3]
    energy,calc,evals,occs=find_range_sep(solutes,"is_opt_watr","rs_opt",nw,basis=basis,
                                          func=func,solvent=solvent,charges=charges,
                                          rs_range=rs_range,all_readonly=True)
    ehomo = {}
    elumo = {}
    seed = solutes[0]
    all_rs = []
    all_Ja = []
    all_Jb = []
    all_J2 = []
    for rsf in rs_range:
        rs = f'{rsf:.2f}'
        if seed in charges:
            charge = charges[seed]
        else:
            charge = 0

        for c in [charge+1,charge-1,charge]:
            # Assume when c==charge, system is closed-shell
            if c != charge:
                occfac = 0.99
                s = 0.5
            else:
                occfac = 1.99
                s = 0
            
            ih = max(np.argwhere(occs[rs,c]>occfac))[0]
            il = min(np.argwhere(occs[rs,c]<occfac))[0]
            ehomo[rs,c] = (evals[rs,c])[ih]
            elumo[rs,c] = (evals[rs,c])[il]
            #except:
            #    print('fail for rs=',rs)
        
        all_rs.append(float(rs))
        IP = energy[rs,charge+1] - energy[rs,charge]
        EH = ehomo[rs,charge]
        all_Ja.append((EH+IP)**2)
        
        IPm = energy[rs,charge] - energy[rs,charge-1]
        EHm = ehomo[rs,charge-1]
        print(f'E_H-IP for rs={rs},Q={charge} is {EH:0.3f}+{IP:0.3f} = {EH+IP:0.3f}',
              f', Q={charge-1} is {EHm:0.3f}+{IPm:0.3f} = {EHm+IPm:0.3f}')
        all_Jb.append((EHm+IPm)**2)
        all_J2.append((EHm+IPm)**2+all_Ja[-1])
        #print(f'-IP for rs={rs},Q={charge} is {-IP}')
    import matplotlib.pyplot as plt
    plt.plot(all_rs,all_Ja)
    #plt.plot(all_rs,all_Jb)
    #plt.plot(all_rs,all_J2)
    plt.show()

    if task.directory is not None:
        chdir("..")


# # Run MD calculations on all pairs of solvent + solute

# In[ ]:


from esteem.tasks import solvate
from shutil import copyfile
import itertools
from os import path,chdir,makedirs

def solvate_driver(all_solutes,all_solvents,seed,task,make_sbatch=None):
    """
    Driver to run set up and run MD on solvated boxes containing a solute molecule and solvent.
    If the script is run as an array task, it performs the task for the requested solute/solute pair
    only, out of the nsolutes * nsolvents possible combinations.

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents. Entries are the full names.
    task: SolvateTask class
        Argument list for the whole clusters job - see Solvate module documentation for more detail.
        
        Arguments used within this routine include:
        
        ``task.md_suffix``: Directory suffix for where the MD runs will take place
        
        ``task.md_geom_prefix``: Directory prefix for where the geometries from the Solutes calculation should be obtained.

    wrapper: class
        Wrapper that will be used in the Solvate task - see Solvate module documentation for more detail.
    """

    md_suffix = task.md_suffix

    # Make list of all combinations of solvent and solute
    all_pairs = list(itertools.product(all_solvents.items(),all_solutes.items()))
    for i,((solvent,fullsolvent),(solute,fullsolute)) in enumerate(all_pairs):
        if solute in all_solvents and (solute!=solvent):
            del all_pairs[i]
    
    all_paths = [f'{solute}_{solvent}_{md_suffix}' for (solvent,_),(solute,_) in all_pairs]
    base_path = path.basename(getcwd())

    # Determine if we are in a job array, in which case just run one pair
    task_id = parallel.get_array_task_id()
    if task_id is not None:
        all_pairs = [all_pairs[task_id]]
    
    # Iterate over pairs of solvent and solute
    for i,((solvent,fullsolvent),(solute,fullsolute)) in enumerate(all_pairs):
        
        sol_sol_str = f'{solute} ({fullsolute}) in {solvent} ({fullsolvent})'
        md_path = f'{solute}_{solvent}_{md_suffix}'

        if solute in all_solvents and (solute!=solvent):
            continue

        # If we are in the parent directory of this project, or a subdirectory with
        # appropriate symlinks, setup clusters subdirectories if not already present
        if base_path not in all_paths:
            task_str = f'# Task {i} of {len(all_pairs)}'
            print(f'{task_str}: Setting up solvate calculation for {sol_sol_str}\n')

            # Set up files and directories for MD run
            if not path.exists(md_path):
                print(f'# Creating directory {md_path}')
                try:
                    makedirs(md_path)
                except FileExistsError:
                    print(f"# Possible mid-air collision between jobs - directory {md_path} exists")
                
            # See if a geometry file for a "complex" exists, or the solvent and/or solute geometries
            md_geom_prefix = task.md_geom_prefix
            if solvent is not None:
                md_geom_prefix = md_geom_prefix + '/is_opt_{solv}'
            else:
                md_geom_prefix = md_geom_prefix + '/opt'
            md_geom_prefix = sub_solu_solv_names(md_geom_prefix,f'{solute}_{solvent}',
                                                 all_solutes,all_solvents)
            # Copy in optimised geometry of solute in implicit solvent
            try:
                infile = f'{md_geom_prefix}/{solute}.xyz'
                outfile = f'{md_path}/{solute}.xyz'
                copyfile(infile,outfile)
            except FileNotFoundError:
                print(f'# Optimised geometry for {solute} in {solvent} not found')
                print(f'# Expected: {infile}')
                continue

            # Copy in optimised geometry of solvent in implicit solvent
            try:
                infile = f'{md_geom_prefix}/{solvent}.xyz'
                outfile = f'{md_path}/{solvent}.xyz'
                copyfile(infile,outfile)
            except FileNotFoundError:
                print(f'# Optimised geometry for {solvent} not found')
                print(f'# Expected: {infile}')
                continue

            print(f'# Changing directory to {md_path}')
            chdir(md_path)
            
            # Write job script for submission to HPC cluster
            if make_sbatch is not None:
                task.script_settings['jobname'] = f'{solute}_{solvent}_{md_suffix}'
                task.script_settings['execpath'] = '../'
                make_sbatch(seed=seed,task='solvate',**task.script_settings)

            # Go back to base directory
            print(f'# Returning to parent directory\n\n')
            chdir('..')

        # We are in a subdirectory already so run a particular calculation
        elif base_path==md_path:
            # Now run MD in md_path
            task.solute = solute
            task.solvent = solvent
            # If we provided a dictionary for the boxsize, find the specific entry that we need
            if isinstance(task.boxsize,dict):
                if solvent in task.boxsize:
                    task.boxsize = task.boxsize[solvent]
                else:
                    raise Exception(f"task.boxsize is a dictionary but contains no entry for solvent '{solvent}'")
            task.setup_amber()
            task.run()


# # Run Clusters script on MD snapshots to extract region near solute

# In[ ]:


from shutil import copyfile
import itertools
from os import path,chdir,makedirs,getcwd,symlink

def clusters_driver(all_solutes,all_solvents,seed,task,make_sbatch=None,dryrun=False):
    """
    Driver to extract isolated clusters from solvated models, for a range of solute/solvent pairs.
    
    Takes MD results from the directory {task.md_prefix} and performs excitation
    calculation in the directory {solute}_{solvent}_{task.exc_suffix}.
    
    If invoked from the base directory, rather than the excitation directory, it sets up the
    excitation directory and writes a job script then exits.
    
    If invoked from the excitation directory, it performs the excitation calculations for all
    the extracted clusters. If the script is run as an array task, it performs the task for the
    requested cluster only.
    
    *Arguments*

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents. Entries are the full names.
    seed: str
        Overall 'seed' name for the run - used in creation of job scripts for the calculation
    task: ClustersTask class
        Argument list for the whole clusters job - see Clusters module documentation for more detail.
        
        Arguments used predominantly in the driver rather than the task main routine include:
        
        ``task.md_prefix``: Directory where MD outputs are to be found.

        ``task.md_suffix``: Suffix of MD output trajectories.
        
        ``task.exc_suffix``: Directory where results of Cluster excitation calculations will go.
    make_sbatch: function
        Function that writes a job submission script for the clusters jobs
        
    *Output*:
    
        On first run, from the base directory of the project, the script will create subdirectories
        for each solute-solvent pair, with path '{solute}_{solvent}_{exc_suffix}'. The default value
        of ``exc_suffix`` is 'exc'.
        
        To each directory, this routine will copy the trajectory file 
        '{md_prefix}/{solute}_{solvent}_{md_suffix}.traj'
        which consists of ``task.nsnaps`` snapshots.
        
        It will then create a job script using ``make_sbatch`` and ``settings`` with the
        name '{solute}_{solvent}_{exc_suffix}_sub' and exit.
        
        The user then needs to run the individual job scripts from each subdirectory, probably
        as a job array: the array task ID should range from 0 to ``task.nsnaps - 1``
        
        The output of those runs will be the excited state energies of the clusters, which
        can be averaged over in the Spectra task.
    """
    
    # Iterate over all pairs of solute and solvent, unless in a specific directory already,
    # in which case just run that pair
    
    # First take "which_traj" off the end of exc_suffix, if it is present
    if hasattr(task,'exc_dir_suffix'):
        exc_dir_suffix = task.exc_dir_suffix
    else:
        exc_dir_suffix = task.exc_suffix
    exc_parts = exc_dir_suffix.split('_')
    if exc_parts[-1] == task.which_traj:
        exc_dir_suffix = '_'.join(exc_parts[0:-1])
    else:
        exc_dir_suffix = exc_dir_suffix
    all_pairs = list(itertools.product(all_solvents.items(),all_solutes.items()))
    all_paths = [f'{solute}_{solvent}_{exc_dir_suffix}' for (solvent,_),(solute,_) in all_pairs]
    from re import sub
    all_paths = [sub("_None","",p) for p in all_paths]
    base_path = path.basename(getcwd())
    orig_path = getcwd()
    for i,((solvent,fullsolvent),(solute,fullsolute)) in enumerate(all_pairs):
        sol_sol_str = f'{solute} ({fullsolute}) in {solvent} ({fullsolvent})'
        if solvent is None:
            sol_sol_str = f'{solute} ({fullsolute})'
        clus_path = f'{solute}_{solvent}_{exc_dir_suffix}' if solvent is not None else f'{solute}_{exc_dir_suffix}'
        md_path = sub_solu_solv_names(task.md_prefix,f'{solute}_{solvent}',all_solutes,all_solvents)

        if solute in all_solvents and (solute!=solvent):
            continue
        
        # We are in the parent directory of this project, so setup clusters
        # subdirectories if not already present 
        if base_path not in all_paths:
            # Check that MD has finished - skip to next file if not
            if not isinstance(task.md_suffix,list):
                md_suffix = [task.md_suffix]
            else:
                md_suffix = task.md_suffix

            trajfile = {}
            if not path.exists(md_path):
                print(f'\n# Skipping Clusters for: {sol_sol_str} - MD directory not found')
                print(f'# Expected: {md_path}')
                continue
            else:
                for mds in md_suffix:
                    trajfile[mds] = f'{md_path}/{solute}_{solvent}_{mds}.traj'
                    if solvent is None:
                        trajfile[mds] = f'{md_path}/{solute}_{mds}.traj'
                if not all([path.exists(trajfile[mds]) for mds in trajfile]):
                    print(f'\n# Skipping Clusters for: {sol_sol_str} - trajectory file(s) not found')
                    for trajf in trajfile:
                        if not path.exists(trajfile[trajf]):
                            print(f'Not found: {trajfile[trajf]}')
                    continue

            print(f'\nTask {i} of {len(all_pairs)}: Setting up clusters calculation for {sol_sol_str}')
            if not path.exists(clus_path):
                print(f'# Creating directory {clus_path}')
                try:
                    makedirs(clus_path)
                except FileExistsError:
                    print(f"# Possible mid-air collision between jobs - directory {clus_path} exists")

            # Create symlinks to trajectory files in excitations directory
            for mds in md_suffix:
                trajlink = f'{solute}_{solvent}_{mds}.traj'
                if solvent is None:
                    trajlink = f'{solute}_{mds}.traj'
                if not (path.isfile(f'{clus_path}/{trajlink}') or path.islink(f'{clus_path}/{trajlink}')):
                    chdir(clus_path)
                    print(f'# Creating link from ../{trajfile[mds]} to {trajlink}')
                    symlink('../'+trajfile[mds],trajlink)
                    chdir(orig_path)

            # Copy the geometries of the solute and solvent to the excitations directory
            # so they can be used in cluster extraction
            try:
                copyfile(f'{md_path}/{solute}.xyz',f'{clus_path}/{solute}.xyz')
            except FileNotFoundError:
                print(f'# Geometry file {md_path}/{solute}.xyz not found')
            except PermissionError:
                print(f'# Could not write to {clus_path}/{solute}.xyz (PermissionError)')
            if solvent is not None:
                try:
                    copyfile(f'{md_path}/{solvent}.xyz',f'{clus_path}/{solvent}.xyz')
                except FileNotFoundError:
                    print(f'# Geometry file {md_path}/{solvent}.xyz not found')
                except PermissionError:
                    print(f'# Could not write to {clus_path}/{solvent}.xyz (PermissionError)')

            print(f'# Changing directory to {clus_path}')
            chdir(clus_path)
            
            # Write job script for submission to HPC cluster
            if make_sbatch is not None:
                whichstr = f'_{task.which_traj}' if task.which_traj is not None else ''
                task.script_settings['jobname'] = f'{solute}_{solvent}_{task.exc_suffix}{whichstr}'
                if solvent is None:
                    task.script_settings['jobname'] = f'{solute}_{task.exc_suffix}{whichstr}'
                task.script_settings['execpath'] = '../'
                seed = f'{solute}_{solvent}' if solvent is not None else solute
                make_sbatch(seed=seed,task='clusters',**task.script_settings)

            # Go back to base directory
            chdir(orig_path)

        # We are in a subdirectory already so run a particular calculation
        elif clus_path==base_path:
            whichstr = f' for trajectory {task.which_traj}' if task.which_traj is not None else ''
            print(f'\n# Processing clusters for {sol_sol_str}{whichstr}')
            # Prepare Cluster excitation calculation
            task.solute = solute
            task.solvent = solvent
            if task.solute == task.solvent:
                reset_roots = False
                if type(task.target)!=list:
                    if task.nroots > task.target:
                        reset_roots = True
                else:
                    if task.nroots > min(task.target):
                        task.target = min(task.target)
                        reset_roots = True
                if reset_roots:
                    print(f'\n# Warning: solute==solvent, so assuming excitations are not required. Setting nroots=0')
                    task.nroots = 0
            if isinstance(task.impsolv,dict):
                task.impsolv['solvent'] = solvent
            else:
                task.impsolv = solvent
            # If we provided a dictionary for the radius, find the specific entry that we need
            if isinstance(task.radius,dict):
                if solvent in task.radius:
                    task.radius = task.radius[solvent]
                else:
                    raise Exception(f"task.radius is a dictionary but contains no entry for solvent '{solvent}'")
            if hasattr(task,'calc_seed'):
                if task.calc_seed is not None:
                    seed = f'{solute}_{solvent}' if solvent is not None else solute
                    task.calc_seed = sub_solu_solv_names(task.calc_seed,seed,all_solutes,all_solvents)
            task.task_id = parallel.get_array_task_id()
            # Finally, actually run the task
            task.run(dryrun=dryrun)


# In[1]:


def get_solu_solv_names(seed):
    # Guess solute and solvent names from seed name
    solu = seed.split("_")[0]
    try:
        solv = seed.split("_")[1]
    except:
        solv = "NO_SOLVENT_FOUND"
    return solu,solv

def sub_solu_solv_names(string,seed,all_solutes,all_solvents):
 
    solu,solv = get_solu_solv_names(seed)
    string_sub = string
    if '{seed}' in string_sub:
        string_sub = string_sub.replace("{seed}",seed)
    if '{solu}' in string_sub:
        if solu in all_solutes:
            string_sub = string_sub.replace("{solu}",solu)
    if '{solv}' in string_sub:
        if solv in all_solvents:
            string_sub = string_sub.replace("{solv}",solv)
    return string_sub

def merge_xyzs(solu,solv,seed_xyz,merge_path='.'):
    from os import path
    from ase.io import read,write

    solu_xyz = f'{merge_path}/{solu}.xyz'
    solv_xyz = f'{merge_path}/{solv}.xyz'
    print(f'# Merging {solu_xyz} and {solv_xyz}, writing to {seed_xyz}')
    if ((path.isfile(solu_xyz) or path.islink(solu_xyz)) and
        (path.isfile(solv_xyz) or path.islink(solv_xyz))):
        solu_atoms = read(solu_xyz)
        solv_atoms = read(solv_xyz)
        solv_atoms.translate([20.0,0.0,0.0])
        solu_solv_atoms = solu_atoms + solv_atoms
        write(seed_xyz,solu_solv_atoms)
        return
    else:
        print(f'# Source files {solu}.xyz and {solv}.xyz not found in {merge_path}')
        return

def atom_energies_driver(atomen):

    from ase.io import read, Trajectory
    from ase import Atoms
    from esteem.tasks.clusters import get_ref_mol_energy

    # Determine if we are in a job array
    task_id = parallel.get_array_task_id()
    if task_id is None:
        task_id = 0
    
    # Guess solute and solvent names from seed name
    solu,solv = get_solu_solv_names(atomen.seed)
  
    base_path = path.basename(getcwd())
        
    wrapper_is_ml = False
    if wrapper_is_ml:
        calc_params = {'calc_seed': atomen.seed,
                       'calc_suffix': atomen.calc_suffix,
                       'calc_dir_suffix': atomen.calc_dir_suffix,
                       'calc_prefix': '',
                       'target': atomen.target}
    else:
        # QM Wrapper
        calc_params = {'basis':atomen.basis,'func':atomen.func,'target':None,'disp':atomen.disp}  # TODO

    orig_dir = getcwd()
    if solv=="NO_SOLVENT_FOUND":
        solv = None
    ref_mol = solv if solv is not None else solu
    if hasattr(atomen,'ref_mol_dir'):
        ref_mol_dir = atomen.ref_mol_dir
    else:
        ref_mol_dir = 'PBE0'
    if hasattr(atomen,'ref_mol_xyz'):
        ref_mol_xyz = f'{ref_mol_dir}/{atomen.ref_mol_xyz}'
    else:
        if solv is not None:
            ref_mol_xyz = f'{ref_mol_dir}/is_opt_{solv}/{ref_mol}.xyz'
        else:
            ref_mol_xyz = f'{ref_mol_dir}/opt/{ref_mol}.xyz'
    ref_mol_energy, ref_mol_model = get_ref_mol_energy(atomen.wrapper,ref_mol,
                                                       solv,calc_params,ref_mol_xyz,ref_mol_dir,
                                                       silent=False)
    print(f'# Reference molecule energy = {ref_mol_energy}')

    # Prepare directory for calculation
    atoms_dir = f'{atomen.seed}_atoms_{atomen.target}'
    atoms_xyz = f'{atomen.seed}.xyz'
    if base_path != atoms_dir:
        if not path.exists(atoms_dir):
            print(f'# Creating directory {atoms_dir}')
            try:
                makedirs(atoms_dir)
            except FileExistsError:
                print(f"# Possible mid-air collision between jobs - directory {atoms_dir} exists")
        if not path.exists('atoms_dir/{solu}.xyz'):
            if solv is not None:
                copyfile(f'{ref_mol_dir}/is_opt_{solv}/{solu}.xyz',f'{atoms_dir}/{solu}.xyz')
            else:
                copyfile(f'{ref_mol_dir}/opt/{solu}.xyz',f'{atoms_dir}/{solu}.xyz')
        if not path.exists('atoms_dir/{solv}.xyz'):
            if solv is not None:
                copyfile(f'{ref_mol_dir}/is_opt_{solv}/{solv}.xyz',f'{atoms_dir}/{solv}.xyz')
        if not path.isfile(f'{atoms_dir}/{atoms_xyz}') and not path.islink(f'{atoms_dir}/{atoms_xyz}'):
            print(f'# Source file {atoms_xyz} not found - trying to combine {solu} and {solv} xyzs')
            merge_xyzs(solu,solv,f'{atoms_dir}/{atoms_xyz}',atoms_dir)
        if not path.isfile(f'{atoms_dir}/{atoms_xyz}'):
            try:
                copyfile(atoms_xyz,f'{atoms_dir}/{atoms_xyz}')
                print(f'# Source file for atoms {atoms_xyz} copied to {atoms_dir}')
            except PermissionError:
                print(f'# Could not copy source file for atoms {atoms_xyz} to {atoms_dir} - permission error')
        chdir(atoms_dir)

    model_opt = read(atoms_xyz)
    atom_list = set(model_opt.symbols)
    atom_energies = {}
    atom_traj_file = f'../{atomen.seed}_atoms_{atomen.target}.traj'
    atom_traj = Trajectory(atom_traj_file,'w')
    atom_model = {}

    for atom in atom_list:
        atom_model[atom] = Atoms(atom)
        spin=0
        if atom=='H':
            spin=0.5
        if atom=='O':
            spin=1
        if atom=='N':
            spin=0.5
        if atom=='C':
            spin=0
        if wrapper_is_ml:
            atom_energies[atom],forces,dipole,calc = atomen.wrapper.singlepoint(atom_model[atom],
                f'{atomen.seed}',calc_params,solvent=solv,dipole=True,forces=True,)
        else:
            atom_energies[atom],forces,dipole,calc = atomen.wrapper.singlepoint(atom_model[atom],
                f'{atomen.seed}_{atom}',calc_params,solvent=solv,dipole=True,forces=True,spin=spin)
    
    # Rescale atom energies
    sum_ni_Ei = 0
    for i in atom_energies:
        ni = ref_mol_model.symbols.count(i)
        sum_ni_Ei += atom_energies[i]*ni
    E_mol = ref_mol_energy

    alpha = E_mol/(sum_ni_Ei)
    print(f'# Sum of atomic energies for reference model: {sum_ni_Ei} \nRescaling by: {alpha}')
    for atom in atom_list:
        atom_energies[atom] *= alpha
        atom_model[atom].calc.results['energy'] = atom_energies[atom]
        atom_traj.write(atom_model[atom])
        print(f'atom_energies[{atom}] = {atom_energies[atom]}')
    atom_traj.close()


def qmd_driver(qmdtraj,all_solutes,all_solvents):
    """
    Driver to run calculations to generate a range of Quantum Molecular
    Dynamics trajectories.
    If the script is run as an array task, it performs the task for the
    requested trajectory only.

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents (implicit only, here). Entries are the full names.
    task: QMDTrajTask class
        Argument list for the whole QMD_Trajectories job - see QMD_Trajectories module documentation for more detail.
        
        Arguments used only within the driver routine include:

        ``task.directory``: Directory prefix for where output of this particular 'target' of the Solutes
        calculation will be written
        
        Arguments for which the driver routine will perform solute & solvent name substitution:

        ``task.solvent``: 
    """
    
    import string
    from esteem.trajectories import get_trajectory_list, targstr

    # Substitute names into geom_prefix if required
    qmdtraj.geom_prefix = sub_solu_solv_names(qmdtraj.geom_prefix,qmdtraj.seed,
                                              all_solutes,all_solvents)

    base_path = path.basename(getcwd())
    qmd_dir = f"{qmdtraj.seed}_qmd"
    seed_xyz = f'{qmdtraj.seed}.xyz'
    if base_path != qmd_dir:
        if not path.exists(qmd_dir):
            print(f'# Creating directory {qmd_dir}')
            try:
                makedirs(qmd_dir)
            except FileExistsError:
                print(f"# Possible mid-air collision between jobs - directory {qmd_dir} exists")
        if not path.isfile(f'{qmd_dir}/{seed_xyz}'):
            copyfile(f'{qmdtraj.geom_prefix}/{seed_xyz}',f'{qmd_dir}/{seed_xyz}')
            print(f'# Source file {seed_xyz} copied to {qmd_dir}')
        chdir(qmd_dir)

    # Substitute names into solvent if required
    if qmdtraj.solvent is not None:
        qmdtraj.solvent = sub_solu_solv_names(qmdtraj.solvent,qmdtraj.seed,
                                              all_solutes,all_solvents)

    # Make list of trajectories
    all_trajs = get_trajectory_list(qmdtraj.ntraj)

    # Determine if we are in a job array, in which case just run one trajectory
    task_id = parallel.get_array_task_id()
    if isinstance(qmdtraj.which_trajs,str):
        raise Exception("# which_trajs must be a list, not a string")
    if qmdtraj.which_trajs is None:
        qmdtraj.which_trajs = all_trajs
    trajs_in_task = qmdtraj.which_trajs
    if task_id is not None:
        if all_trajs[task_id] in trajs_in_task:
            qmdtraj.which_trajs = [all_trajs[task_id]]
        else:
            raise Exception(f"# Requested trajectory {all_trajs[task_id]} is not part of this task.")

    qmdtraj.run()
    
def make_traj_links(mltrain_task,traj_links,train_dir,prefix,all_solutes,all_solvents):
    
    from ase.io import Trajectory
    
    origdir = getcwd()
    # switch to the training directory, so links are made there
    chdir(train_dir)

    # obtain a list of the trajectories to be used in the training
    which_trajs,trajnames = mltrain_task.get_trajnames(prefix)
    trajnames = dict(zip(which_trajs,trajnames))
    if prefix == "":
        print(f'# Creating symlinks to training trajectories')
    elif prefix == "valid":
        print(f'# Creating symlinks to validation trajectories')
    elif prefix == "test":
        print(f'# Creating symlinks to testing trajectories')
    for traj in which_trajs:
        if traj not in traj_links:
            raise Exception(f'# Invalid trajectory in traj_links: {traj}. Expected: {traj_links}')

        # If there are multiple links that must be set up to assemble this trajectory,
        # the user may supply a list of links. Otherwise, we make a list with just
        # the single entry provided
        if isinstance(traj_links[traj],list):
            traj_links_list = traj_links[traj]
        else:
            traj_links_list = [traj_links[traj]]
        # Loop over the list made above
        for traj_link in traj_links_list:
            # Check if the traj_link string contains "{solu}" or "{solv}", in
            # which case replace these with the appropriate solvent or solute string
            traj_link = sub_solu_solv_names(traj_link,mltrain_task.seed,all_solutes,all_solvents)

            # Find the length of the trajectory and print it along with the proposed link
            try:
                t = Trajectory('../'+traj_link); l = len(t); t.close()
            except:
                l = 0
            print('#',traj_link,'<->',trajnames[traj],' len=',l)

            # Check the link destination exists, and if so make the link
            if not path.isfile('../'+traj_link) and not path.islink('../'+traj_link):
                raise Exception(f'# File to link to not found for trajectory {traj}: {traj_link}')
            if not path.isfile('../'+trajnames[traj]) and not path.islink('../'+trajnames[traj]):
                symlink('../'+traj_link,'../'+trajnames[traj])
    chdir(origdir)

def mltrain_driver(mltrain_task,all_solutes={},all_solvents={}):
    """
    Driver to train a Neural Network (or other machine-learning approach) for
    a dataset. Key arguments are the wrapper, which supplies the interface
    to the ML model, and the specification of the trajectory data.

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents (implicit only, here). Entries are the full names.
    task: MLTrainTask class
        Argument list for the whole MLTrainTask job - see ML_Training module documentation for more detail.
        
        Arguments used only within the driver routine include:

        ``mltrain_task.traj_links``: Expressed as a dictionary containing trajectory
        labels and full paths of trajectory data files, relative to the path
        from which the script was invokes
        
        Arguments for which the driver routine will perform solute & solvent name substitution:

        ``mltrain_task.seed``: 
    """
    from os import symlink, path
    from ase.io import Trajectory, read, write

    # Determine if we are in a job array, in which case find task_id
    task_id = parallel.get_array_task_id()
    if task_id is None:
        task_id = 0

    # Determine train_dir and solute/solvent names
    if mltrain_task.calc_dir_suffix is None:
        mltrain_task.calc_dir_suffix = mltrain_task.calc_suffix
    train_dir = mltrain_task.wrapper.calc_filename(mltrain_task.seed,
                target=mltrain_task.target,
                prefix=mltrain_task.calc_prefix,
                suffix=mltrain_task.calc_dir_suffix)
    solu,solv = get_solu_solv_names(mltrain_task.seed)
    
    # Determine location of xyz files
    if mltrain_task.geom_prefix is not None:
        geom_prefix = sub_solu_solv_names(mltrain_task.geom_prefix,mltrain_task.seed,all_solutes,all_solvents)
    else:
        geom_prefix = '.'    
    if solu=='all':
        solu_checked = list(all_solutes)[0]
    else:
        solu_checked = solu
    if solv is not None:
        train_seed_xyz = f'{train_dir}/{solu_checked}_{solv}.xyz'
    else:
        train_seed_xyz = f'{train_dir}/{solu_checked}.xyz'

    # Check we are in parent directory and train_dir exists
    base_path = path.basename(getcwd())
    if base_path != train_dir:
        if not path.exists(train_dir):
            print(f'# Creating directory {train_dir}')
            try:
                makedirs(train_dir)
            except FileExistsError:
                print(f"# Possible mid-air collision between jobs - directory {train_dir} exists")
    else:
        chdir('..')

    # Check that the xyz file exists, create it if it does not
    if not path.isfile(train_seed_xyz):
        if solv is not None:
            source_seed_xyz = f'{geom_prefix}/{solu_checked}_{solv}.xyz'
        else:
            source_seed_xyz = f'{geom_prefix}/{solu_checked}.xyz'
        if not path.isfile(source_seed_xyz):
            print(f'# Source file {source_seed_xyz} not found - trying to combine {solu_checked} and {solv} xyzs')
            merge_xyzs(solu_checked,solv,train_seed_xyz)
        else:
            copyfile(source_seed_xyz,f'{train_seed_xyz}')
        print(f'# Source file {source_seed_xyz} copied to {train_dir}')
    if mltrain_task.calc_prefix is None or mltrain_task.calc_prefix == "":
        mltrain_task.calc_prefix = train_dir+"/"
    
    # Process any traj_links present in the input task
    make_traj_links(mltrain_task,mltrain_task.traj_links,train_dir,'',all_solutes,all_solvents)
    if mltrain_task.traj_links_valid is not None:
        make_traj_links(mltrain_task,mltrain_task.traj_links_valid,train_dir,'valid',all_solutes,all_solvents)
    if mltrain_task.traj_links_test is not None:
        make_traj_links(mltrain_task,mltrain_task.traj_links_test,train_dir,'test',all_solutes,all_solvents)
    mltrain_task.run()
    
def mltest_driver(mltest,all_solutes,all_solvents):
    """
    Driver to run tests of a Neural Network (or other machine-learning approach)
    by comparing it to results from a dataset, which may have been
    calculated already using eg a Clusters task.
    Key arguments are the wrapper, which supplies the interface
    to the ML model, and the specification of the test data.

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents (implicit only, here). Entries are the full names.
    mltest_task: MLTestTask class
        Argument list for the whole MLTestTask job - see :mod:`~esteem.tasks.ml_testing` module documentation for more detail.
        
        Arguments used only within the driver routine include:

        ``mltest_task.traj_links``: Expressed as a dictionary containing trajectory
        labels and full paths of trajectory data files, relative to the path
        from which the script was invokes
        
        Arguments for which the driver routine will perform solute & solvent name substitution:

        ``mltest_task.calc_seed``:
        
        ``mltest_task.traj_prefix``:
    """
    
    from ase.io import Trajectory
    from esteem.wrappers import amp
    from esteem.trajectories import targstr
    from os.path import commonprefix
    from os import getcwd, readlink, remove

    base_path = path.basename(getcwd())

    # Determine if we are in a job array
    task_id = parallel.get_array_task_id()
    if task_id is None:
        task_id = 0
    
    # Check if solute or solute names need substituting into calculator seed name
    mltest.calc_seed = sub_solu_solv_names(mltest.calc_seed,mltest.seed,all_solutes,all_solvents)

    seed_state_str = f'{mltest.calc_seed}_{targstr(mltest.target)}'
    if mltest.calc_dir_suffix is None:
        mltest.calc_dir_suffix = mltest.calc_suffix
    test_dir = f'{seed_state_str}_{mltest.calc_dir_suffix}_test'
    if base_path != test_dir:
        if not path.exists(test_dir):
            print(f'# Creating directory {test_dir}')
            try:
                makedirs(test_dir)
            except FileExistsError:
                print(f"# Possible mid-air collision between jobs - directory {test_dir} exists")
        chdir(test_dir)
        print(f'# Moved to {getcwd()}')

    if path.exists(f"../{mltest.seed}.xyz"):
        try:
            copyfile(f"../{mltest.seed}.xyz",f"{mltest.seed}.xyz")
        except PermissionError:
            print(f"# Could not copy from ../{mltest.seed}.xyz to {mltest.seed}.xyz - permission error")

    if isinstance(mltest.wrapper,amp.AMPWrapper):
        calcfn = mltest.wrapper.calc_filename(mltest.seed,mltest.target,prefix=mltest.calc_prefix,suffix=mltest.calc_suffix)
        calc_file = calcfn+ml_wrapper.calc_ext
        mltest.calc_suffix = suffix+"_test"
        calcfn = ml_wrapper.calc_filename(mltest.seed,mltest.target,prefix=mltest.calc_prefix,suffix=mltest.calc_suffix)
        test_calc_file = calcfn+ml_wrapper.calc_ext
        test_calc_log = calcfn+ml_wrapper.log_ext
        if os.path.isfile(calc_file):
            print(f"# Copying {calc_file} to {test_calc_file} for testing")
            shutil.copyfile(calc_file,test_calc_file)
        else:
            raise Exception("# Calculator file does not exist: ",calc_file)
    
    if mltest.calc_seed is None:
        mltest.calc_seed = mltest.seed

    # Check if the traj_prefix string contains "{solu}" or "{solv}", in
    # which case replace these with the appropriate solvent or solute string
    mltest.traj_prefix = sub_solu_solv_names(mltest.traj_prefix,mltest.calc_seed,all_solutes,all_solvents)
    
    # Process any traj_links present in the input task
    if hasattr(mltest,'traj_links'):
        if mltest.traj_links is not None:
            origdir = getcwd()
            # switch to the appropriate directory, so links are made there
            if mltest.traj_prefix != "" and mltest.traj_prefix is not None:
                chdir("../"+mltest.traj_prefix)
                print(f'# Moved to {getcwd()}')

            # obtain a list of the trajectories to be used in the training
            which_trajs,trajnames = mltest.get_trajnames()
            trajnames = dict(zip(which_trajs,trajnames))
            print(f'# Creating symlinks to trajectories')
            for traj in which_trajs:
                if traj not in mltest.traj_links:
                    continue

                # If there are multiple links that must be set up to assemble this trajectory,
                # the user may supply a list of links. Otherwise, we make a list with just
                # the single entry provided
                if isinstance(mltest.traj_links[traj],list):
                    traj_links = mltest.traj_links[traj]
                else:
                    traj_links = [mltest.traj_links[traj]]

                # Loop over the list made above
                for traj_link in traj_links:

                    # Check if the traj_link string contains "{solu}" or "{solv}", in
                    # which case replace these with the appropriate solvent or solute string
                    traj_link = sub_solu_solv_names(traj_link,mltest.seed,all_solutes,all_solvents)
                        
                    # Find the length of the trajectory and print it along with the proposed link
                    try:
                        t = Trajectory('../'+traj_link); l = len(t); t.close()
                    except Exception as e:
                        print(e)
                        l = 0
                    print('#',traj_link,'<->',trajnames[traj],' len=',l)

                    # Check the link destination exists, and if so make the link
                    if not path.isfile('../'+traj_link) and not path.islink('../'+traj_link):
                        raise Exception(f'# File to link to not found for trajectory {traj}: ../{traj_link} from {getcwd()}')
                    if path.islink(trajnames[traj]):
                        if (readlink(trajnames[traj])!='../'+traj_link):
                            print(f'# Removing pre-existing link to {readlink(trajnames[traj])}')
                            remove(trajnames[traj])
                    if not path.isfile(trajnames[traj]) and not path.islink(trajnames[traj]):
                        print(f'# Making link to: ../{traj_link} called {trajnames[traj]} in {getcwd()}')
                        symlink('../'+traj_link,trajnames[traj])

            # switch back to starting directory
            if mltest.traj_prefix != "" and mltest.traj_prefix is not None:
                chdir(origdir)

    if isinstance(mltest.which_trajs,dict):
        traj_dict = mltest.which_trajs
        for t in traj_dict:
            if t in mltest.ref_mol_seed_dict:
                ref_mol_seed = mltest.ref_mol_seed_dict[t]
                mltest.seed = sub_solu_solv_names(ref_mol_seed,mltest.calc_seed,all_solutes,all_solvents)
                print(f'# Reference molecule seed set to {mltest.seed}')
            mltest.which_trajs = traj_dict[t]
            plotfile = seed_state_str + '_' + mltest.calc_suffix + '_' + str(t) + '.png'
            mltest.plotfile = sub_solu_solv_names(plotfile,mltest.calc_seed,all_solutes,all_solvents)
            print(f"# Plotfile set to {mltest.plotfile}")
            try:
                mltest.run()
            except Exception as e:
                print(f'# Error when testing trajectory, continuing')
                print(e)
    else: # single trajectory to test
        if mltest.plotfile is not None:
            mltest.plotfile = sub_solu_solv_names(mltest.plotfile,mltest.calc_seed,all_solutes,all_solvents)
        print(f"# Plotfile set to {mltest.plotfile}")
        mltest.run()
    
    if isinstance(mltest.wrapper,amp.AMPWrapper):
        if os.path.isfile(test_calc_log):
            os.remove(test_calc_log)

def mltraj_driver(mltraj,all_solutes,all_solvents,cleanup_only=False):
    from os import symlink, path, remove
    from esteem.trajectories import get_trajectory_list,targstr,merge_traj

    # Determine if we are in a job array, in which case just run one trajectory
    task_id = parallel.get_array_task_id()
    if task_id is not None:
        try:
            mltraj.which_trajs = [get_trajectory_list(mltraj.ntraj)[task_id]]
        except:
            print(f'# Could not find an entry for task_id {task_id} in trajectory list {get_trajectory_list(mltraj.ntraj)}')
            return
    
    # Check if solute or solute names need substituting into calculator seed name
    mltraj.calc_seed = sub_solu_solv_names(mltraj.calc_seed,mltraj.seed,all_solutes,all_solvents)

    # Same for snap_calc_params if present
    if hasattr(mltraj,'snap_calc_params'):
        if mltraj.snap_calc_params is not None:
            if 'calc_seed' in mltraj.snap_calc_params:
                mltraj.snap_calc_params['calc_seed'] = sub_solu_solv_names(
                    mltraj.snap_calc_params['calc_seed'],mltraj.seed,all_solutes,all_solvents)
    
    # Check if we are in the right directory, create it if it does not exists
    base_path = path.basename(getcwd())
    seed_state_str = f'{mltraj.seed}_{targstr(mltraj.target)}'
    if mltraj.calc_dir_suffix is not None:
        traj_dir = f'{seed_state_str}_{mltraj.calc_dir_suffix}_mldyn'
    else:
        traj_dir = f'{seed_state_str}_{mltraj.calc_suffix}_mldyn'
    if base_path != traj_dir:
        if not path.exists(traj_dir):
            print(f'# Creating directory {traj_dir}')
            try:
                makedirs(traj_dir)
            except FileExistsError:
                print(f"# Possible mid-air collision between jobs - directory {traj_dir} exists")
        chdir(traj_dir)

    # Check if we need to create a symlink to or merge the initial trajectories file
    if mltraj.md_init_traj_link is not None:
        init_traj = seed_state_str+"_md_init.traj"
        if not isinstance(mltraj.md_init_traj_link,list):
            traj_link = sub_solu_solv_names(mltraj.md_init_traj_link,mltraj.seed,all_solutes,all_solvents)
            if not path.isfile(init_traj) and not path.islink(init_traj):
                print(f"# Creating symlink: {init_traj} -> ../{traj_link} ")
                symlink('../'+traj_link,init_traj)
        else:
            all_traj_links = ['../'+sub_solu_solv_names(traj_link,mltraj.seed,all_solutes,all_solvents) for traj_link in mltraj.md_init_traj_link]
            if task_id is not None:
                if not path.isfile(init_traj):
                    print(f'# File not found: {init_traj}')
                    raise Exception(f"# Please merge trajectories by running with no task_id before running individual task_id's")
            else:
                if path.islink(init_traj):
                    print(f'# Merged initial trajectories file {init_traj} is a link but should be a file')
                    raise Exception("# Please remove the link before merging the trajectories")
                merge_traj(all_traj_links,init_traj)

    # check if any of the initial geometry files are present in the parent directory, and if so copy
    # them to the current directory
    solu, solv = get_solu_solv_names(mltraj.seed)
    for suffix in [".xyz"]:
        # See if a geometry file for a "complex" exists, or the solvent and/or solute geometries
        geom_prefix = mltraj.geom_prefix
        if solv is not None:
            geom_prefix = '../' + geom_prefix + '/is_opt_{solv}'
        else:
            geom_prefix = '../' + geom_prefix + '/opt'
        geom_prefix = sub_solu_solv_names(geom_prefix,mltraj.seed,all_solutes,all_solvents)
        for geomfile in [seed_state_str+suffix,solu+suffix,solv+suffix]:
            geomfile_in = geom_prefix+'/'+geomfile
            if path.isfile(geomfile_in) or path.islink(geomfile_in):
                if not path.isfile(geomfile):
                    print(f'# Copying from {geomfile_in} to {geomfile}')
                    copyfile(geomfile_in,geomfile)
                else:
                    print(f'# Geometry {geomfile} already present')
            else:
                print(f'# Geometry {geomfile_in} not found to copy')

    # Set default value of calc_seed if not otherwise set
    if mltraj.calc_seed is None:
        mltraj.calc_seed = mltraj.seed

    # Run the MLTraj task
    if not cleanup_only:
        mltraj.run()
    
    # If requested, carve spheres centered on solute from trajectory, then delete full trajectory
    if mltraj.carve_trajectory_radius is not None:
        # If we provided a dictionary for the radius, find the specific entry that we need
        if isinstance(mltraj.carve_trajectory_radius,dict):
            if solv in mltraj.carve_trajectory_radius:
                mltraj.carve_trajectory_radius = mltraj.carve_trajectory_radius[solv]
            else:
                raise Exception(f"mltraj.carve_trajectory_radius is a dictionary but contains no entry for solvent '{solv}'")
        mltraj_cleanup(mltraj)

def mltraj_cleanup(mltraj):
    from esteem.tasks.clusters import ClustersTask
    from os import symlink, path, remove
    from esteem.trajectories import get_trajectory_list,targstr

    ct = ClustersTask()
    ct.solute,ct.solvent = get_solu_solv_names(mltraj.seed)
    ct.which_target = mltraj.target
    for ct.which_traj in mltraj.which_trajs:
        ct.min_snapshots = 0;
        ct.max_snapshots = mltraj.nsnap
        ct.carved_suffix = f"{mltraj.traj_suffix}_carved"
        ct.md_suffix = f"{targstr(ct.which_target)}_{ct.which_traj}_{mltraj.traj_suffix}"
        solvstr = f'_{ct.solvent}' if ct.solvent is not None else ''
        traj_carved_file = f'{ct.solute}{solvstr}_{targstr(ct.which_target)}_{ct.which_traj}_{ct.carved_suffix}.traj'
        if path.exists(traj_carved_file) and path.getsize(traj_carved_file)>0:
            print(f'# Skipping carving spheres for postprocessing - {traj_carved_file} already present')
        else:
            ct.carve_spheres(solvent_radius=mltraj.carve_trajectory_radius)
        #remove(f'{ct.solute}_{ct.solvent}_{ct.md_suffix}.traj')
        if mltraj.recalculate_carved_traj:
            ct.wrapper = mltraj.snap_wrapper
            ct.output = f"{mltraj.traj_suffix}_recalc"
            ct.calc_params = mltraj.snap_calc_params
            ct.target = mltraj.snap_calc_params['target']
            ct.nroots = mltraj.target
            traj_recalc_file = f'{ct.solute}{solvstr}_{targstr(ct.which_target)}_{ct.which_traj}_{ct.output}.traj'
            if path.exists(traj_recalc_file) and path.getsize(traj_recalc_file)>0:
                print(f'# Skipping recalculating clusters in postprocessing - {traj_recalc_file} already present')
            else:
                ct.run()
        if mltraj.store_full_traj:
            # Remove equilibration trajectory data
            eq_dir = f"{mltraj.traj_suffix}_equil"
            ct.md_suffix = f"{targstr(ct.which_target)}_{ct.which_traj}_{mltraj.traj_suffix}_equil0000"
            if mltraj.nequil>0:
                eq_file = f'{eq_dir}/{ct.solute}_{ct.solvent}_{ct.md_suffix}.traj'
                if path.exists(eq_file):
                    remove(eq_file)
            # Move to MD directory
            md_dir = f"{mltraj.traj_suffix}_md"
            chdir(md_dir)
            # Copy geometry files to current directory
            copyfile(f"../{ct.solute}.xyz",f"{ct.solute}.xyz")
            copyfile(f"../{ct.solvent}.xyz",f"{ct.solvent}.xyz")
            # Carve main md trajectory data
            ct.md_suffix = f"{targstr(ct.which_target)}_{ct.which_traj}_{mltraj.traj_suffix}_md0000"
            if mltraj.nsnap>0:
                md_file = f'{ct.solute}_{ct.solvent}_{ct.md_suffix}.traj'
                if not path.exists(md_file):
                    print(f"# No MD trajectory file found: {md_file} - continuing")
                    continue
            ct.carved_suffix = f"{mltraj.traj_suffix}_md0000_carved"
            ct.max_snapshots = mltraj.nsnap * mltraj.md_steps
            ct.carve_spheres(solvent_radius=mltraj.carve_trajectory_radius)
            # Remove full md trajectory data
            remove(f'{ct.solute}_{ct.solvent}_{ct.md_suffix}.traj')
            chdir('..')


# # Plot Spectra from cluster calculations

# In[ ]:


import glob
import numpy as np

def add_arrows_and_label(warp_params,arrow1_pos,arrow2_pos,s_ax,task):
    
    s_ax.arrow(arrow1_pos[0],arrow1_pos[1],arrow1_pos[2],arrow1_pos[3],
               head_width=0.1, head_length=10, fc='k', ec='k',
               length_includes_head = True)
    if task.warp_scheme == 'alphabeta' or task.warp_scheme == 'betabeta':
        s_ax.arrow(arrow2_pos[0],arrow2_pos[1],arrow2_pos[2],arrow2_pos[3],
                   head_width=0.1, head_length=10, fc='k', ec='k',
                   length_includes_head = True)

    warp_str = ''
    if task.warp_scheme=='alphabeta':
        warp_str = warp_str + f'\u03B1 = {warp_params[0]:0.3f}, '
        warp_str = warp_str + f'\u03B2 = {warp_params[1]:0.3f}'
    elif task.warp_scheme=='betabeta':
        warp_str = warp_str + f'\u03B2 1 = {warp_params[0]:0.3f}, '
        warp_str = warp_str + f'\u03B2 2 = {warp_params[1]:0.3f}'
    else:
        warp_str = warp_str + f'\u03B2 = {warp_params[0]:0.3f}'
    
    return warp_str

def spectral_warp_driver(all_solutes,all_solvents,task,annotation=None):
    """
    Driver to calculate spectral warping parameters for a range of solute/solvent pairs

    all_solutes: dict of strings
        Keys are shortnames of the solutes. Entries are the full names.
    all_solvents: dict of strings
        Keys are shortnames of the solvents. Entries are the full names.
    task: SpectraTask class
        Argument list for the whole spectra job - see Spectra module documentation for more detail.
        
        Arguments used only in the driver include:
        
        ``task.exc_suffix``: Directory in which results of excitation calculations performed
        by the Clusters task can be found. The pattern used to find matches is:
        '{solute}_{solvent}_{exc_suffix}/{solute}_{solvent}_solv*.out'
        
        ``task.warp_origin_ref_peak_range``: Peak range searched when looking for 'reference' peaks.
        in the origin spectrum for spectral warping.
        
        ``task.warp_dest_ref_peak_range``: Peak range searched when looking for 'reference' peaks.
        in the destination spectrum for spectral warping.
        
        ``task.warp_broad``: Broadening to be applied to origin and destination spectra.
        
        ``task.warp_inputformat``: Format of the files to be loaded for origin and destination spectra.
        [TODO: May need to be adjusted to allow separate ``task.warp_ref_inputformat`` and ``task.warp_dest_inputformat``]
        
        ``task.warp_files``: File pattern to search for when looking for origin and destination spectra
        for spectral warping.
        
        ``task.merge_solutes``: Dictionary: each entry should be a list of solute names that will be merged into the
        corresponding key
    """
    from copy import deepcopy
    import matplotlib.pyplot as pyplot
    import glob
    
    # Set up lists of solute-solvent pairs
    exc_suffix = task.exc_suffix
    all_pairs = list(itertools.product(all_solvents.items(),all_solutes.items()))

    # Retrieve spectral warping settings from args
    #warp_origin_prefix = task.warp_origin_prefix
    #warp_dest_prefix = task.warp_dest_prefix
    warp_inputformat = task.warp_inputformat
    #warp_dest_files = task.warp_files
    warp_params = []
    store_wavelength = deepcopy(task.wavelength)
    store_broad = task.broad
    store_renorm = task.renorm
    store_inputformat = task.inputformat
    
    if isinstance(task.warp_origin_ref_peak_range,dict):
        all_warp_origin_ref_peak_ranges = task.warp_origin_ref_peak_range
    else:
        all_warp_origin_ref_peak_ranges = None
    if isinstance(task.warp_dest_ref_peak_range,dict):
        all_warp_dest_ref_peak_ranges = task.warp_dest_ref_peak_range
    else:
        all_warp_dest_ref_peak_ranges = None
    if task.warp_broad is not None:
        task.broad = task.warp_broad
    task.renorm = False
    
    # Loop over solvent-solute pairs to find warp params for each pair
    for i,((solvent,fullsolvent),(solute,fullsolute)) in enumerate(all_pairs):

        print('\nSpectral Warping for',solute)
        if all_warp_origin_ref_peak_ranges is not None:
            if solute in all_warp_origin_ref_peak_ranges:
                task.warp_origin_ref_peak_range = all_warp_origin_ref_peak_ranges[solute]
        if all_warp_dest_ref_peak_ranges is not None:
            if solute in all_warp_dest_ref_peak_ranges:
                task.warp_dest_ref_peak_range = all_warp_dest_ref_peak_ranges[solute]
        # if peak ranges for spectral warps are not specified, use full wavelength range
        task.wavelength = store_wavelength
        if task.warp_origin_ref_peak_range is None:
            task.warp_origin_ref_peak_range = deepcopy(task.wavelength)
        if task.warp_dest_ref_peak_range is None:
            task.warp_dest_ref_peak_range = deepcopy(task.wavelength)

        # Process Solute spectra
        task.warp_params = [0.0]
        if task.warp_scheme == 'alphabeta':
            task.warp_params = [1.0,0.0]
        if task.warp_scheme == 'betabeta':
            task.warp_params = [0.0,0.0,0.0,0.0]
        arrow1_pos = [0]*4
        arrow2_pos = [0]*4
        s_fig = None; s_ax = None
                
        # Plot destination spectrum
        # Cut down wavelength according to ref_peak_range values
        task.wavelength = deepcopy(store_wavelength)
        if task.warp_dest_ref_peak_range[0]<task.wavelength[0]:
            task.wavelength[0] = task.warp_dest_ref_peak_range[0]
        if task.warp_dest_ref_peak_range[1]>task.wavelength[1]:
            task.wavelength[1] = task.warp_dest_ref_peak_range[1]
        pre_glob_files = sub_solu_solv_names(task.warp_dest_files,f'{solute}_{solvent}',all_solutes,all_solvents)
        task.files = glob.glob(pre_glob_files)
        if task.files==[]:
            raise Exception(f"Error in spectral_warp_driver: No files were found matching destination task.files: {pre_glob_files}")
        task.inputformat = warp_inputformat
        dest_spectrum,spec,s_fig,s_ax,trans_orig_dest,_ = task.run(s_fig,s_ax,plotlabel='dest')
        if dest_spectrum is None:
            raise Exception(f"Error in spectral_warp_driver: No spectrum was plotted for task.files: {pre_glob_files} {task.files}")
        if annotation is not None:
            s_ax.text(task.wavelength[0],0.95,annotation)
        pyplot.setp(spec,color='r')

        # Plot origin spectrum
        task.wavelength = store_wavelength.copy()
        if task.warp_origin_ref_peak_range[0]<task.wavelength[0]:
            task.wavelength[0] = task.warp_origin_ref_peak_range[0]
        if task.warp_origin_ref_peak_range[1]>task.wavelength[1]:
            task.wavelength[1] = task.warp_origin_ref_peak_range[1]
        #print(task.warp_origin_files)
        pre_glob_files = sub_solu_solv_names(task.warp_origin_files,f'{solute}_{solvent}',all_solutes,all_solvents)
        task.files = glob.glob(pre_glob_files)
        if task.files==[]:
            raise Exception(f"Error in spectral_warp_driver: No files were found matching origin task.files: {pre_glob_files}")
        origin_spectrum,spec,s_fig,s_ax,trans_orig_origin,_ = task.run(s_fig,s_ax,plotlabel='origin')
        pyplot.setp(spec,color='b')
        wp = task.find_spectral_warp_params(dest_spectrum,origin_spectrum,arrow1_pos,arrow2_pos)
        warp_params.append(wp)
        task.warp_params  = warp_params[-1]
        task.wavelength = [min(store_wavelength[0],task.warp_origin_ref_peak_range[0],task.warp_dest_ref_peak_range[0]),
                           max(store_wavelength[1],task.warp_origin_ref_peak_range[1],task.warp_dest_ref_peak_range[1]),
                           task.warp_origin_ref_peak_range[2]]
        warped_spectrum,spec,s_fig,s_ax,_,_ = task.run(s_fig,s_ax,plotlabel='warped')
        pyplot.setp(spec,color='g')

        s_ax.legend()
        if True:
            warp_str = add_arrows_and_label(wp,arrow1_pos,arrow2_pos,s_ax,task)
            s_ax.set_title(f'Spectral warp params for {fullsolute} in {fullsolvent}: {warp_str} eV')
        
        #print(f'Transition origins for {fullsolute} in {fullsolvent}:')
        max_trans = min(len(trans_orig_dest[0]),len(trans_orig_origin[0]))
        max_trans = 0
        for iexc in range(max_trans):
            to = trans_orig_origin[0][iexc]
            td = trans_orig_dest[0][iexc]
            print(f'excitation {iexc+1}: {to[0][1]} eV ({to[0][2]:0.4f}) --> {td[0][1]} eV ({td[0][2]:0.4f})')
            to = [s[:2] for s in to[1:] if abs(s[2])>0.3]
            td = [s[:2] for s in td[1:] if abs(s[2])>0.3]
            print(f'origins    {iexc+1}: {to} --> {td}')
        s_fig.savefig(f'{solute}_{solvent}_check_warp.png')

    # Restore settings that we over-wrote when doing spectral warping
    task.renorm = store_renorm
    task.wavelength = deepcopy(store_wavelength)
    task.broad = store_broad
    task.inputformat = store_inputformat
    task.warp_origin_ref_peak_range = all_warp_origin_ref_peak_ranges
    task.warp_dest_ref_peak_range = all_warp_dest_ref_peak_ranges
    
    return warp_params

def spectra_driver(all_solutes,all_solvents,task,warp_params=None,cluster_spectra=[],c_fig=None,c_ax=None):

    import glob
    
    # Set up lists of solute-solvent pairs
    exc_suffix = task.exc_suffix
    all_pairs = list(itertools.product(all_solvents.items(),all_solutes.items()))
    all_paths = []
    for (solvent,_),(solute,_) in all_pairs:
        if isinstance(task.exc_suffix,dict):
            if solvent is not None:
                exc_path = f'{solute}_{solvent}_{exc_suffix[solute]}'
            else:
                exc_path = f'{solute}_{exc_suffix[solute]}'
        else:
            if solvent is not None:
                exc_path = f'{solute}_{solvent}_{exc_suffix}'
            else:
                exc_path = f'{solute}_{exc_suffix}'
        all_paths.append(exc_path)
    base_path = path.basename(getcwd())
    orig_output = task.output

    # Now plot explicit solvent spectrum
    for i,((solvent,fullsolvent),(solute,fullsolute)) in enumerate(all_pairs):

        # Find cluster spectra path names 
        exc_suffix = task.exc_suffix
        if isinstance(exc_suffix,dict):
            if solute in exc_suffix:
                exc_suffix = task.exc_suffix[solute]

        if solvent is not None:
            exc_path = f'{solute}_{solvent}_{exc_suffix}'
        else:
            exc_path = f'{solute}_{exc_suffix}'
        solusolvstr = f'{fullsolute} in {fullsolvent}' if solvent is not None else fullsolute
        
        if base_path in all_paths and base_path!=exc_path:
            continue

        if base_path not in all_paths:
            if not path.exists(exc_path):
                print(f'\nSkipping Spectra for: {solusolvstr}')
                print(f'Directory {exc_path} not found')
                continue
            else:
                chdir(exc_path)

        print(f'\nPlotting Spectra for {solusolvstr}')

        # List files matching pattern:
        if task.files is not None:
            task.files = sub_solu_solv_names(task.files,f'{solute}_{solvent}',all_solutes,all_solvents)
            if '*' in task.files:
                task.files = glob.glob(task.files)
            if task.renorm == -1:
                task.renorm = 1.0/float(len(task.files))

            # Check if any other solutes are this solute's merge list, and if so add their files
            if solute in task.merge_solutes:
                for other_solute in task.merge_solutes[solute]:
                    other_files = glob.glob(f'../{other_solute}_{solvent}_{exc_suffix}/{other_solute}_{solvent}_solv*.out')
                    task.files = task.files + other_files
                
            # Check if this solute is in any other solute's merge list
            found_solute = False
            for other_solute in task.merge_solutes:
                if solute in task.merge_solutes[other_solute]:
                    found_solute = True
                    continue
            if found_solute:
                chdir('..')
                continue
            print('Cluster files to process: ',len(task.files),task.files)
                
        if task.trajectory is not None:
            for i in range(len(task.trajectory)):
                for j in range(len(task.trajectory[i])):
                    t = task.trajectory[i][j]
                    task.trajectory[i][j] = sub_solu_solv_names(t,f'{solute}_{solvent}',all_solutes,all_solvents)
                    if task.correction_trajectory is not None:
                        t = task.correction_trajectory[i][j]
                        task.correction_trajectory[i][j] = sub_solu_solv_names(t,f'{solute}_{solvent}',all_solutes,all_solvents)
                    if task.vibration_trajectory is not None:
                        t = task.vibration_trajectory[i][j]
                        task.vibration_trajectory[i][j] = sub_solu_solv_names(t,f'{solute}_{solvent}',all_solutes,all_solvents)
            if hasattr(task.wrapper,'rootname'):
                task.wrapper.rootname = sub_solu_solv_names(task.wrapper.rootname,
                                                            f'{solute}_{solvent}',all_solutes,all_solvents)
                task.wrapper.input_filename = sub_solu_solv_names(task.wrapper.input_filename,
                                                                  f'{solute}_{solvent}',all_solutes,all_solvents)

        # Retrieve spectral warping parameters
        if warp_params is not None:
            task.warp_params = warp_params[i]
        else:
            task.warp_params = None

        rgb = np.array((0,0,1.0))
        if isinstance(task.line_colours,list):
            rgb = task.line_colours[i]

        task.output = sub_solu_solv_names(orig_output,f'{solute}_{solvent}',all_solutes,all_solvents)
        
        cluster_spectrum,spec,c_fig,c_ax,_,_ = task.run(c_fig,c_ax,
              plotlabel=f'{fullsolute} in {fullsolvent}',rgb=rgb)
        cluster_spectra.append(cluster_spectrum)
        
        # Return to parent directory if we are looping over dirs
        if base_path not in all_paths:
            chdir('..')


    return cluster_spectra,c_fig,c_ax


# # Helper functions

# In[ ]:


# return args objects with default values for each script
from esteem.tasks import solutes, solvate, clusters, spectra

def get_default_args():
    parser = solutes.make_parser()
    solutes_args = parser.parse_args("")
    parser = solvate.make_parser()
    solvate_args = parser.parse_args(['--solute','a','--solvent','b'])
    parser = clusters.make_parser()
    clusters_args = parser.parse_args(['--solute','a','--solvent','b'])
    parser = spectra.make_parser()
    spectra_args = parser.parse_args("")
    
    return solutes_args, solvate_args, clusters_args, spectra_args

