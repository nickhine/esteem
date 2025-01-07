#!/usr/bin/env python
# coding: utf-8


"""
Defines routines that implement Active Learning by duplicating a prototype task
across an array of iterations, targets, random seeds etc for each of the 4 steps
of an Active Learning cycle: MLTrain, MLTraj, Clusters, MLTest, and spectra tasks
once training is completed.

To use, create a prototype for each, and lists of calculators, targets, random seeds,
and call then call each of the create_* routines to return lists of tasks to
pass to drivers.main()
"""


from esteem.trajectories import get_trajectory_list
from esteem.tasks.clusters import ClustersTask
from esteem.tasks.ml_training import MLTrainingTask
from esteem.tasks.ml_trajectories import MLTrajTask
from esteem.tasks.ml_testing import MLTestingTask
from esteem.tasks.spectra import SpectraTask

from copy import deepcopy

# Mapping of trajectory labels to subset selection methods (can be overridden if needed)
traj_to_ssm_map = {'S':'E','U':'U','T':'D','R':'R','Q':'R'}
# Mapping of calculator names to trajectory labels (can be overridden if needed)
calc_to_traj_map = {'r':'R','s':'S','t':'T','u':'U'}

def get_traj_from_calc(calc):
    calc = calc[-1]
    if calc not in calc_to_traj_map:
        raise Exception(f'Unknown calculator label {calc}. Expected calculator labels: {calc_to_traj_map}')
    return calc_to_traj_map[calc]

def get_ssm_from_traj(traj):
    if traj not in traj_to_ssm_map:
        raise Exception(f'Unknown trajectory label {traj}. Expected trajectory labels: {traj_to_ssm_map}')
    return traj_to_ssm_map[traj]

def get_gen_from_calc(calc):
    try:
        return int(calc[6:-1])
    except:
        return None

def pref(calc):
    if '_' in calc:
        return calc.split('_')[0]
    else:
        return calc[0:6]

def suff(calc):
    return calc[4:]

def create_clusters_tasks(task:ClustersTask,train_calcs,seed,traj_suffix,md_suffix,
                          md_dir_suffix,targets,rand_seed,meth,truth,
                          separate_valid=False):
    """
    Returns a dictionary of clusters tasks, based on an input prototype task supplied by
    the user, for all the required clusters tasks for an Active Learning task.
    
    Takes lists of calculators, targets, random seeds, and strings stating the ML method
    and the ground truth method
    """

    # By default, the trajectory for validation is the same size as the main traj
    init_min_snapshots = task.min_snapshots
    init_max_snapshots = task.max_snapshots
    # It can be overridden by setting valid_snapshots
    if separate_valid:
        if task.valid_snapshots is not None:
            valid_snapshots = task.valid_snapshots
        else:
            valid_snapshots = task.max_snapshots - task.min_snapshots
    # Define empty dictionary for new tasks
    new_clusters_tasks = {}
    # Loop over calculators and trajectory targets
    for t in train_calcs:
        for tp in [t]:
            for target in targets:
                targstr = targets[target]
                if isinstance(targstr,dict):
                    targstr = "".join((targstr[p] if p!="diff" else "") for p in targstr)
                if isinstance(targets[target],dict):
                    task.target = targets[target]
                    targets_dict = targets[target]
                else:
                    task.target = list(targets)
                    targets_dict = {target:targets[target]}
                task.exc_suffix = f'{targstr}_{meth}{t}'
                task.output = f'{truth}_{suff(tp)}'
                task.carved_suffix = f'carved_{suff(tp)}'
                task.selected_suffix = f'selected_{suff(tp)}'
                if traj_suffix!='mlclus':
                    task.exc_suffix = f'{task.exc_suffix}_{traj_suffix}'
                    task.output = f'{task.output}_{traj_suffix}'
                    task.carved_suffix = f'{task.carved_suffix}_{traj_suffix}'
                    task.selected_suffix = f'{task.selected_suffix}_{traj_suffix}'
                if hasattr(task,'exc_dir_suffix'):
                    if task.exc_dir_suffix is None:
                        task.exc_dir_suffix = f'{targstr}_{meth}{pref(t)}_{traj_suffix}'
                    else:
                        task.exc_dir_suffix = task.exc_dir_suffix.replace('{targ}',targstr)
                else:
                    task.exc_dir_suffix = f'{targstr}_{meth}{pref(t)}_{traj_suffix}'
                task.script_settings['logdir'] = task.output
                wlist = [get_traj_from_calc(tp)]
                if separate_valid:
                    wlist += ['Q']
                # Make a list of trajectories to find
                wplist = get_trajectory_list(len(rand_seed))
                rslist = list(rand_seed)
                for iw,w in enumerate(wlist):
                    # for the main trajectory, reset the number of snapshots
                    if iw==0:
                        task.min_snapshots = init_min_snapshots
                        task.max_snapshots = init_max_snapshots
                        if task.subset_selection_method is None:
                            task.subset_selection_method = get_ssm_from_traj(w)
                        task.subset_selection_which_traj = w
                    elif separate_valid: # for the validation/testing trajectories, offset the snapshots
                        task.min_snapshots = task.max_snapshots
                        task.max_snapshots = task.max_snapshots + valid_snapshots
                    if not hasattr(task,'md_dir_suffix'):
                        task.md_prefix = f'{seed}_{targstr}_{meth}{pref(tp)}_{md_dir_suffix}'
                    else:
                        task.md_prefix = f'{seed}_{targstr}_{task.md_dir_suffix}'
                    task.md_suffix = []
                    # loop over random seeds defining committee calcs
                    for irs,rs in enumerate(rslist):
                        # loop over targets if targets[target] is dict
                        for itarg,targets_dict_key in enumerate(targets_dict.keys()):
                            if targets_dict_key=='diff':
                                continue
                            # find target of this MD trajectory
                            targstrp = targets_dict[targets_dict_key]
                            wp = chr(ord('A')+irs+itarg*len(rslist))
                            # make suffix for the MD trajectory and append to list
                            mdsuff = f'{targstrp}_{wp}_{meth}{tp}{rs}_{md_suffix}'
                            task.md_suffix.append(mdsuff)
                    # Collapse list if it just contains one entry
                    if len(task.md_suffix)==1:
                        task.md_suffix = task.md_suffix[0]
                    task.which_traj = w
                    traj_char = '_'+w #'' if w==t[-1].upper() else '_'+w
                    new_clusters_tasks[task.exc_suffix+traj_char] = deepcopy(task)
    return new_clusters_tasks


def get_keys(task):
    all_keys = ['train']
    if hasattr(task,'traj_links_valid') and task.traj_links_valid is not None:
        all_keys += ['valid']
    if hasattr(task,'traj_links_test') and task.traj_links_test is not None:
        all_keys += ['test']
    return all_keys

def add_trajectories(task,seeds,calc,traj_suffixes,dir_suffixes,ntraj,targets,target,truth):
    """
    Adds static trajectories
    """
    # Loop over initial source trajectories
    targstr = targets[target]
    passed = {i:0 for i in targets}
    for traj_suffix in traj_suffixes:
        dir_suffix = dir_suffixes[traj_suffix]
        # Make a list of the trajectories from this source
        for itarg1,target1 in enumerate(targets):
            if target1=='diff':
                continue
            targstr1 = targets[target1]
            if isinstance(targstr1,dict):
                continue
            offset = chr(ord('A')+itarg1-1) if itarg1>0 else ''
            if "qmd" in traj_suffix:
                # For QMD trajectories, the trajectory name depends on the target
                # (separately from the target of the trajectory)
                fullsuffix = f"{targstr1}_{traj_suffix}"
            else:
                fullsuffix = truth
            # handle case where seeds is a dictionary, and keys are target,suffix tuples
            if (isinstance(seeds,dict)):
               seeds_list = seeds[targstr1,traj_suffix]
            else:
               seeds_list = seeds
            # Loop over seeds
            for seed in seeds_list:
                all_keys = get_keys(task)
                targstr2_dict = targstr
                if not isinstance(targstr2_dict,dict):
                    targstr2_dict = {0: targstr2_dict}
                for targstr2_key in targstr2_dict:
                    if targstr2_key=='diff':
                        continue
                    targstr2 = targstr2_dict[targstr2_key]
                    targstr2_orig = targstr2
                    if seed=='{solv}_{solv}' and targstr2!='gs':
                        targstr2 = 'gs'
                    for ikey,key in enumerate(all_keys):
                        all_traj = get_trajectory_list(passed[target1]+ntraj[targstr1,traj_suffix])
                        for itraj,traj in enumerate(all_traj[passed[target1]:]):
                            trajsource = all_traj[itraj+ikey*ntraj[targstr1,traj_suffix]]
                            traj_dest = f"{seed}_{dir_suffix}/{seed}_{targstr2}_{trajsource}_{fullsuffix}.traj"
                            if key=='train':
                                task.traj_links[offset+traj] = traj_dest
                                new_trajs = [offset+all_traj[passed[target1]:][itraj]]
                                if isinstance(task.which_trajs,list):
                                    task.which_trajs += new_trajs
                                    task.which_targets += [targstr2_orig]
                                else:
                                    task.which_trajs.update({jtraj:jtraj for jtraj in new_trajs})
                                    task.which_targets.update({jtraj:targstr2_orig for jtraj in new_trajs})
                                    task.ref_mol_seed_dict.update({jtraj:seed for jtraj in new_trajs})
                                #print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs} (seed={seed}, itarg1={itarg1}, itraj={itraj}, passed[{target1}]={passed[target1]}')
                            elif key=='valid':
                                task.traj_links_valid[offset+traj] = traj_dest
                                new_trajs = [offset+all_traj[passed[target1]:][itraj]]
                                if isinstance(task.which_trajs_valid,list):
                                    task.which_trajs_valid += new_trajs
                                    task.which_targets_valid += [targstr2_orig]
                                else:
                                    task.which_trajs_valid.update({jtraj:jtraj for jtraj in new_trajs})
                                    task.which_targets_valid.update({jtraj:targstr2_orig for jtraj in new_trajs})
                                    task.ref_mol_seed_dict.update({jtraj:seed for jtraj in new_trajs})
                                #print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs_valid}')
                            elif key=='test':
                                task.traj_links_test[offset+traj] = traj_dest
                                new_trajs = [offset+all_traj[passed[target1]:][itraj]]
                                if isinstance(task.which_trajs_test,list):
                                    task.which_trajs_test += new_trajs
                                else:
                                    task.which_trajs_test.update({jtraj:jtraj for jtraj in new_trajs})
                                    task.which_targets_test.update({jtraj:targstr2_orig for jtraj in new_trajs})
                                    task.ref_mol_seed_dict.update({jtraj:seed for jtraj in new_trajs})
                                #print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs_test}')
                        passed[target1] += ntraj[targstr1,traj_suffix]
                        if passed[target1] > 26:
                            print('# Warning: more than 26 input trajectories for this target')
                            print('# Please ensure no overlap with other targets:')
                            print(task.which_trajs)

def add_iterating_trajectories(task,seeds,calc,iter_dir_suffixes,targets,target,meth,truth,only_gen=None):
    """
    Adds iterating trajectories
    """
    
    gen = get_gen_from_calc(calc)
    genstart = 0
    genend = gen
    if type(task) is MLTestingTask and gen is not None:
        if only_gen is not None:
            if only_gen==-1:
                genstart = 0
                genend = 0
            else:
                genstart = only_gen
                genend = only_gen + 1
        else:
            genend = gen + 1
    all_used_trajs = list(task.which_trajs.copy())
    if task.which_trajs_valid is not None:
        all_used_trajs += list(task.which_trajs_valid)
    if task.which_trajs_test is not None:
        all_used_trajs += list(task.which_trajs_test)
    if len(all_used_trajs)>0:
        last_static_traj_char = sorted(all_used_trajs)[-1]
    else:
        last_static_traj_char = chr(ord('A')-1)
    if gen is None or (gen < 1 and type(task) is not MLTestingTask):
        return
    targstr = targets[target]
    # Loop over generations prior to current
    for g in range(genstart,genend):
        calcp = f'{pref(calc)}{g}{calc[-1]}'
        # Use fixed traj_suffix along the lines of "orca_ac0u" currently - perhaps make templatable?
        traj_suffix = f'{truth}_{suff(calcp)}'
        if isinstance(iter_dir_suffixes,list):
            iter_dir_suffixes_dict = {traj_suffix: dir_suffix for dir_suffix in iter_dir_suffixes}
        else:
            iter_dir_suffixes_dict = {}
            for traj_suffix in iter_dir_suffixes:
                traj_suffix_new = f'{truth}_{suff(calcp)}'
                if traj_suffix is not None and traj_suffix!="":
                    traj_suffix_new = f'{traj_suffix_new}_{traj_suffix}'
                iter_dir_suffixes_dict[traj_suffix_new] = iter_dir_suffixes[traj_suffix]
        # Find character for generation: ''=0, 'A'=1, 'B'=2 etc
        gen_char = chr(ord('A')-1+g) if g>0 else ''
        # Loop over all targets for source trajectories
        offset = 1
        for targetp in targets:
            #if targetp > target:
            #    continue
            if targetp=='diff':
                continue
            targstrp = targets[targetp]
            if isinstance(targstrp,dict):
                continue
            # Loop over all dir suffixes and seeds
            for traj_suffix in iter_dir_suffixes_dict:
                ids_value = iter_dir_suffixes_dict[traj_suffix]
                if isinstance(ids_value,tuple):
                    dir_suffix, traj_seeds = ids_value[0:2]
                else:
                    dir_suffix = ids_value
                    traj_seeds = seeds
                # handle case where seeds is a dictionary, and keys are target,suffix tuples
                if (isinstance(traj_seeds,dict)):
                   seeds_list = traj_seeds[str(targstrp),str(dir_suffix)]
                elif (isinstance(traj_seeds,list)):
                   seeds_list = traj_seeds
                else:
                   seeds_list = [traj_seeds]
                # Loop over seeds
                for seed in seeds_list:
                    targstr2_dict = targstr
                    if not isinstance(targstr2_dict,dict):
                        targstr2_dict = {0: targstr2_dict}
                    for targstr2_key in targstr2_dict:
                        if targstr2_key=='diff':
                            continue
                        targstr2 = targstr2_dict[targstr2_key]
                        targstr2_orig = targstr2
                        if seed=='{solv}_{solv}':
                            if targstrp!='gs':
                                continue
                            if targstr2!='gs':
                                targstr2 = 'gs'
                        all_keys = get_keys(task)
                        for ikey,key in enumerate(all_keys):
                            # First get the base character for this type of trajectory
                            traj_type_char = get_traj_from_calc(calc)
                            if (ikey==1):
                                traj_type_char = 'Q'
                            # Offset first available char by the number of previous trajectories passed
                            traj_char = chr(ord(last_static_traj_char)+offset)
                            # Find the directory and filename for this trajectory
                            traj_link_dir = f"{seed}_{targstrp}_{meth}{pref(calcp)}_{dir_suffix}"
                            traj_link_file = f"{seed}_{targstr2}_{traj_type_char}_{traj_suffix}.traj"
                            # Add it to the list of links to make
                            # and to the list of trajectory characters to link
                            traj_dest = f"{traj_link_dir}/{traj_link_file}"
                            if key=='train':
                                task.traj_links[gen_char+traj_char] = traj_dest
                                new_traj = f'{gen_char+traj_char}'
                                if isinstance(task.which_trajs,list):
                                    task.which_trajs += [new_traj]
                                    task.which_targets += [targstr2_orig]
                                else:
                                    task.which_trajs[new_traj] = new_traj
                                    task.which_targets[new_traj] = targstr2_orig
                                    task.ref_mol_seed_dict[new_traj] = seed
                                #print(f'adding for {calc}: {targstr}_{calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs}')
                            elif key=='valid':
                                task.traj_links_valid[gen_char+traj_char] = traj_dest
                                new_traj = f'{gen_char}{traj_char}'
                                if isinstance(task.which_trajs_valid,list):
                                    task.which_trajs_valid += [new_traj]
                                    task.which_targets_valid += [targstr2_orig]
                                else:
                                    task.which_trajs_valid[new_traj] = new_traj
                                    task.which_targets_valid[new_traj] = targstr2_orig
                                    task.ref_mol_seed_dict[new_traj] = seed
                                #print(f'adding for {calc}: {targstr}_{calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs_valid}')
                            elif key=='test':
                                task.traj_links_test[gen_char+traj_char] = traj_dest
                                new_traj = f'{gen_char}{traj_char}'
                                if isinstance(task.which_trajs_test,list):
                                    task.which_trajs_test += [new_traj]
                                    #task.which_targets_test += [targstr2_orig]
                                else:
                                    task.which_trajs_test[new_traj] = new_traj
                                    #task.which_targets_test[new_traj] = targstr2_orig
                                    task.ref_mol_seed_dict[new_traj] = seed
                                #print(f'adding: {targstr}_{calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs_test}')
                            offset = offset + 1

def create_mltrain_tasks(train_task:MLTrainingTask,train_calcs,seeds,targets,rand_seed,meth,truth,
                         traj_suffixes=[],dir_suffixes={},ntraj={},
                         iter_dir_suffixes=[],delta_epochs=200,separate_valid=False):
    """
    Returns a dictionary of MLTrain tasks, based on an input prototype task supplied by
    the user, for all the required MLTrain tasks for an Active Learning task.
    
    Takes lists of calculators, targets, random seeds, and strings stating the ML method
    and the ground truth method, and lists of trajectories to use as initial inputs
    (plus the number of trajectories for each target and their location)
    """

    new_mltrain_tasks = {}
    if hasattr(train_task.wrapper.train_args,'max_num_epochs'):
        init_epochs = train_task.wrapper.train_args.max_num_epochs # MACE specific
        swa_init_epochs = train_task.wrapper.train_args.start_swa  # MACE specific
        if swa_init_epochs is None:
            swa_init_epochs = 100000

    for target in targets:
        if target=='diff':
            continue
        for t in train_calcs:
            # Calculator basic info
            train_task.traj_suffix = truth
            if isinstance(targets[target],dict):
                train_task.target = targets[target]
            else:
                train_task.target = target
            train_task.calc_dir_suffix = f"{meth}{pref(t)}"
            train_task.calc_prefix = ""
            # Set up links to trajectories - first empty the lists
            train_task.traj_links = {}
            train_task.which_trajs = []
            train_task.which_targets = []
            if separate_valid:
                train_task.traj_links_valid = {}
                train_task.which_trajs_valid = []
                train_task.which_targets_valid = []
            else:
                train_task.traj_links_valid = None
                train_task.which_trajs_valid = None
                train_task.which_targets_valid = None
            # Then add "static" configurations, that do not increase with AL generation
            add_trajectories(train_task,seeds,t,traj_suffixes,dir_suffixes,ntraj,targets,target,truth)
            # For generations > 0, we now add chosen subset trajectories for active learning
            add_iterating_trajectories(train_task,seeds,t,iter_dir_suffixes,targets,target,meth,truth)
            # extra epochs for each generation
            if hasattr(train_task.wrapper.train_args,'max_num_epochs'):  # MACE specific
                gen = get_gen_from_calc(t)
                if gen is not None:
                    train_task.wrapper.train_args.max_num_epochs = init_epochs + gen*delta_epochs
                    # same number of extra epochs for SWA
                    train_task.wrapper.train_args.start_swa = swa_init_epochs + gen*delta_epochs
            # Save this calculator to the list for each seed
            for rs in rand_seed:
                # Seed-specific info
                train_task.rand_seed = rand_seed[rs]
                train_task.wrapper.train_args.seed = rand_seed[rs] # MACE specific
                train_task.calc_suffix = f"{meth}{t}{rs}"
                targstr = targets[target]
                if isinstance(targstr,dict):
                    targstr = "".join((targstr[p] if p!="diff" else "") for p in targstr)
                new_mltrain_tasks[targstr+'_'+train_task.calc_suffix] = deepcopy(train_task)
    return new_mltrain_tasks

def create_mltraj_tasks(mltraj_task:MLTrajTask,train_calcs,targets,rand_seed,meth,md_wrapper,
                        traj_suffix='mldyn',snap_wrapper=None,two_targets=False,snap_targets=None):
    """
    Returns a dictionary of MLTraj tasks, based on an input prototype task supplied by
    the user, for all the required MLTraj tasks for an Active Learning task.
    
    Takes lists of calculators, targets, random seeds, and strings stating the ML method
    and wrappers for the MD itself and for the "committee" MD
    """

    new_mltraj_tasks = {}
    if mltraj_task.calc_seed is None:
        mltraj_task.calc_seed = f"{{solu}}" #_{{solv}}"
    for target in targets:
        if target=='diff':
            continue
        for t in train_calcs:
            mltraj_task.wrapper = md_wrapper
            mltraj_task.calc_prefix = ""
            if mltraj_task.calc_dir_suffix is None:
                mltraj_task.calc_dir_suffix = f'{meth}{pref(t)}'
            mltraj_task.target = target
            if isinstance(targets[target],dict):
                mltraj_task.target = targets[target]
            targstr = targets[target]
            if isinstance(targstr,dict):
                targstr = "".join((targstr[p] if p!="diff" else "") for p in targstr)
                targets_dict = targets[target]
            else:
                targets_dict = {target:targets[target]}
            for irs,rs in enumerate(rand_seed):
                mltraj_task.which_trajs = []
                mltraj_task.which_targets = []
                for itarg,targets_dict_key in enumerate(targets_dict.keys()):
                    if targets_dict_key=='diff':
                        continue
                    # Save a task for just using one calculator at a time
                    # Offset trajectory letter by index of entry in rand_seed list,
                    # and by length of rand seed list times index of entry in target list
                    # eg for 3 committee calcs, 2 targets (gs, es1), we should get
                    # trajectories A,B,C for gs and D,E,F for es1
                    mltraj_task.which_trajs.append(chr(ord('A')+irs+itarg*len(rand_seed)))
                    mltraj_task.which_targets.append(targets_dict_key)
                    mltraj_task.snap_wrapper = None
                    taskname = f'{targstr}_{meth}{t}{rs}'
                    mltraj_task.wrapper.train_args.seed = rand_seed[rs]
                    mltraj_task.calc_suffix = f'{meth}{t}{rs}'
                    if snap_wrapper is None:
                        mltraj_task.snap_calc_params = None
                    else:
                        mltraj_task.snap_wrapper = snap_wrapper
                        if snap_targets is not None:
                            if two_targets:
                                snap_targets = [0,1] if target==0 else [1,0]
                            calc_suffix = mltraj_task.calc_suffix
                            if mltraj_task.carved_suffix is not None and mltraj_task.carved_suffix!='':
                                taskname = f"{taskname}_{traj_suffix}_{mltraj_task.carved_suffix}"
                            else:
                                taskname = f"{taskname}_{traj_suffix}"
                            calc_targets = snap_targets
                            mltraj_task.snap_target = snap_targets
                        else:
                            if isinstance(targets[target],dict):
                                calc_targets = targets[target]
                            else:
                                calc_targets = target
                            taskname = taskname + f'x{len(rand_seed)}'
                            calc_suffix = {f'{meth}{t}{rs}':rseed for (rs,rseed) in rand_seed.items()}
                            mltraj_task.snap_target = calc_targets

                        mltraj_task.snap_calc_params = {'target':calc_targets,
                                                        'calc_prefix':'../../',
                                                        'calc_dir_suffix':mltraj_task.calc_dir_suffix,
                                                        'calc_suffix':calc_suffix,
                                                        'calc_seed':mltraj_task.calc_seed}
                    mltraj_task.traj_suffix = f'{mltraj_task.calc_suffix}_{traj_suffix}'
                    new_mltraj_tasks[taskname] = deepcopy(mltraj_task)
    return new_mltraj_tasks

def create_mltest_tasks(test_task:MLTestingTask,train_calcs,seeds,targets,rand_seed,truth,meth,
                        traj_suffixes={},dir_suffixes={},iter_dir_suffixes={},ntraj={},separate_valid=False):
    """
    Returns a dictionary of MLTest tasks, based on an input prototype task supplied by
    the user, for all the required MLTest tasks for an Active Learning task.
    
    Takes lists of calculators, targets, random seeds, and strings stating the ground
    truth method and the ML method
    """
    new_test_tasks = {}
    for target in targets:
        for t in train_calcs:
            # This test uses the calculator directory from the MLTrain task as the traj location
            test_task.traj_suffix = truth
            test_task.calc_prefix = ""
            test_task.calc_dir_suffix = f'{meth}{pref(t)}'
            test_task.which_trajs = list('A')
            targstr = targets[target]
            if isinstance(targstr,dict):
                targstr = "".join((targstr[p] if p!="diff" else "") for p in targstr)
            test_task.traj_prefix = f"{test_task.calc_seed}_{targstr}_{meth}{pref(t)}_test/"
            if isinstance(targets[target],dict):
                test_task.target = targets[target]
            else:
                test_task.target = target
            test_task.traj_links = {}
            test_task.which_trajs = {}
            test_task.which_targets = {}
            test_task.ref_mol_seed_dict = {}
            if separate_valid:
                test_task.traj_links_valid = test_task.traj_links
                test_task.which_trajs_valid = test_task.which_trajs
                test_task.which_targets_valid = test_task.which_targets
            else:
                test_task.traj_links_valid = None
                test_task.which_trajs_valid = None
                test_task.which_targets_valid = None
            add_trajectories(test_task,seeds,t,traj_suffixes,dir_suffixes,ntraj,targets,target,truth)
            # For generations > 0, we now add chosen subset trajectories for active learning
            add_iterating_trajectories(test_task,seeds,t,iter_dir_suffixes,targets,target,meth,truth)
            for rs in rand_seed:
                test_task.wrapper.train_args.seed = rand_seed[rs]
                test_task.calc_suffix = f'{meth}{t}{rs}'
                test_task.plotfile = f'{{solu}}_{{solv}}_{test_task.calc_suffix}.png'
                # Store a test task for evaluating the success of the calculator on its training data
                new_test_tasks[f"{targstr}_{meth}{t}{rs}"] = deepcopy(test_task)
            # Now set up tasks for testing against ground truth results sampled from a specific set of trajectory data
            for tp in train_calcs:
                test_task.traj_links = {}
                test_task.which_trajs = {}
                if separate_valid:
                    test_task.traj_links_valid = test_task.traj_links
                    test_task.which_trajs_valid = test_task.which_trajs
                else:
                    test_task.traj_links_valid = None
                    test_task.which_trajs_valid = None
                only_gen = get_gen_from_calc(tp)
                add_iterating_trajectories(test_task,seeds,t,iter_dir_suffixes,targets,target,meth,truth,only_gen=only_gen)
                for rs in rand_seed:
                    test_task.wrapper.train_args.seed = rand_seed[rs]
                    test_task.calc_suffix = f'{meth}{t}{rs}'
                    test_task.plotfile = f'{{solu}}_{{solv}}_{test_task.calc_suffix}_mltraj_{meth}{tp}.png'
                    new_test_tasks[f"{targstr}_{meth}{t}{rs}_mltraj_{meth}{tp}"] = deepcopy(test_task)
    return new_test_tasks

def create_spectra_tasks(spectra_task:SpectraTask,train_calcs,targets,rand_seed,meth,
                         traj_suffix='specdyn_recalc',corr_traj=False,diff_traj=False,task_suffix=None):
    """
    Returns a dictionary of Spectra tasks, based on an input prototype task supplied by
    the user, for all the required Spectra tasks for an Active Learning task.
    
    Takes lists of calculators, targets, random seeds
    """
    new_spectra_tasks = {}
    task_suffix = traj_suffix if task_suffix is None else task_suffix
    # Loop over target states
    for target in targets:
        targstr = targets[target]
        if isinstance(targstr,dict):
            targstr = "".join((targstr[p] if p!="diff" else "") for p in targstr)
            targets_dict = targets[target]
        else:
            targets_dict = {target:targets[target]}
        # Loop over calculators
        for t in train_calcs:
            for itarg,traj_target in enumerate(targets_dict):
                if traj_target=='diff':
                    continue
                # Find the initial and final target strings
                traj_targstr = targets_dict[traj_target]
                traj_targstrp = "gs" if traj_targstr=="es1" else "es1"
                all_trajs = []
                all_corr_trajs = [] if corr_traj else None
                spectra_task.vibration_trajectory = None
                spectra_task.mode = "absorption" if traj_targstr=="gs" else "emission"
                spectra_task.verbosity = 'normal'
                # Set directory where trajectories will be found
                if hasattr(spectra_task,'exc_dir_suffix'):
                    exc_dir_suffix = spectra_task.exc_dir_suffix
                else:
                    exc_dir_suffix = 'mldyn'
                spectra_task.exc_suffix = f'{targstr}_{meth}{pref(t)}_{exc_dir_suffix}'
                # Set parameters for wrapper
                if spectra_task.wrapper is not None:
                    spectra_task.wrapper.task = spectra_task.mode.upper()
                    if task_suffix is None:
                        task_suffix = "spec"
                    spectra_task.wrapper.rootname = f'{{solu}}_{{solv}}_{traj_targstr}_{task_suffix}'
                    spectra_task.wrapper.input_filename = f'{{solu}}_{{solv}}_{traj_targstr}_{task_suffix}_input'
                # Set output plot file
                spectra_task.output = f'{{solu}}_{{solv}}_{targstr}_{meth}{t}_{task_suffix}.png'
                tdir = '.'
                rslist = list(rand_seed) # ['a','b','c'...]
                # Loop over trajectories to process
                for irs,rs in enumerate(rslist):
                    w = chr(ord('A')+irs+itarg*len(rand_seed))
                    # Add normal and correction trajectories (gs and es for each entry)
                    if not diff_traj:
                        trajtargs = [traj_targstr,traj_targstrp]
                    else:
                        trajtargs = ['esdiff']
                    all_trajs.append([f"{tdir}/{{solu}}_{{solv}}_{trajtarg}_{w}_{meth}{t}{rs}_{traj_suffix}.traj"
                                      for trajtarg in trajtargs])
                    if corr_traj:
                        all_corr_trajs.append([f"{tdir}/{{solu}}_{{solv}}_{trajtarg}_{w}_{meth}{t}{rs}_{traj_suffix}_nosolu.traj"
                                               for trajtarg in trajtargs])
                spectra_task.trajectory = all_trajs
                spectra_task.correction_trajectory = all_corr_trajs
                if isinstance(targets[target],dict):
                    taskname = f'{targstr}_{traj_targstr}_{meth}{t}_{task_suffix}'
                else:
                    taskname = f'{targstr}_{meth}{t}_{task_suffix}'
                new_spectra_tasks[taskname] = deepcopy(spectra_task)
            
    return new_spectra_tasks


def setup_scripts(scriptname,seed,targstr,num_calcs,calc_suffix,method,script_settings,make_sbatch,allseed=None,task_list=None):

    # Store original contents of declarations section
    store_decs = script_settings['declarations']
    if allseed is None:
        allseed = seed
    # append to initial declarations to set up variables for Active Learning loop
    script_settings['declarations'] += f'''
M="{calc_suffix[-1]}"
T="{targstr}"
scr="{scriptname}"
SP="{seed}"
SA="{allseed}"
C="{num_calcs}"
letters=({{a..z}})
W="{method}"

X=$((SLURM_ARRAY_TASK_ID/10))
YP=$((SLURM_ARRAY_TASK_ID%10))
Y=${{letters[$YP]}}
export SLURM_ARRAY_TASK_ID=$YP
echo "X="$X "YP="$YP
    '''

    if task_list is None:
        task_list = ['mltrain','mltraj','mltest','mlfinaltest','specdyn','spectra','cumul_spectra']
    # Write job script for submission to HPC cluster
    for task in task_list:
        # Set up default target and task name
        script_settings['target'] = '$T"_"$W"ac"$X$M$Y'
        script_task = task
        # Set up targets and task names for tasks with different patterns
        if task=="mltraj":
            script_settings['target'] += '"x"$C'
        if task=="specdyn":
            script_settings['target'] += '"_specdyn"'
            script_task = 'mltraj'
        if task=="spectra":
            script_settings['target'] = '$T"_"$W"ac"$X$M"_specdyn_recalc_carved"'
        if task=="cumul_spectra":
            script_settings['target'] = '$T"_"$W"ac"$X$M"_specpycode"'
            script_task = "spectra"
        if task=="mlfinaltest":
            script_settings['target'] += '"_mltraj_"$W"ac2"$M' 
            script_task = 'mltest'
        # Set up where to find executable
        script_settings['scriptname'] = '$scr'
        script_settings['execpath'] = '../'
        # Set up seeds and jobnames in script
        if task=='mltrain' or task=='mltest' or task=='mlfinaltest':
            script_settings['seed'] = '$SA'
            script_settings['jobname'] = f'{allseed}_{targstr}_{calc_suffix}_{task}'
        else:
            script_settings['seed'] = '$SP'
            script_settings['jobname'] = f'{seed}_{targstr}_{calc_suffix}_{task}'
        # Set up name of log file
        if task!='specdyn': # avoid writing '_specdyn_mltraj' as part of log name
            script_settings['postfix'] = f'| tee -a {script_settings["seed"]}"_"{script_settings["target"]}"_{script_task}"$LOGSUFFIX.log'
        else:
            script_settings['postfix'] = f'| tee -a {script_settings["seed"]}"_"{script_settings["target"]}$LOGSUFFIX.log'
        # Write the script
        script_settings['num_threads'] = 1
        make_sbatch(task=script_task,**script_settings)

    # restore previous declarations section
    script_settings['declarations'] = store_decs


def unit_test():
    """
    Unit test for Active Learning
    """
    from types import SimpleNamespace
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    wrapper = SimpleNamespace()
    wrapper.script_settings = {}
    clusters_task = SimpleNamespace()
    clusters_task.script_settings = wrapper.script_settings
    clusters_task.radius = None
    clusters_task.repeat_without_solute = False
    clusters_task.wrapper = wrapper
    clusters_task.subset_selection_nmax = 20
    clusters_task.subset_selection_min_spacing = 20
    train_calcs = ['it0r','it1r','it0s','it1s']
    targets = {0:'gs',1:'es1'}
    rand_seed = {'a':1234}
    meth="MACE"
    truth="orca"
    new_clusters_tasks = create_clusters_tasks(clusters_task,train_calcs,targets,rand_seed,meth,truth)
    pp.pprint('new_clusters_tasks:')
    pp.pprint(new_clusters_tasks)

    training_task = SimpleNamespace()
    training_task.wrapper = wrapper
    training_task.wrapper.train_args = {'max_num_epochs':1000,'start_swa':50}
    traj_suffixes = ["rattled","qmd800"]
    dir_suffixes = {"rattled":"rattled","qmd800":"qmd"}
    ntraj = {}
    ntraj[targets[0],"rattled"] = 1
    ntraj[targets[1],"rattled"] = 0
    ntraj[targets[0],"qmd800"] = 1
    ntraj[targets[1],"qmd800"] = 0
    new_training_tasks = create_mltrain_tasks(training_task,train_calcs,targets,rand_seed,meth,truth,traj_suffixes,dir_suffixes,ntraj)
    print('new_training_tasks:')
    pp.pprint(new_training_tasks)

    mltraj_task = SimpleNamespace()
    md_wrapper = wrapper
    snap_wrapper = wrapper
    mltraj_task.wrapper = wrapper
    new_mltraj_tasks = create_mltraj_tasks(mltraj_task,train_calcs,targets,rand_seed,meth,md_wrapper,snap_wrapper)
    pp.pprint('new_mltraj_tasks:')
    pp.pprint(new_mltraj_tasks)
    
    mltest_task = SimpleNamespace()
    mltest_task.wrapper = wrapper
    new_mltest_tasks = create_mltest_tasks(mltest_task,train_calcs,targets,rand_seed,truth,meth)
    pp.pprint('new_mltest_tasks:')
    pp.pprint(new_mltest_tasks)
    
    spectra_task = SimpleNamespace()
    spectra_task.wrapper = wrapper
    spectra_task.wrapper.num_trajs = 0
    new_spectra_tasks = create_spectra_tasks(mltest_task,train_calcs,targets,rand_seed)
    pp.pprint('new_spectra_tasks:')
    pp.pprint(new_spectra_tasks)
    
do_unit_test = False
if do_unit_test:
    unit_test()





