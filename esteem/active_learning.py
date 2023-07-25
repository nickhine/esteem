#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Defines routines that implement Active Learning by duplicating a prototype task
across an array of iterations, targets, random seeds etc for each of the 4 steps
of an Active Learning cycle: MLTrain, MLTraj, Clusters, MLTest, and spectra tasks
once training is completed.

To use, create a prototype for each, and lists of calculators, targets, random seeds,
and call then call each of the create_* routines to return lists of tasks to
pass to drivers.main()
"""


# In[ ]:


from esteem.trajectories import get_trajectory_list
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

def create_clusters_tasks(task,train_calcs,seed,traj_suffix,md_suffix,
                          md_dir_suffix,targets,rand_seed,meth,truth):
    """
    Returns a dictionary of clusters tasks, based on an input prototype task supplied by
    the user, for all the required clusters tasks for an Active Learning task.
    
    Takes lists of calculators, targets, random seeds, and strings stating the ML method
    and the ground truth method
    """
    
    # By default, the Q trajectory for validation is the same size as the main traj
    init_min_snapshots = task.min_snapshots
    init_max_snapshots = task.max_snapshots
    # It can be overridden by setting valid_snapshots
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
                task.target = list(targets)
                task.exc_suffix = f'{targets[target]}_{meth}{t}_{traj_suffix}'
                task.exc_suffix = f'{targets[target]}_{meth}{t}'
                task.exc_dir_suffix = f'{targets[target]}_{meth}{pref(t)}_{traj_suffix}'
                task.output = f'{truth}_{suff(tp)}'
                task.carved_suffix = f'carved_{suff(tp)}'
                task.selected_suffix = f'selected_{suff(tp)}'
                task.script_settings['logdir'] = task.output
                wlist = [get_traj_from_calc(tp)]
                wlist += ['Q']
                wplist = get_trajectory_list(len(rand_seed))
                rslist = list(rand_seed)
                for iw,w in enumerate(wlist):
                    # for the main trajectory, reset the number of snapshots
                    if iw==0:
                        task.min_snapshots = init_min_snapshots
                        task.max_snapshots = init_max_snapshots
                        task.subset_selection_method = get_ssm_from_traj(w)
                        task.subset_selection_which_traj = w
                    else: # for the validation/testing trajectories, offset the snapshots
                        task.min_snapshots = task.max_snapshots
                        task.max_snapshots = task.max_snapshots + valid_snapshots
                    task.md_prefix = f'{seed}_{targets[target]}_{meth}{pref(tp)}_{md_dir_suffix}'
                    task.md_suffix = [f'{targets[target]}_{wp}_{meth}{tp}{rslist[i]}_{md_suffix}' for i,wp in enumerate(wplist)]
                    # Collapse list if it just contains one entry
                    if len(wplist)==1:
                        task.md_suffix = task.md_suffix[0]
                    task.which_traj = w
                    traj_char = '_'+w #'' if w==t[-1].upper() else '_'+w
                    new_clusters_tasks[task.exc_suffix+traj_char] = deepcopy(task)
                task.subset_selection_method = None
    return new_clusters_tasks


# In[ ]:


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
            targstr1 = targets[target1]
            offset = chr(ord('A')+itarg1-1) if itarg1>0 else ''
            if "qmd" in traj_suffix:
                # For QMD trajectories, the trajectory name depends on the target
                # (separately from the target of the trajectory)
                fullsuffix = f"{targstr1}_{traj_suffix}"
            else:
                fullsuffix = truth
            for seed in seeds:
                all_keys = get_keys(task)
                targstr2 = targstr
                if seed=='{solv}_{solv}' and targstr2=='es1':
                    targstr2 = 'gs'
                for ikey,key in enumerate(all_keys):
                    all_traj = get_trajectory_list(passed[target1]+ntraj[targstr1,traj_suffix])
                    for itraj,traj in enumerate(all_traj[passed[target1]:]):
                        trajsource = all_traj[itraj+ikey*ntraj[targstr1,traj_suffix]]
                        traj_dest = f"{seed}_{dir_suffix}/{seed}_{targstr2}_{trajsource}_{fullsuffix}.traj"
                        if key=='train':
                            task.traj_links[offset+traj] = traj_dest
                            task.which_trajs += [offset+i for i in all_traj[passed[target1]:]]
                            #print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs}')
                        elif key=='valid':
                            task.traj_links_valid[offset+traj] = traj_dest
                            task.which_trajs_valid += [offset+i for i in all_traj[passed[target1]:]]
                            #print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs_valid}')
                        elif key=='test':
                            task.traj_links_test[offset+traj] = traj_dest
                            task.which_trajs_test += [offset+i for i in all_traj[passed[target1]:]]
                            #print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs_test}')
                    passed[target1] += ntraj[targstr1,traj_suffix]
                    if passed[target1] > 26:
                        print('# Warning: more than 26 input trajectories for this target')
                        print('# Please ensure no overlap with other targets:')
                        print(task.which_trajs)

def add_iterating_trajectories(task,seeds,calc,iter_dir_suffixes,targets,target,meth,truth):
    """
    Adds iterating trajectories
    """
    from esteem.tasks.ml_testing import MLTestingTask
    gen = get_gen_from_calc(calc)
    if type(task)==MLTestingTask and gen is not None:
        gen = gen + 1
    all_used_trajs = task.which_trajs.copy()
    if task.which_trajs_valid is not None:
        all_used_trajs += task.which_trajs_valid
    if task.which_trajs_test is not None:
        all_used_trajs += task.which_trajs_test
    last_static_traj_char = sorted(all_used_trajs)[-1]
    if gen is None or gen < 1:
        return
    targstr = targets[target]
    # Loop over generations prior to current
    for g in range(gen):
        calcp = f'{pref(calc)}{g}{calc[-1]}'
        # Use fixed traj_suffix along the lines of "orca_ac9ra" currently - perhaps make templatable?
        traj_suffix = f'{truth}_{suff(calcp)}'
        # Find character for generation: ''=0, 'A'=1, 'B'=2 etc
        gen_char = chr(ord('A')-1+g) if g>0 else ''
        # Loop over all targets for source trajectories
        offset = 1
        for targetp in targets:
            #if targetp > target:
            #    continue
            targstrp = targets[targetp]
            # Loop over all dir suffixes and seeds
            for dir_suffix in iter_dir_suffixes:
                for seed in seeds:
                    # temporary hack - will need a better way to skip this
                    if (seed=='{solv}_{solv}' and targstrp=='es1'):
                        continue
                    targstr2 = targstr
                    if seed=='{solv}_{solv}' and targstr2=='es1':
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
                            task.which_trajs += [f'{gen_char+traj_char}']
                            #print(f'adding: {calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs}')
                        elif key=='valid':
                            task.traj_links_valid[gen_char+traj_char] = traj_dest
                            task.which_trajs_valid += [f'{gen_char}{traj_char}']
                            #print(f'adding: {calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs_valid}')
                        elif key=='test':
                            task.traj_links_test[gen_char+traj_char] = traj_dest
                            task.which_trajs_test += [f'{gen_char}{traj_char}']
                            #print(f'adding: {calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs_test}')
                        offset = offset + 1
                
def create_mltrain_tasks(train_task,train_calcs,seeds,targets,rand_seed,meth,truth,
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
    if 'max_num_epochs' in train_task.wrapper.train_args:
        init_epochs = train_task.wrapper.train_args['max_num_epochs'] # MACE specific
        swa_init_epochs = train_task.wrapper.train_args['start_swa']  # MACE specific

    for target in targets:
        for t in train_calcs:
            # Calculator basic info
            train_task.traj_suffix = truth
            train_task.target = target
            train_task.calc_dir_suffix = f"{meth}{pref(t)}"
            train_task.calc_prefix = ""
            # Set up links to trajectories - first empty the lists
            train_task.traj_links = {}
            train_task.which_trajs = []
            if separate_valid:
                train_task.traj_links_valid = {}
                train_task.which_trajs_valid = []
            else:
                train_task.traj_links_valid = None
                train_task.which_trajs_valid = None
            # Then add "static" configurations, that do not increase with AL generation
            add_trajectories(train_task,seeds,t,traj_suffixes,dir_suffixes,ntraj,targets,target,truth)
            # For generations > 0, we now add chosen subset trajectories for active learning
            add_iterating_trajectories(train_task,seeds,t,iter_dir_suffixes,targets,target,meth,truth)
            # extra epochs for each generation
            if 'max_num_epochs' in train_task.wrapper.train_args:  # MACE specific
                gen = get_gen_from_calc(t)
                train_task.wrapper.train_args['max_num_epochs'] = init_epochs + gen*delta_epochs
                # same number of extra epochs for SWA
                train_task.wrapper.train_args['start_swa'] = swa_init_epochs + gen*delta_epochs
            # Save this calculator to the list for each seed
            for rs in rand_seed:
                # Seed-specific info
                train_task.wrapper.train_args['seed'] = rand_seed[rs] # MACE specific
                train_task.calc_suffix = f"{meth}{t}{rs}"
                new_mltrain_tasks[targets[target]+'_'+train_task.calc_suffix] = deepcopy(train_task)
    return new_mltrain_tasks


# In[ ]:


def create_mltraj_tasks(mltraj_task,train_calcs,targets,rand_seed,meth,md_wrapper,
                        traj_suffix='mldyn',snap_wrapper=None,two_targets=False):
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
        for t in train_calcs:
            mltraj_task.wrapper = md_wrapper
            mltraj_task.calc_prefix = ""
            mltraj_task.calc_dir_suffix = f'{meth}{pref(t)}'
            mltraj_task.target = target
            targstr = targets[target]
            for rs in rand_seed:
                # Save a task for just using one calculator at a time
                mltraj_task.snap_wrapper = None
                taskname = f'{targstr}_{meth}{t}{rs}'
                if snap_wrapper is None:
                    mltraj_task.snap_calc_params = None
                else:
                    targ = target
                    mltraj_task.snap_wrapper = snap_wrapper
                    if two_targets:
                        calc_suffix = mltraj_task.calc_suffix
                        taskname = taskname + '_spec'
                        targ = [0,1] if target==0 else [1,0]
                    else:
                        taskname = taskname + f'x{len(rand_seed)}'
                        calc_suffix = {f'{meth}{t}{rs}':rseed for (rs,rseed) in rand_seed.items()}
                    mltraj_task.snap_calc_params = {'target':targ,
                                                    'calc_prefix':'../../',
                                                    'calc_dir_suffix':mltraj_task.calc_dir_suffix,
                                                    'calc_suffix':calc_suffix,
                                                    'calc_seed':mltraj_task.calc_seed}                    
                mltraj_task.wrapper.train_args['seed'] = rand_seed[rs]
                mltraj_task.calc_suffix = f'{meth}{t}{rs}'
                mltraj_task.traj_suffix = f'{mltraj_task.calc_suffix}_{traj_suffix}'
                new_mltraj_tasks[taskname] = deepcopy(mltraj_task)
    return new_mltraj_tasks


# In[ ]:


def create_mltest_tasks(test_task,train_calcs,seeds,targets,rand_seed,truth,meth,
                        traj_suffixes={},dir_suffixes={},iter_dir_suffixes={},ntraj={}):
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
            test_task.traj_prefix = f"{test_task.calc_seed}_{targets[target]}_{meth}{pref(t)}_test/"
            test_task.target = target
            targstr = targets[target]
            test_task.traj_links = {}
            test_task.which_trajs = []
            add_trajectories(test_task,seeds,t,traj_suffixes,dir_suffixes,ntraj,targets,target,truth)
            # For generations > 0, we now add chosen subset trajectories for active learning
            add_iterating_trajectories(test_task,seeds,t,iter_dir_suffixes,targets,target,meth,truth)
            for rs in rand_seed:
                test_task.wrapper.train_args['seed'] = rand_seed[rs]
                test_task.calc_suffix = f'{meth}{t}{rs}'
                test_task.plotfile = f'{{solu}}_{test_task.calc_suffix}.png'
                # Store a test task for evaluating the success of the calculator on its training data
                new_test_tasks[f"{targets[target]}_{meth}{t}{rs}"] = deepcopy(test_task)
                # Now set up a task for testing against ground truth results sampled from each set of trajectory data
                for itarg1,target1 in enumerate(targets): #targets):
                    mltraj_target = targets[target1]
                    if mltraj_target != "gs":
                        continue
                    for tp in train_calcs:
                        gen = get_gen_from_calc(tp) # which generation is this
                        if gen is None:
                            continue
                        v = chr(ord('A')-1+gen) if gen>0 else ''
                        which_trajs = [v+get_traj_from_calc(tp)]
                        which_trajs += [v+'Q']
                        test_task.traj_prefix = f'{{solu}}_{targets[target]}_{meth}{pref(tp)}/'
                        for w in which_trajs:
                            test_task.which_trajs = [w]
                            traj_suffix = '' if w==tp[-1].upper() else '_'+w[-1]
                            test_task.traj_links[w] = f'{{solu}}_{mltraj_target}_{meth}{pref(tp)}_mlclus/{{solu}}_{targets[target]}_{w[-1]}_{truth}_{suff(tp)}.traj'
                            new_test_tasks[f"{targets[target]}_{meth}{t}{rs}_mltraj_{meth}{tp}{rs}"+traj_suffix] = deepcopy(test_task)
    return new_test_tasks


# In[12]:


def create_spectra_tasks(spectra_task,train_calcs,targets,rand_seed,meth,corr_traj=False):
    """
    Returns a dictionary of Spectra tasks, based on an input prototype task supplied by
    the user, for all the required Spectra tasks for an Active Learning task.
    
    Takes lists of calculators, targets, random seeds
    """
    new_spectra_tasks = {}
    for target in targets:
        targstr = targets[target]
        for t in train_calcs:
            all_trajs = []
            all_corr_trajs = [] if corr_traj else None
            spectra_task.vibration_trajectory = None
            spectra_task.mode = "absorption" if targstr=="gs" else "emission"
            spectra_task.verbosity = 'normal'
            if spectra_task.wrapper is not None:
                spectra_task.wrapper.task = spectra_task.mode.upper()
                spectra_task.wrapper.rootname = f'{{solu}}_{targstr}_spec'
                spectra_task.wrapper.input_filename = f'{{solu}}_{targstr}_spec_input'
            spectra_task.exc_suffix = f'{targstr}_{meth}{pref(t)}_mldyn'
            spectra_task.output = f'{{solu}}_{spectra_task.exc_suffix}_spectrum.png'
            tdir = '.'
            rs = 'a'
            for w in get_trajectory_list(1):
                all_trajs.append([f"{tdir}/{{solu}}_{targstr}_{w}_{meth}{t}{rs}_specdyn.traj"])
                if corr_traj:
                    all_corr_trajs.append([f"{tdir}/{{solu}}_{targstr}_{w}_{meth}{t}{rs}_nosolu.traj"])
                if spectra_task.wrapper is not None:
                    spectra_task.wrapper.num_trajs += 1
            spectra_task.trajectory = all_trajs
            spectra_task.correction_trajectory = all_corr_trajs
            new_spectra_tasks[f'{targstr}_{meth}{t}_specdyn'] = deepcopy(spectra_task)
            
    return new_spectra_tasks


# In[ ]:


def setup_scripts(scriptname,seed,targstr,num_calcs,calc_suffix,method,script_settings,make_sbatch):

    store_decs = script_settings['declarations']
    script_settings['declarations'] += f'''
M="{calc_suffix[-1]}"
T="{targstr}"
scr="{scriptname}"
S="{seed}"
C="{num_calcs}"
letters=({{a..z}})
W="{method}"

X=$((SLURM_ARRAY_TASK_ID/10))
YP=$((SLURM_ARRAY_TASK_ID%10))
Y=${{letters[$YP]}}
export SLURM_ARRAY_TASK_ID=$YP
echo "X="$X "YP="$YP
    '''

    # Write job script for submission to HPC cluster
    for task in ['mltrain','mltraj','mltest']:
        script_settings['jobname'] = f'{seed}_{targstr}_{calc_suffix}_{task}'
        script_settings['target'] = '$T"_"$W"ac"$X$M$Y'
        if task=="mltraj":
            script_settings['target'] += '"x"$C'
        script_settings['scriptname'] = '$scr'
        script_settings['seed'] = '$S'
        script_settings['num_threads'] = 1
        script_settings['postfix'] = f'| tee -a $S"_"{script_settings["target"]}"_"{task}$LOGSUFFIX.log'
        make_sbatch(task=task,**script_settings)

    script_settings['declarations'] = store_decs


# In[1]:


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


# In[ ]:





# In[ ]:




