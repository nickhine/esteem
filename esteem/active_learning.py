#!/usr/bin/env python
# coding: utf-8


"""
Builds the many task objects needed by an Active Learning (AL) workflow for
machine-learned interatomic potentials, by copying a prototype task across
generations, target states and committee members.

.. rubric:: The Active Learning cycle

Each generation of the cycle has four steps, each run as an ESTEEM task:

1. **mltrain** (:func:`create_mltrain_tasks`): train a committee of ML models
   (eg three MACE models with different random seeds) for each target state, on
   the initial ground-truth data plus all the data selected by earlier generations.
2. **mltraj** (:func:`create_mltraj_tasks`): run MD with each committee member,
   with the whole committee evaluating every snapshot, so that the spread of the
   committee's predictions measures the uncertainty.
3. **mlclus** (:func:`create_clusters_tasks`): select snapshots from those
   trajectories, biased towards high committee uncertainty, and label them with
   the ground-truth method (eg DFT and TDDFT with ORCA).
4. **mltest** (:func:`create_mltest_tasks`): test the models against the
   ground-truth data, including the snapshots just selected, which the models
   have not seen.

The next generation is then trained on the enlarged data set. Once training is
complete, :func:`create_spectra_tasks` sets up spectra from MD with the final
models.

.. rubric:: Calculator names

Each AL calculator is named by a string such as ``'MACEac0u'``, given in the list
``train_calcs`` (one entry per generation, eg ``['MACEac0u', 'MACEac1u', 'MACEac2u']``).
The characters have fixed positions:

- ``calc[0:4]`` (``'MACE'``): the ML method.
- ``calc[4:6]`` (``'ac'``): a tag, conventionally ``'ac'``. :func:`pref` returns
  ``calc[0:6]`` (``'MACEac'``), which is shared by all generations and names the
  directories; :func:`suff` returns ``calc[4:]`` (``'ac0u'``), which names the
  ground-truth trajectories of each generation, eg ``'orca_ac0u'``.
- ``calc[6:-1]`` (``'0'``): the generation, returned as an integer by
  :func:`get_gen_from_calc`. Generation 0 is trained on the initial data only.
- ``calc[-1]`` (``'u'``): the selection type, which :func:`get_traj_from_calc`
  maps to an upper-case trajectory label with ``calc_to_traj_map``
  (``'r'``, ``'s'``, ``'t'``, ``'u'`` give ``'R'``, ``'S'``, ``'T'``, ``'U'``).
  :func:`get_ssm_from_traj` maps that label to the clusters task's
  ``subset_selection_method`` with ``traj_to_ssm_map``: ``'R'`` random,
  ``'S'`` highest committee energy spread (``'E'``), ``'T'`` highest dipole
  spread (``'D'``), ``'U'`` random biased towards high energy spread (``'U'``).
  ``'Q'`` labels the separate validation trajectories, selected at random.

Each committee member is a key of the dict ``rand_seed``, eg
``{'a': 123, 'b': 456, 'c': 789}``: the key is appended to the calculator name
(``'MACEac0ua'``) and the value is the random seed for training that member.
The string ``meth`` (often ``''``) is prepended to calculator names, and
``truth`` (eg ``'orca'``) names the ground-truth data.

The states are given by ``targets``, a dict such as ``{0: 'gs', 1: 'es1'}``.
An entry whose value is itself a dict (eg ``'mult': {0: 'gs', 1: 'es1'}``)
describes a multi-head model; its name is the concatenation of the state names
(``'gses1'``), leaving out any ``'diff'`` entry, which names an energy-difference head.
A top-level ``'diff'`` key is skipped by most of the functions.

.. rubric:: Names of tasks, directories and files

With ``meth=''``, ``truth='orca'``, seed ``{solu}_{solv}`` (written ``{seed}``
below), target ``'gs'``, committee members ``a``, ``b``, ``c`` and the calculator
``'MACEac0u'``, the default names are:

- mltrain: tasks ``'gs_MACEac0ua'``, ``'gs_MACEac0ub'``, ``'gs_MACEac0uc'``, all
  training in the directory ``{seed}_gs_MACEac``.
- mltraj: tasks ``'gs_MACEac0uax3'`` etc, writing
  ``{seed}_gs_A_MACEac0ua_mldyn.traj`` (``B`` for member ``b``, and so on) in
  the directory ``{seed}_gs_MACEac_mldyn``.
- mlclus: task ``'gs_MACEac0u_U'``, which selects from the trajectories of all
  the committee members and runs in the directory ``{seed}_gs_MACEac_mlclus``,
  writing ``{seed}_{targ}_U_orca_ac0u.traj`` for each state ``{targ}``.
- mltest: tasks ``'gs_MACEac0ua'`` (training data and this generation's new
  data) and ``'gs_MACEac0ua_mltraj_MACEac1u'`` (only the data selected from the
  MD of the generation-1 models), running in ``{seed}_gs_MACEac_test``.

Trajectory labels in the training directory are ``A``, ``B``, ... for the
initial ("static") data, then the selected data of generation ``g``, with
labels whose first letter encodes ``g`` (no letter for ``g=0``, ``'A'`` for
``g=1``, ``'B'`` for ``g=2``, ...), eg ``'B'``, ``'C'`` from generation 0 and
``'AB'``, ``'AC'`` from generation 1. The prototype's ``ntraj`` must be large
enough to include all these labels (eg 270).

.. rubric:: Usage

Set up a prototype task of each type (wrapper, script settings and
the settings common to all its copies), then call each ``create_*`` function
with the same ``train_calcs``, ``targets``, ``rand_seed``, ``meth`` and ``truth``,
and pass the dicts it returns (merged with any other tasks) to
:func:`esteem.drivers.main`, eg ``all_mltrain_tasks=create_mltrain_tasks(...)``.
Each dict is keyed by task name, which is the target given on the command line,
eg ``python script.py mltrain {solu}_{solv} gs_MACEac0ua``.
The functions modify the prototype as they go and return deep copies of it,
so the prototype is left with the settings of the last task made.
See ``examples/cate_ml.py`` for a complete workflow script.
"""


from esteem.trajectories import get_trajectory_list
from esteem.tasks.clusters import ClustersTask
from esteem.tasks.ml_training import MLTrainingTask
from esteem.tasks.ml_trajectories import MLTrajTask
from esteem.tasks.ml_testing import MLTestingTask
from esteem.tasks.spectra import SpectraTask

from copy import deepcopy

# Mapping of trajectory labels to subset selection methods (can be overridden if needed)
traj_to_ssm_map = {'S': 'E', 'U': 'U', 'T': 'D', 'R': 'R', 'Q': 'R'}
# Mapping of calculator names to trajectory labels (can be overridden if needed)
calc_to_traj_map = {'r': 'R', 's': 'S', 't': 'T', 'u': 'U'}


def get_traj_from_calc(calc):
    """
    Returns the trajectory label for a calculator name, from its last character
    (the selection type) via ``calc_to_traj_map``, eg ``'MACEac0u'`` gives ``'U'``.
    Raises an Exception if the character is not in the map.
    """
    calc = calc[-1]
    if calc not in calc_to_traj_map:
        raise Exception(f'Unknown calculator label {calc}. Expected calculator labels: {calc_to_traj_map}')
    return calc_to_traj_map[calc]


def get_ssm_from_traj(traj):
    """
    Returns the subset selection method (``ClustersTask.subset_selection_method``) for
    a trajectory label, via ``traj_to_ssm_map``, eg ``'U'`` gives ``'U'`` and ``'S'``
    gives ``'E'``. Raises an Exception if the label is not in the map.
    """
    if traj not in traj_to_ssm_map:
        raise Exception(f'Unknown trajectory label {traj}. Expected trajectory labels: {traj_to_ssm_map}')
    return traj_to_ssm_map[traj]


def get_gen_from_calc(calc):
    """
    Returns the generation of a calculator as an integer, from the characters between
    position 6 and the last character, eg ``'MACEac2u'`` gives 2. Returns None if they
    do not form an integer.
    """
    try:
        return int(calc[6:-1])
    except:
        return None


def pref(calc):
    """
    Returns the prefix of a calculator name shared by all its generations, used in
    directory names: the part before the first ``'_'`` if there is one, otherwise the first
    six characters, eg ``'MACEac0u'`` gives ``'MACEac'``.
    """
    if '_' in calc:
        return calc.split('_')[0]
    else:
        return calc[0:6]


def suff(calc):
    """
    Returns the calculator name from its fifth character onwards, eg ``'MACEac0u'`` gives
    ``'ac0u'``. Used in the names of the ground-truth trajectories of each generation, such
    as ``'orca_ac0u'``.
    """
    return calc[4:]


def create_clusters_tasks(task: ClustersTask, train_calcs, seed, traj_suffix, md_suffix,
                          md_dir_suffix, targets, rand_seed, meth, truth,
                          separate_valid=False):
    """
    Returns a dict of clusters tasks, copied from a prototype, that select snapshots from
    the ML MD of each AL generation and label them with the ground-truth method.

    One task is made for each calculator in ``train_calcs`` and each target state of the MD.
    It reads the trajectories written by the tasks of :func:`create_mltraj_tasks` for all
    the committee members, from the directory ``{seed}_{targ}_{meth}{pref(calc)}_{md_dir_suffix}``
    and the files ``{seed}_{targ}_{label}_{meth}{calc}{member}_{md_suffix}.traj``
    (``label`` is ``A``, ``B``, ... for the members), selects snapshots with the subset selection
    method given by the calculator's selection type (see :func:`get_ssm_from_traj`), and runs
    the ground-truth calculations for all the states. With the defaults, the task for
    ``'MACEac0u'`` on ``'gs'`` runs in ``{seed}_gs_MACEac_mlclus`` and writes
    ``{seed}_{state}_U_orca_ac0u.traj`` for each state, which :func:`add_iterating_trajectories`
    links into the training tasks of later generations.

    task: ClustersTask
        Prototype task, with the wrapper, script settings, functional, ``nroots`` and
        subset selection settings (eg ``subset_selection_nmax``, ``subset_selection_bias_beta``)
        set. If ``subset_selection_method`` is None, it is set from the first calculator.
        If ``exc_dir_suffix`` or ``md_dir_suffix`` is set on the prototype, it replaces the
        default directory name described below.
    train_calcs: list of str
        Calculator names, one per generation, eg ``['MACEac0u', 'MACEac1u', 'MACEac2u']``.
    seed: str
        Seed used in the name of the MD directory, eg ``'{solu}_{solv}'``.
    traj_suffix: str
        Name of this set of selected data, usually ``'mlclus'``. It is the suffix of the
        directory in which the tasks run, and any other value is also appended to the output
        names and the task name (``exc_suffix``).
    md_suffix: str
        Suffix of the MD trajectory files after the calculator name, eg ``'mldyn'``, or
        ``'mldyn_recalc_carved'`` for clusters carved from MD in explicit solvent.
    md_dir_suffix: str
        Suffix of the MD directory after ``{meth}{pref(calc)}``, eg ``'mldyn'``.
    targets: dict
        Target states, eg ``{0: 'gs', 1: 'es1'}``. One task is made for MD on each
        state. A dict-valued entry, eg ``{0: 'gs', 1: 'es1'}``, makes one task that reads the MD
        of a multi-head model on all its states.
    rand_seed: dict
        Committee members and their random seeds, eg ``{'a': 123, 'b': 456, 'c': 789}``.
        Only the keys are used here.
    meth: str
        Prefix of the calculator names, eg ``''``.
    truth: str
        Name of the ground-truth method, eg ``'orca'``, used in the output names.
    separate_valid: bool
        If True, also make a task selecting a separate validation set, labelled ``'Q'``,
        from the frames after ``max_snapshots`` (``valid_snapshots`` of them, or as many as
        in the main range if that is None).

    Returns a dict of :class:`~esteem.tasks.clusters.ClustersTask` objects keyed by
    ``'{targ}_{meth}{calc}_{X}'``, eg ``'gs_MACEac0u_U'``, where ``X`` is the trajectory
    label from :func:`get_traj_from_calc` (or ``'Q'`` for the validation task), with
    ``'_{traj_suffix}'`` before ``_{X}`` if ``traj_suffix`` is not ``'mlclus'``. The
    attributes set on each task include:

    - ``exc_suffix``: ``'{targ}_{meth}{calc}'``, used in job names;
    - ``exc_dir_suffix``: ``'{targ}_{meth}{pref(calc)}_{traj_suffix}'``, so that all
      generations share the directory ``{seed}_{targ}_MACEac_mlclus``;
    - ``output``: ``'{truth}_{suff(calc)}'``, eg ``'orca_ac0u'``, also used as
      ``script_settings['logdir']``;
    - ``carved_suffix`` and ``selected_suffix``: ``'carved_{suff(calc)}'`` and
      ``'selected_{suff(calc)}'``;
    - ``md_prefix`` and ``md_suffix``: the MD directory and the list of MD trajectory
      suffixes, one per committee member (a single string if there is only one);
    - ``which_traj`` and ``subset_selection_which_traj``: the trajectory label, eg ``'U'``;
    - ``which_target``: the target state of the MD;
    - ``target``: the states to calculate (all the integer keys of ``targets``);
    - ``min_snapshots`` and ``max_snapshots`` (offset for the validation task).
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
    if not hasattr(task, 'exc_dir_suffix'):
        task.exc_dir_suffix = None
    orig_exc_dir_suffix = task.exc_dir_suffix
    for t in train_calcs:
        for tp in [t]:
            for target in targets:
                targstr = targets[target]
                if isinstance(targstr, dict):
                    targstr = "".join((targstr[p] if p != "diff" else "") for p in targstr)
                if isinstance(targets[target], dict):
                    task.target = [targ for targ in targets[target]]
                    targets_dict = targets[target]
                    target = targets[target]
                else:
                    task.target = [targ for targ in targets if type(targ) is not str]
                    targets_dict = {target: targets[target]}
                task.exc_suffix = f'{targstr}_{meth}{t}'
                task.output = f'{truth}_{suff(tp)}'
                task.carved_suffix = f'carved_{suff(tp)}'
                task.selected_suffix = f'selected_{suff(tp)}'
                if traj_suffix != 'mlclus':
                    task.exc_suffix = f'{task.exc_suffix}_{traj_suffix}'
                    task.output = f'{task.output}_{traj_suffix}'
                    task.carved_suffix = f'{task.carved_suffix}_{traj_suffix}'
                    task.selected_suffix = f'{task.selected_suffix}_{traj_suffix}'
                task.exc_dir_suffix = orig_exc_dir_suffix
                if task.exc_dir_suffix is None:
                    task.exc_dir_suffix = f'{targstr}_{meth}{pref(t)}_{traj_suffix}'
                task.script_settings['logdir'] = task.output
                wlist = [get_traj_from_calc(tp)]
                if separate_valid:
                    wlist += ['Q']
                # Make a list of trajectories to find
                rslist = list(rand_seed)
                for iw, w in enumerate(wlist):
                    # for the main trajectory, reset the number of snapshots
                    if iw == 0:
                        task.min_snapshots = init_min_snapshots
                        task.max_snapshots = init_max_snapshots
                        if task.subset_selection_method is None:
                            task.subset_selection_method = get_ssm_from_traj(w)
                        task.subset_selection_which_traj = w
                    elif separate_valid:  # for the validation/testing trajectories, offset the snapshots
                        task.min_snapshots = task.max_snapshots
                        task.max_snapshots = task.max_snapshots + valid_snapshots
                    if not hasattr(task, 'md_dir_suffix'):
                        task.md_prefix = f'{seed}_{targstr}_{meth}{pref(tp)}_{md_dir_suffix}'
                    else:
                        task.md_prefix = f'{seed}_{targstr}_{task.md_dir_suffix}'
                    task.md_suffix = []
                    # loop over random seeds defining committee calcs
                    for irs, rs in enumerate(rslist):
                        # loop over targets if targets[target] is dict
                        for itarg, targets_dict_key in enumerate(targets_dict.keys()):
                            if targets_dict_key == 'diff':
                                continue
                            # find target of this MD trajectory
                            targstrp = targets_dict[targets_dict_key]
                            wp = chr(ord('A') + irs + itarg * len(rslist))
                            # make suffix for the MD trajectory and append to list
                            mdsuff = f'{targstrp}_{wp}_{meth}{tp}{rs}_{md_suffix}'
                            task.md_suffix.append(mdsuff)
                    # Collapse list if it just contains one entry
                    if len(task.md_suffix) == 1:
                        task.md_suffix = task.md_suffix[0]
                    task.which_traj = w
                    task.which_target = target
                    traj_char = '_' + w  # '' if w==t[-1].upper() else '_'+w
                    new_clusters_tasks[task.exc_suffix + traj_char] = deepcopy(task)
    return new_clusters_tasks


def get_keys(task):
    """
    Returns the data sets a training or testing task has: ``['train']``, plus ``'valid'``
    if ``task.traj_links_valid`` is set and ``'test'`` if ``task.traj_links_test`` is set.
    """
    all_keys = ['train']
    if hasattr(task, 'traj_links_valid') and task.traj_links_valid is not None:
        all_keys += ['valid']
    if hasattr(task, 'traj_links_test') and task.traj_links_test is not None:
        all_keys += ['test']
    return all_keys


def add_trajectories(task, seeds, calc, traj_suffixes, dir_suffixes, ntraj, targets, target, truth):
    """
    Adds the "static" trajectories, which are the same for every generation (eg the initial
    ground-truth data), to the ``traj_links``, ``which_trajs`` and ``which_targets`` of a
    training or testing task (and to their ``_valid`` and ``_test`` versions, see
    :func:`get_keys`). When ``which_trajs`` is a dict (testing tasks), the seed of each
    trajectory is also stored in ``task.ref_mol_seed_dict``.

    For each suffix in ``traj_suffixes``, each source state ``targets[target1]`` (the state
    on which the data was sampled; ``'diff'`` and dict-valued entries are skipped) and each
    seed, ``ntraj[(targets[target1], traj_suffix)]`` trajectories are linked, from the files::

        {seed}_{dir_suffixes[traj_suffix]}/{seed}_{state}_{label}_{truth}.traj

    where ``{state}`` is the state of this task's target (each state, for a multi-head
    target) and ``{label}`` is ``A``, ``B``, ... (the validation and test sets use the labels
    after those of the training set). If ``traj_suffix`` contains ``'qmd'``, the file
    suffix is ``{targets[target1]}_{traj_suffix}`` instead of ``truth``. For the seed
    ``'{solv}_{solv}'`` (solvent in solvent), the ``'gs'`` file is always used.

    In the task, the labels follow on from those already added for the same source state,
    with no prefix for the first source state, ``'A'`` for the second, ``'B'`` for the
    third, and so on.

    task: MLTrainingTask or MLTestingTask
        Task to add the trajectories to. Its ``which_trajs`` (and ``which_targets``) must
        already be a list or dict, and ``traj_links`` a dict.
    seeds: list of str, or dict
        Seeds (eg ``['{solu}_{solv}', '{solv}_{solv}']``) for which data is linked, or a dict
        mapping ``(state name, traj_suffix)`` tuples to such lists.
    calc: str
        Calculator name (not used).
    traj_suffixes: list of str
        Names of the sources of static data, eg ``['orca']``.
    dir_suffixes: dict
        Directory suffix for each entry of ``traj_suffixes``, eg ``{'orca': 'init'}``, so that
        the files are found in ``{seed}_init/``.
    ntraj: dict
        Number of trajectories for each ``(state name, traj_suffix)`` tuple, eg
        ``{('gs', 'orca'): 1, ('es1', 'orca'): 0}``.
    targets: dict
        All the target states, eg ``{0: 'gs', 1: 'es1'}``.
    target:
        Key in ``targets`` of this task's target.
    truth: str
        Name of the ground-truth method, used as the file suffix, eg ``'orca'``.
    """
    # Loop over initial source trajectories
    targstr = targets[target]
    passed = {i: 0 for i in targets}
    for traj_suffix in traj_suffixes:
        dir_suffix = dir_suffixes[traj_suffix]
        # Make a list of the trajectories from this source
        for itarg1, target1 in enumerate(targets):
            if target1 == 'diff':
                continue
            targstr1 = targets[target1]
            if isinstance(targstr1, dict):
                continue
            offset = chr(ord('A') + itarg1 - 1) if itarg1 > 0 else ''
            if "qmd" in traj_suffix:
                # For QMD trajectories, the trajectory name depends on the target
                # (separately from the target of the trajectory)
                fullsuffix = f"{targstr1}_{traj_suffix}"
            else:
                fullsuffix = truth
            # handle case where seeds is a dictionary, and keys are target,suffix tuples
            if (isinstance(seeds, dict)):
                seeds_list = seeds[targstr1, traj_suffix]
            else:
                seeds_list = seeds
            # Loop over seeds
            for seed in seeds_list:
                all_keys = get_keys(task)
                targstr2_dict = targstr
                if not isinstance(targstr2_dict, dict):
                    targstr2_dict = {0: targstr2_dict}
                for targstr2_key in targstr2_dict:
                    if targstr2_key == 'diff':
                        continue
                    targstr2 = targstr2_dict[targstr2_key]
                    targstr2_orig = targstr2
                    if seed == '{solv}_{solv}' and targstr2 != 'gs':
                        targstr2 = 'gs'
                    for ikey, key in enumerate(all_keys):
                        all_traj = get_trajectory_list(passed[target1] + ntraj[targstr1, traj_suffix])
                        for itraj, traj in enumerate(all_traj[passed[target1]:]):
                            trajsource = all_traj[itraj + ikey * ntraj[targstr1, traj_suffix]]
                            traj_dest = f"{seed}_{dir_suffix}/{seed}_{targstr2}_{trajsource}_{fullsuffix}.traj"
                            if key == 'train':
                                task.traj_links[offset + traj] = traj_dest
                                new_trajs = [offset + all_traj[passed[target1]:][itraj]]
                                if isinstance(task.which_trajs, list):
                                    task.which_trajs += new_trajs
                                    task.which_targets += [targstr2_orig]
                                else:
                                    task.which_trajs.update({jtraj: jtraj for jtraj in new_trajs})
                                    task.which_targets.update({jtraj: targstr2_orig for jtraj in new_trajs})
                                    task.ref_mol_seed_dict.update({jtraj: seed for jtraj in new_trajs})
                                # print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs} (seed={seed}, itarg1={itarg1}, itraj={itraj}, passed[{target1}]={passed[target1]}')
                            elif key == 'valid':
                                task.traj_links_valid[offset + traj] = traj_dest
                                new_trajs = [offset + all_traj[passed[target1]:][itraj]]
                                if isinstance(task.which_trajs_valid, list):
                                    task.which_trajs_valid += new_trajs
                                    task.which_targets_valid += [targstr2_orig]
                                else:
                                    task.which_trajs_valid.update({jtraj: jtraj for jtraj in new_trajs})
                                    task.which_targets_valid.update({jtraj: targstr2_orig for jtraj in new_trajs})
                                    task.ref_mol_seed_dict.update({jtraj: seed for jtraj in new_trajs})
                                # print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs_valid}')
                            elif key == 'test':
                                task.traj_links_test[offset + traj] = traj_dest
                                new_trajs = [offset + all_traj[passed[target1]:][itraj]]
                                if isinstance(task.which_trajs_test, list):
                                    task.which_trajs_test += new_trajs
                                else:
                                    task.which_trajs_test.update({jtraj: jtraj for jtraj in new_trajs})
                                    task.which_targets_test.update({jtraj: targstr2_orig for jtraj in new_trajs})
                                    task.ref_mol_seed_dict.update({jtraj: seed for jtraj in new_trajs})
                                # print(f'adding: {calc}.traj_links[{offset+traj}] = {traj_dest} for {key} {task.which_trajs_test}')
                        passed[target1] += ntraj[targstr1, traj_suffix]
                        if passed[target1] > 26:
                            print('# Warning: more than 26 input trajectories for this target')
                            print('# Please ensure no overlap with other targets:')
                            print(task.which_trajs)


def add_iterating_trajectories(task, seeds, calc, iter_dir_suffixes, targets, target, meth, truth, only_gen=None):
    """
    Adds the "iterating" trajectories, which are the ground-truth data selected by the
    clusters tasks of earlier AL generations, to the ``traj_links``, ``which_trajs`` and
    ``which_targets`` of a training or testing task (and to their ``_valid`` and ``_test``
    versions, see :func:`get_keys`). When ``which_trajs`` is a dict (testing tasks), the seed
    of each trajectory is also stored in ``task.ref_mol_seed_dict``.

    The generations used depend on the generation ``gen`` of ``calc``:

    - for a training task, generations 0 to ``gen-1`` (none for generation 0);
    - for an :class:`~esteem.tasks.ml_testing.MLTestingTask`, generations 0 to ``gen``, or
      only generation ``only_gen`` if it is given (none if it is -1).

    For each generation ``g``, each source state ``targets[targetp]`` (the state of the MD;
    ``'diff'`` and dict-valued entries are skipped), each entry of ``iter_dir_suffixes`` and
    each seed, the files linked are::

        {seed}_{targets[targetp]}_{meth}{pref(calc)}_{dir_suffix}/{seed}_{state}_{X}_{truth}_{suff(calcg)}.traj

    where ``calcg`` is ``calc`` with its generation replaced by ``g`` (so the file suffix is
    eg ``'orca_ac0u'``), ``{state}`` is the state of this task's target (each state, for a
    multi-head target), and ``{X}`` is :func:`get_traj_from_calc` of ``calc``, or ``'Q'`` for
    the validation set. These are the files written by the tasks of
    :func:`create_clusters_tasks`. For the seed ``'{solv}_{solv}'``, only data from
    ground-state MD is used, and its ``'gs'`` file.

    The label of each trajectory in the task is a letter for the generation (none for
    ``g=0``, ``'A'`` for ``g=1``, ``'B'`` for ``g=2``, ...) followed by successive letters
    after the last label already used by the static trajectories.

    task: MLTrainingTask or MLTestingTask
        Task to add the trajectories to.
    seeds: str, list of str, or dict
        Seeds for which data is linked, or a dict mapping ``(state name, dir_suffix)``
        tuples to lists of seeds. Overridden by seeds given in ``iter_dir_suffixes``.
    calc: str
        Calculator name of this task, eg ``'MACEac2u'``.
    iter_dir_suffixes: list of str, or dict
        Where to find the selected data. A list such as ``['mlclus']`` gives the
        directory suffix (only its last entry is used). A dict maps an extra file suffix,
        appended to ``{truth}_{suff(calcg)}`` unless it is ``None`` or ``''``, to a directory
        suffix, or to a tuple ``(dir_suffix, seeds)``. This should match the ``traj_suffix``
        given to :func:`create_clusters_tasks`: for ``traj_suffix='mlclus'``, use
        ``['mlclus']``.
    targets: dict
        All the target states, eg ``{0: 'gs', 1: 'es1'}``.
    target:
        Key in ``targets`` of this task's target.
    meth: str
        Prefix of the calculator names, eg ``''``.
    truth: str
        Name of the ground-truth method, eg ``'orca'``.
    only_gen: int or None
        For testing tasks only: use only the data selected in this generation, or none if
        -1. If None, generations 0 to ``gen`` are used.
    """

    gen = get_gen_from_calc(calc)
    genstart = 0
    genend = gen
    if type(task) is MLTestingTask and gen is not None:
        if only_gen is not None:
            if only_gen == -1:
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
    if len(all_used_trajs) > 0:
        last_static_traj_char = sorted(all_used_trajs)[-1]
    else:
        last_static_traj_char = chr(ord('A') - 1)
    if gen is None or (gen < 1 and type(task) is not MLTestingTask):
        return
    targstr = targets[target]
    # Loop over generations prior to current
    for g in range(genstart, genend):
        calcp = f'{pref(calc)}{g}{calc[-1]}'
        # Use fixed traj_suffix along the lines of "orca_ac0u" currently - perhaps make templatable?
        traj_suffix = f'{truth}_{suff(calcp)}'
        if isinstance(iter_dir_suffixes, list):
            iter_dir_suffixes_dict = {traj_suffix: dir_suffix for dir_suffix in iter_dir_suffixes}
        else:
            iter_dir_suffixes_dict = {}
            for traj_suffix in iter_dir_suffixes:
                traj_suffix_new = f'{truth}_{suff(calcp)}'
                if traj_suffix is not None and traj_suffix != "":
                    traj_suffix_new = f'{traj_suffix_new}_{traj_suffix}'
                iter_dir_suffixes_dict[traj_suffix_new] = iter_dir_suffixes[traj_suffix]
        # Find character for generation: ''=0, 'A'=1, 'B'=2 etc
        gen_char = chr(ord('A') - 1 + g) if g > 0 else ''
        # Loop over all targets for source trajectories
        offset = 1
        for targetp in targets:
            # if targetp > target:
            #    continue
            if targetp == 'diff':
                continue
            targstrp = targets[targetp]
            if isinstance(targstrp, dict):
                continue
            # Loop over all dir suffixes and seeds
            for traj_suffix in iter_dir_suffixes_dict:
                ids_value = iter_dir_suffixes_dict[traj_suffix]
                if isinstance(ids_value, tuple):
                    dir_suffix, traj_seeds = ids_value[0:2]
                else:
                    dir_suffix = ids_value
                    traj_seeds = seeds
                # handle case where seeds is a dictionary, and keys are target,suffix tuples
                if (isinstance(traj_seeds, dict)):
                    seeds_list = traj_seeds[str(targstrp), str(dir_suffix)]
                elif (isinstance(traj_seeds, list)):
                    seeds_list = traj_seeds
                else:
                    seeds_list = [traj_seeds]
                # Loop over seeds
                for seed in seeds_list:
                    targstr2_dict = targstr
                    if not isinstance(targstr2_dict, dict):
                        targstr2_dict = {0: targstr2_dict}
                    for targstr2_key in targstr2_dict:
                        if targstr2_key == 'diff':
                            continue
                        targstr2 = targstr2_dict[targstr2_key]
                        targstr2_orig = targstr2
                        if seed == '{solv}_{solv}':
                            if targstrp != 'gs':
                                continue
                            if targstr2 != 'gs':
                                targstr2 = 'gs'
                        all_keys = get_keys(task)
                        for ikey, key in enumerate(all_keys):
                            # First get the base character for this type of trajectory
                            traj_type_char = get_traj_from_calc(calc)
                            if (ikey == 1):
                                traj_type_char = 'Q'
                            # Offset first available char by the number of previous trajectories passed
                            traj_char = chr(ord(last_static_traj_char) + offset)
                            # Find the directory and filename for this trajectory
                            traj_link_dir = f"{seed}_{targstrp}_{meth}{pref(calcp)}_{dir_suffix}"
                            traj_link_file = f"{seed}_{targstr2}_{traj_type_char}_{traj_suffix}.traj"
                            # Add it to the list of links to make
                            # and to the list of trajectory characters to link
                            traj_dest = f"{traj_link_dir}/{traj_link_file}"
                            if key == 'train':
                                task.traj_links[gen_char + traj_char] = traj_dest
                                new_traj = f'{gen_char + traj_char}'
                                if isinstance(task.which_trajs, list):
                                    task.which_trajs += [new_traj]
                                    task.which_targets += [targstr2_orig]
                                else:
                                    task.which_trajs[new_traj] = new_traj
                                    task.which_targets[new_traj] = targstr2_orig
                                    task.ref_mol_seed_dict[new_traj] = seed
                                # print(f'adding for {calc}: {targstr}_{calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs}')
                            elif key == 'valid':
                                task.traj_links_valid[gen_char + traj_char] = traj_dest
                                new_traj = f'{gen_char}{traj_char}'
                                if isinstance(task.which_trajs_valid, list):
                                    task.which_trajs_valid += [new_traj]
                                    task.which_targets_valid += [targstr2_orig]
                                else:
                                    task.which_trajs_valid[new_traj] = new_traj
                                    task.which_targets_valid[new_traj] = targstr2_orig
                                    task.ref_mol_seed_dict[new_traj] = seed
                                # print(f'adding for {calc}: {targstr}_{calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs_valid}')
                            elif key == 'test':
                                task.traj_links_test[gen_char + traj_char] = traj_dest
                                new_traj = f'{gen_char}{traj_char}'
                                if isinstance(task.which_trajs_test, list):
                                    task.which_trajs_test += [new_traj]
                                    # task.which_targets_test += [targstr2_orig]
                                else:
                                    task.which_trajs_test[new_traj] = new_traj
                                    # task.which_targets_test[new_traj] = targstr2_orig
                                    task.ref_mol_seed_dict[new_traj] = seed
                                # print(f'adding: {targstr}_{calc}.traj_links[{gen_char+traj_char}] = {traj_dest} for {key} {task.which_trajs_test}')
                            offset = offset + 1


def create_mltrain_tasks(train_task: MLTrainingTask, train_calcs, seeds, targets, rand_seed, meth, truth,
                         traj_suffixes=[], dir_suffixes={}, ntraj={},
                         iter_dir_suffixes=[], delta_epochs=200, separate_valid=False):
    """
    Returns a dict of ML training tasks, copied from a prototype, for every target state,
    calculator (generation) and committee member.

    Each task trains on the static data given by ``traj_suffixes``, ``dir_suffixes`` and
    ``ntraj`` (see :func:`add_trajectories`), plus, from generation 1 on, the data selected
    by the clusters tasks of all earlier generations (see :func:`add_iterating_trajectories`).
    All the committee members and generations for a target share the training directory
    ``{seed}_{targ}_{meth}{pref(calc)}``, eg ``{solu}_{solv}_gs_MACEac``.

    train_task: MLTrainingTask
        Prototype task, with the wrapper (whose ``train_args`` are set, eg
        ``max_num_epochs`` and ``start_swa`` for MACE), script settings, ``geom_prefix`` and
        ``ntraj`` set. ``ntraj`` must be large enough to include every trajectory label
        used (eg 270).
    train_calcs: list of str
        Calculator names, one per generation, eg ``['MACEac0u', 'MACEac1u', 'MACEac2u']``.
    seeds: list of str, or dict
        Seeds whose data is used for training, eg ``['{solu}_{solv}']``, or
        ``['cate_{solv}', '{solv}_{solv}']`` to train one calculator on several systems.
        May be a dict mapping ``(state name, suffix)`` tuples to lists of seeds.
    targets: dict
        Target states, eg ``{0: 'gs', 1: 'es1'}``. A dict-valued entry (eg
        ``'mult': {0: 'gs', 1: 'es1'}``) makes multi-head tasks; ``'diff'`` keys are skipped.
    rand_seed: dict
        Committee members and their random seeds, eg ``{'a': 123, 'b': 456, 'c': 789}``.
    meth: str
        Prefix of the calculator names, eg ``''``.
    truth: str
        Name of the ground-truth method, eg ``'orca'``, used as ``traj_suffix`` for the
        trajectory files in the training directory.
    traj_suffixes: list of str
        Names of the sources of static data, eg ``['orca']``.
    dir_suffixes: dict
        Directory suffix for each entry of ``traj_suffixes``, eg ``{'orca': 'init'}`` for
        data in ``{seed}_init/``.
    ntraj: dict
        Number of static trajectories for each ``(state name, traj_suffix)`` tuple, where
        the state is the one on which the data was sampled, eg
        ``{('gs', 'orca'): 1, ('es1', 'orca'): 0}``.
    iter_dir_suffixes: list of str, or dict
        Where to find the data selected by earlier generations, usually ``['mlclus']``
        (see :func:`add_iterating_trajectories`).
    delta_epochs: int
        For wrappers whose ``train_args`` have ``max_num_epochs`` (MACE), the number of
        epochs added to ``max_num_epochs`` and to ``start_swa`` for each generation.
    separate_valid: bool
        If True, set up separate validation sets (``traj_links_valid``,
        ``which_trajs_valid``), using the ``'Q'`` trajectories of the clusters tasks.
        Otherwise validation data is drawn at random from the training data.

    Returns a dict of :class:`~esteem.tasks.ml_training.MLTrainingTask` objects keyed by
    ``'{targ}_{meth}{calc}{member}'``, eg ``'gs_MACEac0ua'``. The attributes set on each
    task include ``target``, ``calc_suffix`` (``'{meth}{calc}{member}'``),
    ``calc_dir_suffix`` (``'{meth}{pref(calc)}'``), ``calc_prefix`` (``''``),
    ``traj_suffix`` (``truth``), ``traj_links``, ``which_trajs`` and ``which_targets`` (and
    their ``_valid`` versions), ``rand_seed`` and ``wrapper.train_args.seed`` (the member's
    random seed), and for MACE ``max_num_epochs`` and ``start_swa``.
    """

    new_mltrain_tasks = {}
    if hasattr(train_task.wrapper.train_args, 'max_num_epochs'):
        init_epochs = train_task.wrapper.train_args.max_num_epochs  # MACE specific
        swa_init_epochs = train_task.wrapper.train_args.start_swa  # MACE specific
        if swa_init_epochs is None:
            swa_init_epochs = 100000

    for target in targets:
        if target == 'diff':
            continue
        for t in train_calcs:
            # Calculator basic info
            train_task.traj_suffix = truth
            if isinstance(targets[target], dict):
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
            add_trajectories(train_task, seeds, t, traj_suffixes, dir_suffixes, ntraj, targets, target, truth)
            # For generations > 0, we now add chosen subset trajectories for active learning
            add_iterating_trajectories(train_task, seeds, t, iter_dir_suffixes, targets, target, meth, truth)
            # extra epochs for each generation
            if hasattr(train_task.wrapper.train_args, 'max_num_epochs'):  # MACE specific
                gen = get_gen_from_calc(t)
                if gen is not None:
                    train_task.wrapper.train_args.max_num_epochs = init_epochs + gen * delta_epochs
                    # same number of extra epochs for SWA
                    train_task.wrapper.train_args.start_swa = swa_init_epochs + gen * delta_epochs
            # Save this calculator to the list for each seed
            for rs in rand_seed:
                # Seed-specific info
                train_task.rand_seed = rand_seed[rs]
                train_task.wrapper.train_args.seed = rand_seed[rs]  # MACE specific
                train_task.calc_suffix = f"{meth}{t}{rs}"
                targstr = targets[target]
                if isinstance(targstr, dict):
                    targstr = "".join((targstr[p] if p != "diff" else "") for p in targstr)
                new_mltrain_tasks[targstr + '_' + train_task.calc_suffix] = deepcopy(train_task)
    return new_mltrain_tasks


def create_mltraj_tasks(mltraj_task: MLTrajTask, train_calcs, targets, rand_seed, meth, md_wrapper,
                        traj_suffix='mldyn', snap_wrapper=None, two_targets=False, snap_targets=None):
    """
    Returns a dict of ML trajectory tasks, copied from a prototype, that run MD driven by
    each committee member of each calculator (generation), for each target state.

    The trajectory of committee member number ``i`` (counting from 0) on state number ``j``
    of the target is labelled ``chr(ord('A') + i + j*len(rand_seed))``, so with three members
    it is ``A``, ``B`` or ``C`` (and ``D``, ``E``, ``F`` for the second state of a
    multi-head target). With the default ``traj_dir_suffix``, the trajectories are written to
    ``{seed}_{targ}_{meth}{pref(calc)}_mldyn/{seed}_{targ}_{label}_{meth}{calc}{member}_{traj_suffix}.traj``,
    eg ``{solu}_{solv}_gs_MACEac_mldyn/{solu}_{solv}_gs_A_MACEac0ua_mldyn.traj``, where the
    tasks of :func:`create_clusters_tasks` and :func:`create_spectra_tasks` look for them.

    mltraj_task: MLTrajTask
        Prototype task, with the MD settings (eg ``md_steps``, ``nsnap``, ``md_timestep``),
        ``geom_prefix``, ``md_init_traj_link`` and ``ntraj`` (the committee size) set. If
        ``calc_seed`` is None it is set to ``'{solu}'``, and if ``calc_dir_suffix`` is None it
        is set to ``'{meth}{pref(calc)}'`` from the first calculator.
    train_calcs: list of str
        Calculator names, one per generation, eg ``['MACEac0u', 'MACEac1u', 'MACEac2u']``.
    targets: dict
        Target states, eg ``{0: 'gs', 1: 'es1'}``. A dict-valued entry makes tasks for a
        multi-head calculator that run MD on each of its states; ``'diff'`` keys are skipped.
    rand_seed: dict
        Committee members and their random seeds, eg ``{'a': 123, 'b': 456, 'c': 789}``.
    meth: str
        Prefix of the calculator names, eg ``''``.
    md_wrapper:
        Wrapper for the calculator that drives the MD, eg a ``MACEWrapper``.
    traj_suffix: str
        Suffix of the trajectory files after the calculator name, eg ``'mldyn'``, or
        ``'specdyn'`` for longer runs for spectra.
    snap_wrapper:
        Wrapper for the calculators that evaluate each snapshot, or None for no
        evaluation. Without ``snap_targets``, the whole committee evaluates each snapshot
        on the MD state, giving the spread of predictions used by the clusters tasks.
    two_targets: bool
        Used only if ``snap_targets`` is given: if True, ``snap_targets`` is replaced by
        ``[0, 1]`` for target 0 and ``[1, 0]`` otherwise, so that the ground state and the
        first excited state are both evaluated, the MD state first.
    snap_targets: list or None
        If given (with ``snap_wrapper``), the states evaluated at each snapshot by the
        calculator driving the MD alone, instead of by the committee.

    Returns a dict of :class:`~esteem.tasks.ml_trajectories.MLTrajTask` objects. The keys are:

    - ``'{targ}_{meth}{calc}{member}x{n}'``, eg ``'gs_MACEac0uax3'``, where ``n`` is the
      committee size, when ``snap_wrapper`` is given without ``snap_targets``;
    - ``'{targ}_{meth}{calc}{member}_{traj_suffix}'`` (followed by ``'_{carved_suffix}'`` if
      the prototype's ``carved_suffix`` is set), eg ``'gs_MACEac0ua_specdyn'``, when both are given;
    - ``'{targ}_{meth}{calc}{member}'`` when ``snap_wrapper`` is None.

    The attributes set on each task include ``wrapper`` (``md_wrapper``, with
    ``train_args.seed`` set to the member's random seed), ``target``, ``calc_suffix``
    (``'{meth}{calc}{member}'``), ``calc_dir_suffix``, ``calc_prefix`` (``''``),
    ``traj_suffix`` (``'{calc_suffix}_{traj_suffix}'``), ``which_trajs`` and
    ``which_targets``, ``snap_wrapper``, ``snap_target``, and ``snap_calc_params``, which
    locates the evaluating calculators: all the committee members (a dict of calculator
    suffixes and random seeds), or only the driving calculator if ``snap_targets`` is given.
    """

    new_mltraj_tasks = {}
    if mltraj_task.calc_seed is None:
        mltraj_task.calc_seed = f"{{solu}}"  # _{{solv}}"
    for target in targets:
        if target == 'diff':
            continue
        for t in train_calcs:
            mltraj_task.wrapper = md_wrapper
            mltraj_task.calc_prefix = ""
            if mltraj_task.calc_dir_suffix is None:
                mltraj_task.calc_dir_suffix = f'{meth}{pref(t)}'
            mltraj_task.target = target
            if isinstance(targets[target], dict):
                mltraj_task.target = targets[target]
            targstr = targets[target]
            if isinstance(targstr, dict):
                targstr = "".join((targstr[p] if p != "diff" else "") for p in targstr)
                targets_dict = targets[target]
            else:
                targets_dict = {target: targets[target]}
            for irs, rs in enumerate(rand_seed):
                mltraj_task.which_trajs = []
                mltraj_task.which_targets = []
                for itarg, targets_dict_key in enumerate(targets_dict.keys()):
                    if targets_dict_key == 'diff':
                        continue
                    # Save a task for just using one calculator at a time
                    # Offset trajectory letter by index of entry in rand_seed list,
                    # and by length of rand seed list times index of entry in target list
                    # eg for 3 committee calcs, 2 targets (gs, es1), we should get
                    # trajectories A,B,C for gs and D,E,F for es1
                    mltraj_task.which_trajs.append(chr(ord('A') + irs + itarg * len(rand_seed)))
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
                                snap_targets = [0, 1] if target == 0 else [1, 0]
                            calc_suffix = mltraj_task.calc_suffix
                            if mltraj_task.carved_suffix is not None and mltraj_task.carved_suffix != '':
                                taskname = f"{taskname}_{traj_suffix}_{mltraj_task.carved_suffix}"
                            else:
                                taskname = f"{taskname}_{traj_suffix}"
                            calc_targets = snap_targets
                            mltraj_task.snap_target = snap_targets
                        else:
                            if isinstance(targets[target], dict):
                                calc_targets = targets[target]
                            else:
                                calc_targets = target
                            taskname = taskname + f'x{len(rand_seed)}'
                            calc_suffix = {f'{meth}{t}{rs}': rseed for (rs, rseed) in rand_seed.items()}
                            mltraj_task.snap_target = calc_targets

                        mltraj_task.snap_calc_params = {'target': calc_targets,
                                                        'calc_prefix': '../../',
                                                        'calc_dir_suffix': mltraj_task.calc_dir_suffix,
                                                        'calc_suffix': calc_suffix,
                                                        'calc_seed': mltraj_task.calc_seed}
                    mltraj_task.traj_suffix = f'{mltraj_task.calc_suffix}_{traj_suffix}'
                    new_mltraj_tasks[taskname] = deepcopy(mltraj_task)
    return new_mltraj_tasks


def create_mltest_tasks(test_task: MLTestingTask, train_calcs, seeds, targets, rand_seed, truth, meth,
                        traj_suffixes={}, dir_suffixes={}, iter_dir_suffixes={}, ntraj={}, separate_valid=False):
    """
    Returns a dict of ML testing tasks, copied from a prototype, that compare each
    committee member of each calculator (generation) with the ground-truth data.

    Two kinds of task are made for each target state, calculator ``calc`` and committee
    member:

    - a test against the static data (as for training, see :func:`add_trajectories`) and
      the data selected by the clusters tasks of generations 0 to that of ``calc``. The data
      of the calculator's own generation was selected from its own MD, after it was trained,
      so it is a genuine test;
    - for each calculator ``calc2`` in ``train_calcs``, a test against only the data
      selected from the MD of generation ``calc2`` (see the ``only_gen`` argument of
      :func:`add_iterating_trajectories`).

    The tests run in ``{calc_seed}_{targ}_{meth}{pref(calc)}_test``, where links to the
    trajectories are made (``traj_prefix``).

    test_task: MLTestingTask
        Prototype task, with the wrapper, script settings, ``calc_seed`` (eg
        ``'{solu}_{solv}'``), ``ref_mol_dir`` and ``ntraj`` set. ``ntraj`` must be large
        enough to include every trajectory label used.
    train_calcs: list of str
        Calculator names, one per generation, eg ``['MACEac0u', 'MACEac1u', 'MACEac2u']``.
    seeds: list of str, or dict
        Seeds whose data is tested against, as for :func:`create_mltrain_tasks`.
    targets: dict
        Target states, eg ``{0: 'gs', 1: 'es1'}``. A dict-valued entry makes tests of a
        multi-head calculator. Unlike the other functions, a ``'diff'`` key is not skipped.
    rand_seed: dict
        Committee members and their random seeds, eg ``{'a': 123, 'b': 456, 'c': 789}``.
    truth: str
        Name of the ground-truth method, eg ``'orca'``, used as ``traj_suffix``.
    meth: str
        Prefix of the calculator names, eg ``''``.
    traj_suffixes: list of str
        Names of the sources of static data, eg ``['orca']``, as for
        :func:`create_mltrain_tasks`.
    dir_suffixes: dict
        Directory suffix for each entry of ``traj_suffixes``, eg ``{'orca': 'init'}``.
    iter_dir_suffixes: list of str, or dict
        Where to find the data selected by the clusters tasks, usually ``['mlclus']``.
    ntraj: dict
        Number of static trajectories for each ``(state name, traj_suffix)`` tuple, eg
        ``{('gs', 'orca'): 1, ('es1', 'orca'): 0}``.
    separate_valid: bool
        If True, ``traj_links_valid``, ``which_trajs_valid`` and ``which_targets_valid`` are
        the same objects as ``traj_links``, ``which_trajs`` and ``which_targets``, so the
        validation data (eg the ``'Q'`` trajectories) is added to the data tested against.

    Returns a dict of :class:`~esteem.tasks.ml_testing.MLTestingTask` objects, keyed by
    ``'{targ}_{meth}{calc}{member}'`` (eg ``'gs_MACEac0ua'``) for the first kind of test,
    and ``'{targ}_{meth}{calc}{member}_mltraj_{meth}{calc2}'`` (eg
    ``'gs_MACEac2ua_mltraj_MACEac1u'``) for the second. The attributes set on each task
    include ``target``, ``calc_suffix`` (``'{meth}{calc}{member}'``), ``calc_dir_suffix``
    (``'{meth}{pref(calc)}'``), ``calc_prefix`` (``''``), ``traj_suffix`` (``truth``),
    ``traj_prefix``, ``traj_links``, ``which_trajs`` and ``which_targets`` (dicts, keyed by
    trajectory label), ``ref_mol_seed_dict``, ``wrapper.train_args.seed``, and
    ``plotfile``: ``'{solu}_{solv}_{calc_suffix}.png'``, or
    ``'{solu}_{solv}_{calc_suffix}_mltraj_{meth}{calc2}.png'``.
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
            if isinstance(targstr, dict):
                targstr = "".join((targstr[p] if p != "diff" else "") for p in targstr)
            test_task.traj_prefix = f"{test_task.calc_seed}_{targstr}_{meth}{pref(t)}_test/"
            if isinstance(targets[target], dict):
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
            add_trajectories(test_task, seeds, t, traj_suffixes, dir_suffixes, ntraj, targets, target, truth)
            # For generations > 0, we now add chosen subset trajectories for active learning
            add_iterating_trajectories(test_task, seeds, t, iter_dir_suffixes, targets, target, meth, truth)
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
                add_iterating_trajectories(test_task, seeds, t, iter_dir_suffixes, targets, target, meth, truth, only_gen=only_gen)
                for rs in rand_seed:
                    test_task.wrapper.train_args.seed = rand_seed[rs]
                    test_task.calc_suffix = f'{meth}{t}{rs}'
                    test_task.plotfile = f'{{solu}}_{{solv}}_{test_task.calc_suffix}_mltraj_{meth}{tp}.png'
                    new_test_tasks[f"{targstr}_{meth}{t}{rs}_mltraj_{meth}{tp}"] = deepcopy(test_task)
    return new_test_tasks


def create_spectra_tasks(spectra_task: SpectraTask, train_calcs, targets, rand_seed, meth,
                         traj_suffix='specdyn_recalc', corr_traj=False, diff_traj=False, task_suffix=None):
    """
    Returns a dict of spectra tasks, copied from a prototype, that compute spectra from the
    MD of each calculator, combining the trajectories of all its committee members.

    For each committee member, the trajectory set is a pair of files written by the
    tasks of :func:`create_mltraj_tasks`, both with the member's trajectory label ``{w}``
    (``A``, ``B``, ... as in :func:`create_mltraj_tasks`)::

        ./{solu}_{solv}_{state}_{w}_{meth}{calc}{member}_{traj_suffix}.traj
        ./{solu}_{solv}_{other}_{w}_{meth}{calc}{member}_{traj_suffix}.traj

    where ``{state}`` is the state of the MD and ``{other}`` is ``'es1'`` if it is
    ``'gs'``, and ``'gs'`` otherwise. MD on ``'gs'`` gives an absorption spectrum and MD on
    an excited state an emission spectrum. The files are read in the MD directory, set by
    ``exc_suffix``.

    spectra_task: SpectraTask
        Prototype task, with the broadening, wavelength range and other spectrum settings.
        If it has an ``exc_dir_suffix`` attribute, that replaces ``'mldyn'`` in the
        directory name. If its ``wrapper`` is not None, the wrapper's ``task``,
        ``rootname`` and ``input_filename`` are set.
    train_calcs: list of str
        Calculator names, eg ``['MACEac2u']`` for the final generation.
    targets: dict
        Target states, eg ``{0: 'gs', 1: 'es1'}``. A dict-valued entry makes a task for MD
        on each state of a multi-head calculator.
    rand_seed: dict
        Committee members, eg ``{'a': 123, 'b': 456, 'c': 789}``. Only the keys are used.
    meth: str
        Prefix of the calculator names, eg ``''``.
    traj_suffix: str
        Suffix of the MD trajectory files after the calculator name, eg
        ``'specdyn_recalc'`` or ``'specdyn_recalc_carved'``.
    corr_traj: bool
        If True, also set ``correction_trajectory``, from the same files with ``'_nosolu'``
        appended (the clusters without the solute).
    diff_traj: bool
        If True, each trajectory set is a single file whose state is ``'esdiff'`` instead of
        the pair of files.
    task_suffix: str or None
        Suffix of the task names and output files. If None, ``traj_suffix`` is used.

    Returns a dict of :class:`~esteem.tasks.spectra.SpectraTask` objects keyed by
    ``'{targ}_{meth}{calc}_{task_suffix}'``, eg ``'gs_MACEac2u_specdyn_recalc'``, or
    ``'{targ}_{state}_{meth}{calc}_{task_suffix}'`` for each state of a multi-head target.
    The attributes set on each task include ``mode``, ``trajectory``,
    ``correction_trajectory``, ``vibration_trajectory`` (None), ``verbosity``
    (``'normal'``), ``exc_suffix`` (``'{targ}_{meth}{pref(calc)}_mldyn'``) and
    ``output`` (``'{solu}_{solv}_{targ}_{meth}{calc}_{task_suffix}.png'``).
    """
    new_spectra_tasks = {}
    task_suffix = traj_suffix if task_suffix is None else task_suffix
    # Loop over target states
    for target in targets:
        targstr = targets[target]
        if isinstance(targstr, dict):
            targstr = "".join((targstr[p] if p != "diff" else "") for p in targstr)
            targets_dict = targets[target]
        else:
            targets_dict = {target: targets[target]}
        # Loop over calculators
        for t in train_calcs:
            for itarg, traj_target in enumerate(targets_dict):
                if traj_target == 'diff':
                    continue
                # Find the initial and final target strings
                traj_targstr = targets_dict[traj_target]
                traj_targstrp = "gs" if traj_targstr == "es1" else "es1"
                all_trajs = []
                all_corr_trajs = [] if corr_traj else None
                spectra_task.vibration_trajectory = None
                spectra_task.mode = "absorption" if traj_targstr == "gs" else "emission"
                spectra_task.verbosity = 'normal'
                # Set directory where trajectories will be found
                if hasattr(spectra_task, 'exc_dir_suffix'):
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
                rslist = list(rand_seed)  # ['a','b','c'...]
                # Loop over trajectories to process
                for irs, rs in enumerate(rslist):
                    w = chr(ord('A') + irs + itarg * len(rand_seed))
                    # Add normal and correction trajectories (gs and es for each entry)
                    if not diff_traj:
                        trajtargs = [traj_targstr, traj_targstrp]
                    else:
                        trajtargs = ['esdiff']
                    all_trajs.append([f"{tdir}/{{solu}}_{{solv}}_{trajtarg}_{w}_{meth}{t}{rs}_{traj_suffix}.traj"
                                      for trajtarg in trajtargs])
                    if corr_traj:
                        all_corr_trajs.append([f"{tdir}/{{solu}}_{{solv}}_{trajtarg}_{w}_{meth}{t}{rs}_{traj_suffix}_nosolu.traj"
                                               for trajtarg in trajtargs])
                spectra_task.trajectory = all_trajs
                spectra_task.correction_trajectory = all_corr_trajs
                if isinstance(targets[target], dict):
                    taskname = f'{targstr}_{traj_targstr}_{meth}{t}_{task_suffix}'
                else:
                    taskname = f'{targstr}_{meth}{t}_{task_suffix}'
                new_spectra_tasks[taskname] = deepcopy(spectra_task)

    return new_spectra_tasks


def setup_scripts(scriptname, seed, targstr, num_calcs, calc_suffix, method, script_settings, make_sbatch, allseed=None, task_list=None):
    """
    Writes SLURM job scripts, one per task in ``task_list``, that run the AL tasks as array
    jobs. Each array task ID encodes the generation as its tens digit and the committee
    member as its units digit (0 for ``a``, 1 for ``b``, ...), so the target run is eg
    ``'{targstr}_{method}ac{generation}{selection}{member}'``, eg ``'gs_MACEac1ub'`` for ID 11.

    scriptname: str
        Name of the workflow script, without ``'.py'``.
    seed: str
        Seed for the MD and spectra tasks, eg ``'cate_cycl'``.
    targstr: str
        Target state, eg ``'gs'``.
    num_calcs: int
        Committee size, used in the names of the mltraj tasks (``'x3'``).
    calc_suffix: str
        Calculator name, eg ``'MACEac0u'``; its last character is the selection type, and it
        is used in job names.
    method: str
        ML method at the start of the calculator names, eg ``'MACE'``. The tag ``'ac'`` is
        assumed to follow it.
    script_settings: dict
        Settings passed to ``make_sbatch``. Its ``'declarations'`` entry is extended while
        the scripts are written and then restored; other entries are overwritten.
    make_sbatch: function
        Function that writes a job script, eg :func:`esteem.parallel.make_sbatch`.
    allseed: str or None
        Seed for the training and testing tasks, eg ``'all_cycl'``. If None, ``seed`` is used.
    task_list: list of str or None
        Scripts to write, from ``'mltrain'``, ``'mltraj'``, ``'mltest'``, ``'mlfinaltest'``
        (tests against the data from the generation-2 MD), ``'specdyn'``, ``'spectra'`` and
        ``'cumul_spectra'``. If None, all of them are written.
    """

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
        task_list = ['mltrain', 'mltraj', 'mltest', 'mlfinaltest', 'specdyn', 'spectra', 'cumul_spectra']
    # Write job script for submission to HPC cluster
    for task in task_list:
        # Set up default target and task name
        script_settings['target'] = '$T"_"$W"ac"$X$M$Y'
        script_task = task
        # Set up targets and task names for tasks with different patterns
        if task == "mltraj":
            script_settings['target'] += '"x"$C'
        if task == "specdyn":
            script_settings['target'] += '"_specdyn"'
            script_task = 'mltraj'
        if task == "spectra":
            script_settings['target'] = '$T"_"$W"ac"$X$M"_specdyn_recalc_carved"'
        if task == "cumul_spectra":
            script_settings['target'] = '$T"_"$W"ac"$X$M"_specpycode"'
            script_task = "spectra"
        if task == "mlfinaltest":
            script_settings['target'] += '"_mltraj_"$W"ac2"$M'
            script_task = 'mltest'
        # Set up where to find executable
        script_settings['scriptname'] = '$scr'
        script_settings['execpath'] = '../'
        # Set up seeds and jobnames in script
        if task == 'mltrain' or task == 'mltest' or task == 'mlfinaltest':
            script_settings['seed'] = '$SA'
            script_settings['jobname'] = f'{allseed}_{targstr}_{calc_suffix}_{task}'
        else:
            script_settings['seed'] = '$SP'
            script_settings['jobname'] = f'{seed}_{targstr}_{calc_suffix}_{task}'
        # Set up name of log file
        if task != 'specdyn':  # avoid writing '_specdyn_mltraj' as part of log name
            script_settings['postfix'] = f'| tee -a {script_settings["seed"]}"_"{script_settings["target"]}"_{script_task}"$LOGSUFFIX.log'
        else:
            script_settings['postfix'] = f'| tee -a {script_settings["seed"]}"_"{script_settings["target"]}$LOGSUFFIX.log'
        # Write the script
        script_settings['num_threads'] = 1
        make_sbatch(task=script_task, **script_settings)

    # restore previous declarations section
    script_settings['declarations'] = store_decs


def unit_test():
    """
    Unit test for Active Learning (out of date: its calls do not match the current arguments).
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
    train_calcs = ['it0r', 'it1r', 'it0s', 'it1s']
    targets = {0: 'gs', 1: 'es1'}
    rand_seed = {'a': 1234}
    meth = "MACE"
    truth = "orca"
    new_clusters_tasks = create_clusters_tasks(clusters_task, train_calcs, targets, rand_seed, meth, truth)
    pp.pprint('new_clusters_tasks:')
    pp.pprint(new_clusters_tasks)

    training_task = SimpleNamespace()
    training_task.wrapper = wrapper
    training_task.wrapper.train_args = {'max_num_epochs': 1000, 'start_swa': 50}
    traj_suffixes = ["rattled", "qmd800"]
    dir_suffixes = {"rattled": "rattled", "qmd800": "qmd"}
    ntraj = {}
    ntraj[targets[0], "rattled"] = 1
    ntraj[targets[1], "rattled"] = 0
    ntraj[targets[0], "qmd800"] = 1
    ntraj[targets[1], "qmd800"] = 0
    new_training_tasks = create_mltrain_tasks(training_task, train_calcs, targets, rand_seed, meth, truth, traj_suffixes, dir_suffixes, ntraj)
    print('new_training_tasks:')
    pp.pprint(new_training_tasks)

    mltraj_task = SimpleNamespace()
    md_wrapper = wrapper
    snap_wrapper = wrapper
    mltraj_task.wrapper = wrapper
    new_mltraj_tasks = create_mltraj_tasks(mltraj_task, train_calcs, targets, rand_seed, meth, md_wrapper, snap_wrapper)
    pp.pprint('new_mltraj_tasks:')
    pp.pprint(new_mltraj_tasks)

    mltest_task = SimpleNamespace()
    mltest_task.wrapper = wrapper
    new_mltest_tasks = create_mltest_tasks(mltest_task, train_calcs, targets, rand_seed, truth, meth)
    pp.pprint('new_mltest_tasks:')
    pp.pprint(new_mltest_tasks)

    spectra_task = SimpleNamespace()
    spectra_task.wrapper = wrapper
    spectra_task.wrapper.num_trajs = 0
    new_spectra_tasks = create_spectra_tasks(mltest_task, train_calcs, targets, rand_seed)
    pp.pprint('new_spectra_tasks:')
    pp.pprint(new_spectra_tasks)


do_unit_test = False
if do_unit_test:
    unit_test()
