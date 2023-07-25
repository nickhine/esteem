#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Settings and script-writers to run ESTEEM tasks in serial or parallel
"""


# # Make batch scripts

# In[7]:


from copy import deepcopy

gnu_parallel_declarations = (f'NJOBS=$((${{SLURM_NTASKS}}/${{TASKS_PER_PROG}}))\n' +
    f'PARALLEL_OPTS="-N 1 --delay .5 -j ${{NJOBS}} --joblog parallel-${{SLURM_JOBID}}.log"\n' )
gnu_parallel_command = "seq $SLURM_ARRAY_TASK_ID $(($SLURM_ARRAY_TASK_ID+99)) | parallel $PARALLEL_OPTS 'export SLURM_ARRAY_TASK_ID={}; python"

bash_loop_declarations = (f"""
SMAX=1
istart=$SLURM_ARRAY_TASK_ID
iend=$(($SLURM_ARRAY_TASK_ID+SMAX))
for ((i=istart;i<iend;i++)); do
export SLURM_ARRAY_TASK_ID=$i
""")

bash_loop_endmatter = ("""
done
""")

nanosim_1node = {'partition': 'nanosim',
                 'nodes': 1,
                 'ntask': 24,
                 'ncpu': 1,
                 'time': '48:00:00',
                 'mem': '61679mb',
                 'declarations': ''}

nanosim_1core = deepcopy(nanosim_1node)
nanosim_1core['ntask'] = 1

nanosimd_1node = {'partition': 'nanosimd',
                  'nodes': 1,
                  'ntask': 1,
                  'ncpu':  8,
                  'time': '48:00:00',
                  'mem': '14000mb',
                  'declarations': ''}

nanosim_1core = deepcopy(nanosim_1node)
nanosim_1core['ntask'] = 1

archer2_declarations = '''

module -s restore /etc/cray-pe.d/PrgEnv-gnu
module load cray-python
ulimit -s unlimited

export WORK=`echo $HOME | sed 's/home/work/'`
export PYTHONPATH=$PYTHONPATH:$WORK/esteem
export PYTHONUSERBASE=$WORK/.local
export PATH=$PATH:$WORK/.local/bin
export OMP_PLACES=cores
'''

archer2_1node = {'account': 'e89-warp',
                 'partition': 'standard',
                 'qos': 'standard',
                 'nodes': 1,
                 'ntask': 16,
                 'ncpu': 8,
                 'time': '12:00:00',
                 'declarations': archer2_declarations}

archer2_4node = deepcopy(archer2_1node)
archer2_4node['nodes'] = 4
archer2_8node = deepcopy(archer2_1node)
archer2_8node['nodes'] = 8
archer2_16node = deepcopy(archer2_1node)
archer2_16node['nodes'] = 16

avon_tf = '\nmodule purge; module load GCC/10.3.0 CUDA/11.3.1 OpenMPI/4.1.1 TensorFlow/2.6.0\n\n'
avon_1gpu =     {'partition': 'gpu',
                 'gres': 'gpu:quadro_rtx_6000:1',
                 'nodes': 1,
                 'ntask': 1,
                 'ncpu': 12,
                 'time': '24:00:00',
                 'declarations': avon_tf,
                 'endmatter': ''}

avon_intel = '\nmodule purge; module load iccifort/2019.5.281 impi/2019.7.217 NWChem Python\n\n'
avon_1node_intel = {'nodes': 1,
                    'ntask': 48,
                    'ncpu': 1,
                    'time': '24:00:00',
                    'declarations': avon_intel}

avon_gnu = '\nmodule purge; module load GCC/10.3.0 CUDA/11.3.1 OpenMPI/4.1.1 TensorFlow/2.6.0 ORCA AmberTools\n\n'
avon_1node_gnu = {'nodes': 1,
                  'ntask': 48,
                  'ncpu': 1,
                  'time': '24:00:00',
                  'declarations': avon_gnu,
                  'endmatter': ''}

avon_1node_100 = deepcopy(avon_1node_gnu)
parallel_tasks_per_prog = 32
avon_1node_100['declarations'] += f'TASKS_PER_PROG={parallel_tasks_per_prog}\n' + gnu_parallel_declarations
avon_1node_100['command'] =  gnu_parallel_command

sulis_tf = '\nmodule purge; module load GCC/10.3.0 CUDA/11.3.1 OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1\n\n'
sulis_pyt = '\nmodule purge; module load GCC/11.3.0 CUDA/11.7.0 OpenMPI/4.1.4 PyTorch/1.12.1-CUDA-11.7.0 matplotlib\n\n'

sulis_1node =   {'account': 'su007-ndmh',
                 'partition': 'compute',
                 'nodes': 1,
                 'ntask': 128,
                 'ncpu': 1,
                 'time': '24:00:00',
                 'declarations': sulis_pyt,
                 'endmatter': ''}

sulis_1gpu =    {'account': 'su007-ndmh-gpu',
                 'partition': 'gpu',
                 'gres': 'gpu:ampere_a100:1',
                 'nodes': 1,
                 'ntask': 1,
                 'ncpu': 32,
                 'time': '24:00:00',
                 'declarations': sulis_pyt,
                 'endmatter': ''}
sulis_3gpu = deepcopy(sulis_1gpu)
sulis_3gpu['gres'] = 'gpu:ampere_a100:3'

sulis_1node_100 = deepcopy(sulis_1node)
parallel_tasks_per_prog = 32
sulis_1node_100['declarations'] += f'TASKS_PER_PROG={parallel_tasks_per_prog}\n' + gnu_parallel_declarations
sulis_1node_100['command'] =  gnu_parallel_command

def get_default_script_settings(wrapper):

    from esteem.wrappers.onetep import OnetepWrapper
    from esteem.wrappers.nwchem import NWChemWrapper
    from esteem.wrappers.orca import ORCAWrapper
    from esteem.wrappers.physnet import PhysNetWrapper
    from esteem.wrappers.mace import MACEWrapper
    from esteem.wrappers.amber import AmberWrapper

    # Find out fully-qualified domain name
    import socket
    host = socket.getfqdn()

    # Set up ONETEP and queueing system
    if "warwick.ac.uk" in host:
        # local SCRTP
        if "theory" in host or "scrtp" in host:
            script_settings = deepcopy(nanosimd_1node)
        elif "stan" in host:
            script_settings = deepcopy(nanosim_1node)
        if isinstance(wrapper,ORCAWrapper):
            wrapper.setup()
        if isinstance(wrapper,OnetepWrapper):
            wrapper.setup(onetep_cmd='~/onetep/bin/onetep.csc',
                                   set_pseudo_path='/storage/nanosim/NCP17_PBE_OTF/',
                                   set_pseudo_suffix="_NCP17_PBE_OTF.usp")

    elif "avon" in host:
        # avon
        script_settings = deepcopy(avon_1node_gnu)
        script_settings['ntask'] = 48
        script_settings['ncpu'] = 1
        if isinstance(wrapper,OnetepWrapper):
            wrapper.setup(onetep_cmd='~/onetep/bin/onetep.avon',
                                   set_pseudo_path='~/NCP17_PBE_OTF/',
                                   set_pseudo_suffix="_NCP17_PBE_OTF.usp")
        if isinstance(wrapper,PhysNetWrapper):
            script_settings = deepcopy(avon_1gpu)
        if isinstance(wrapper,ORCAWrapper):
            script_settings = deepcopy(avon_1node_gnu)
            script_settings['declarations'] = script_settings['declarations'] + bash_loop_declarations
            script_settings['endmatter'] = script_settings['endmatter'] + bash_loop_endmatter
        if isinstance(wrapper,NWChemWrapper):
            script_settings = deepcopy(avon_1node_intel)
            script_settings['declarations'] = avon_intel
        if isinstance(wrapper,AmberWrapper):
            script_settings = deepcopy(avon_1node_gnu)

    elif "sulis" in host:
        # sulis
        script_settings = deepcopy(sulis_1node)
        script_settings['ntask'] = 128
        script_settings['ncpu'] = 1
        if isinstance(wrapper,OnetepWrapper):
            wrapper.setup(onetep_cmd='~/onetep/bin/onetep.sulis',
                                   set_pseudo_path='~/NCP17_PBE_OTF/',
                                   set_pseudo_suffix="_NCP17_PBE_OTF.usp")
        if isinstance(wrapper,PhysNetWrapper):
            script_settings = deepcopy(sulis_1gpu)
        if isinstance(wrapper,MACEWrapper):
            script_settings = deepcopy(sulis_1gpu)
        if isinstance(wrapper,ORCAWrapper):
            script_settings = deepcopy(sulis_1node)
            script_settings['declarations'] = script_settings['declarations'] + bash_loop_declarations
            script_settings['endmatter'] = script_settings['endmatter'] + bash_loop_endmatter
        if isinstance(wrapper,NWChemWrapper):
            script_settings = deepcopy(sulis_1node)
            script_settings['declarations'] = sulis_intel
        if isinstance(wrapper,AmberWrapper):
            script_settings = deepcopy(sulis_1node)

    elif "uan" in host or "nid" in host:
        # ARCHER2
        if isinstance(wrapper,OnetepWrapper):
            wrapper.setup(onetep_cmd='/work/e89/e89/ndmh3/onetep/bin/onetep.archer2',
                                   mpirun='srun --hint=nomultithread --distribution=block:block',
                                   set_pseudo_path='/work/e89/e89/ndmh3/NCP17_PBE_OTF/',
                                   set_pseudo_suffix="_NCP17_PBE_OTF.usp")
            wrapper.excitations_params['fftbox_batch_size'] = 16
        script_settings = deepcopy(archer2_1node)
        script_settings['ntask'] = 16
        script_settings['ncpu'] = 8
    
    else:
        raise Exception(f"Hostname {host} not recognised for automatic parallelisation setup.\n"+
                          "Please edit definitions in drivers.py or setup parallelisation manually.")
        
    return script_settings

def make_sbatch(seed,task='',**kwargs):
    """
    Writes a SLURM sbatch script for a task
    
    task: str
        Name of the task to be performed
    kwargs: dict of strings
        Many options to control script parameters, including possible keys:
        
        ``account``, ``partition``, ``qos``, ``nodes``, ``ntask``, ``ncpu``,
        ``time``, ``mem``, ``execpath``, ``target``, ``jobname``
    """
    
    # Other defaults
    slurm_command="python"
    target = None
    execpath = ""
    slurm_preamble='#!/bin/bash\n'
    slurm_declarations = "\n"
    slurm_endmatter = ""

    # Retrieve values from kwargs if present
    if 'partition' in kwargs:
        slurm_partition = kwargs['partition']
        slurm_preamble += f'#SBATCH --partition={slurm_partition}\n'
    if 'export' in kwargs:
        slurm_export = kwargs['export']
        slurm_preamble += f'#SBATCH --export={slurm_export}\n'
    if 'qos' in kwargs:
        slurm_qos = kwargs['qos']
        slurm_preamble += f'#SBATCH --qos={slurm_qos}\n'
    if 'gres' in kwargs:
        slurm_gres = kwargs['gres']
        slurm_preamble += f'#SBATCH --gres={slurm_gres}\n'
    if 'nodes' in kwargs:
        slurm_nodes = kwargs['nodes']
        slurm_preamble += f'#SBATCH --nodes={slurm_nodes}\n'
    if 'ntask' in kwargs:
        slurm_ntask = kwargs['ntask']
        slurm_preamble += f'#SBATCH --ntasks-per-node={slurm_ntask}\n'
    if 'ncpu' in kwargs:
        slurm_ncpu = kwargs['ncpu']
        slurm_preamble += f'#SBATCH --cpus-per-task={slurm_ncpu}\n'
    else:
        slurm_ncpu = 1
    if 'time' in kwargs:
        slurm_time = kwargs['time']
        slurm_preamble += f'#SBATCH --time={slurm_time}\n'
    if 'account' in kwargs:
        slurm_acct = kwargs['account']
        slurm_preamble += f'#SBATCH --account={slurm_acct}\n'
    if 'mem' in kwargs:
        slurm_mem = kwargs['mem']
        slurm_preamble += f'#SBATCH --mem={slurm_mem}\n'

    if 'command' in kwargs:
        slurm_command = kwargs['command']
    if 'declarations' in kwargs:
        slurm_declarations = kwargs['declarations']
    if 'endmatter' in kwargs:
        slurm_endmatter = kwargs['endmatter']
    if 'execpath' in kwargs:
        execpath = kwargs['execpath']
    if 'exec' in kwargs:
        slurm_exec = kwargs['exec']
    else:
        slurm_exec = ""
    if 'logdir' in kwargs:
        slurm_logdir = kwargs['logdir']+"/"
    else:
        slurm_logdir = ""
    if 'target' in kwargs:
        target = kwargs['target']

    slurm_jobargs=f'{task} {seed}'
    if target is not None:
        # if target is a string representing an integer, convert to an integer
        if isinstance(target,str):
            if target.isdigit():
                target = int(target)
        # if it is an integer, write ground/excited state string for job names
        if isinstance(target,int):
            if int(target) == 0:
                targstr= "_gs"
            else:
                targstr = f"_es{target}"
        # otherwise just append to jobnames
        else:
            targstr = f"_{target}"
        slurm_jobargs = f'{slurm_jobargs} {target}'
    else:
        targstr=""
    if 'scriptname' in kwargs:
        slurm_scriptname = kwargs['scriptname']
    else:
        slurm_scriptname = seed
    if 'jobname' in kwargs:
        slurm_jobname = kwargs['jobname']
    else:
        slurm_jobname=f'{seed}{targstr}_{task}'
    slurm_preamble += f'#SBATCH --job-name={slurm_jobname}\n'
    if slurm_command.endswith("python"):
        slurm_exec=f'{execpath}{slurm_scriptname}.py'

    if 'num_threads' in kwargs:
        slurm_num_threads = kwargs['num_threads']
    else:
        slurm_num_threads = slurm_ncpu
    slurm_declarations = (f'\nexport OMP_NUM_THREADS={slurm_num_threads}\n'
                          + slurm_declarations +
                          f'[ ! -z "$SLURM_ARRAY_TASK_ID" ] && export LOGSUFFIX="_"$SLURM_ARRAY_TASK_ID\n')
    if 'postfix' in kwargs:
        slurm_postfix = kwargs['postfix']
    else:
        slurm_postfix = f"| tee -a {slurm_logdir}{slurm_jobname}$LOGSUFFIX.log"

    if "'" in slurm_command:
        slurm_postfix += "'\n"
    else:
        slurm_postfix += "\n"
        

    with open(f'{slurm_jobname}_sub', 'w') as f:
        f.write(slurm_preamble)
        f.write(slurm_declarations)
        f.write(f'''{slurm_command} {slurm_exec} {slurm_jobargs} {slurm_postfix}''')
        f.write(f'''{slurm_endmatter}''')


# In[ ]:


def make_pbs(seed,dest='local',task='solutes',execpath='',ncpu=None,target=None):
    """
    Writes a PBS script for a task. Possibly out-of-date (not recently used)
    
    task: str
        Name of the task to be performed
    kwargs: dict of strings
        Many options to control script parameters, including possible keys:
        
        ``account``, ``partition``, ``nodes``, ``ntask``, ``ncpu``, ``time``, ``mem``,
        ``execpath``, ``target``, ``jobname``
    """
    
    if target is not None:
        if target == 0:
            targstr= "gs"
        else:
            targstr = f"es{target}"
    if dest=='archer':
        pbs_acct="e89-warp"
        pbs_queue="compute"
        pbs_nodes=10
        pbs_ntask=4
        pbs_ncpu=6
        pbs_mem=2679
        pbs_time="24:00:00"
    else:
        raise Exception(f'Destination {dest} not recognised')
    
    pbs_jobname=f'{seed}_{task}'
    pbs_jobargs=f'{task}'
    if dest=='archer':
        pbs_command="python3"
    else:
        pbs_command="python"
    pbs_exec=f'{execpath}{seed}.py'
    pbs_options=f'''#!/bin/bash
#PBS -N {pbs_jobname}
#PBS -l select={pbs_nodes}
#PBS -A {pbs_acct}
#PBS -l walltime={pbs_time}
#PBS -j oe
#PBS -r y
#NOTE: adjust this to control the range of jobs being executed
#PBS -J 0-1

'''

    pbs_setup=f'''
# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

# Set parameters for OpenMP
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export OMP_NESTED=true
export MKL_NESTED=true

# Set number of threads per MPI process (must be a divisor of 24)
export OMP_NUM_THREADS={pbs_ncpu}
export OMP_STACKSIZE=64M

# Set number of MPI processes per node 
# (NUM_PPM is the number of processor cores per node)
export NP=$((NUM_PPN / $OMP_NUM_THREADS))

# Set total number of MPI processes 
# (NODE_COUNT is the number of nodes allocated to your job)
export NMPI=$((NP * NODE_COUNT))

# Gather aprun arguments
export APRUN_ARGS="-n $NMPI -N $NP -d $OMP_NUM_THREADS -S $((NP / 2)) -cc numa_node"

# Load python module
module load python-compute/3.6.0_gcc6.1.0

# Execute
'''

    with open(f'{pbs_jobname}_sub', 'w') as f:
        f.write(pbs_options)
        f.write(pbs_setup)
        f.write(f'''{pbs_command} {pbs_exec} {pbs_jobargs} | tee -a {pbs_jobname}.log \n''')


# # Helper functions

# In[ ]:


from os import environ


def get_array_task_id():
    
    task_id = None

    # Try to find from SLURM
    try:
        tmp = int(environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        tmp = None
    if tmp is not None:
        task_id = tmp

    # Try to find from PBS
    try:
        tmp = int(environ["PBS_ARRAY_INDEX"])
    except KeyError:
        tmp = None
    if tmp is not None:
        task_id = tmp

    return task_id
    

