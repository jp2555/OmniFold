#!/bin/sh                                                                                                                                                                                               
#SBATCH -C gpu                                                                                                                                                                                          
#SBATCH -q regular                                                                                                                                                                                      
#SBATCH -n 48                                                                                                                                                                                           
#SBATCH --ntasks-per-node 4                                                                                                                                                                             
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind=none                                                                                                                                                                                 
#SBATCH -t 59                                                                                                                                                                                        
#SBATCH -A m3246                                                                                                                                                                                 


export MPICH_ALLGATHERV_PIPELINE_MSG_SIZE=0
export MPICH_MAX_THREAD_SAFETY=multiple
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0

cd $PSCRATCH/H1PCT/scripts_perlmutter/

module load tensorflow/2.9.0
echo python Unfold.py --config config_4d_general.json --closure --nevts 50e6 --data_folder /global/cfs/cdirs/m3246/jing
srun python Unfold.py --config config_4d_general.json --closure --nevts 50e6 --data_folder /global/cfs/cdirs/m3246/jing
