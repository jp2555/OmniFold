#!/bin/sh                                                                                                                                                                                               
#SBATCH -C gpu                                                                                                                                                                                          
#SBATCH -q regular                                                                                                                                                                                      
#SBATCH -n 64                                                                                                                                                                                           
#SBATCH --ntasks-per-node 4                                                                                                                                                                             
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind=none                                                                                                                                                                                 
#SBATCH -t 30:00                                                                                                                                                                                        
#SBATCH -A atlas_g                                                                                                                                                                                 


export MPICH_ALLGATHERV_PIPELINE_MSG_SIZE=0
export MPICH_MAX_THREAD_SAFETY=multiple
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0

cd $PSCRATCH/OmnifoldPCT/scripts_perlmutter/

module load tensorflow/2.6.0
echo python Unfold.py --config config_general.json
srun python Unfold.py --config config_general.json
