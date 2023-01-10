#!/bin/sh                                                                                                                                                                                               
#SBATCH -C gpu                                                                                                                                                                                          
#SBATCH -q regular                                                                                                                                                                                      
#SBATCH -n 128                                                                                                                                                                                           
#SBATCH --ntasks-per-node 4                                                                                                                                                                             
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind=none                                                                                                                                                                                 
#SBATCH -t 59:00                                                                                                                                                                                        
#SBATCH -A atlas_g                                                                                                                                                                                 


export MPICH_ALLGATHERV_PIPELINE_MSG_SIZE=0
export MPICH_MAX_THREAD_SAFETY=multiple
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0

cd $PSCRATCH/H1PCT/scripts_perlmutter/

module load tensorflow/2.6.0
echo python Unfold.py --config config_trk.json
srun python Unfold.py --config config_trk.json
