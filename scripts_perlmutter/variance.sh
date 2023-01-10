#!/bin/sh                                                                                                                                                                                                           
#SBATCH -C gpu                                                                                                                                                                                                      
#SBATCH -q regular                                                                                                                                                                                            
#SBATCH -n 128                                                                                                                                                                                                       
#SBATCH --ntasks-per-node 4                                                                                                                                                                                         
#SBATCH --gpus-per-task 1  
#SBATCH --gpu-bind=none                                                                                                                                                                                         
#SBATCH -t 45:00                                                                                                                                                                                                 
#SBATCH -A atlas_g  
#SBATCH --array=1-50                                                                                                                                                                                                
                                                                                                                                                                         
export MPICH_ALLGATHERV_PIPELINE_MSG_SIZE=0
export MPICH_MAX_THREAD_SAFETY=multiple
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0

cd $PSCRATCH/H1PCT/scripts_perlmutter/
                                                                                                   
module load tensorflow/2.6.0
echo python Unfold_var.py --config config_general.json --ntrial ${SLURM_ARRAY_TASK_ID}
srun python Unfold_var.py --config config_general.json --ntrial ${SLURM_ARRAY_TASK_ID}
