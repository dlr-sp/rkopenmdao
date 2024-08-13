#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --partition=rome
#SBATCH --time=23:00:00

outer=(1 2 4 8 16 32 64 128)
mid=(128 2048 32768 524288 8388608)
inner=(1 2 4 8)

for i in ${outer[@]}
do
  for j in ${mid[@]}
  do
    for k in ${inner[@]}
    do
      srun --ntasks $i --exclusive python src/rkopenmdao/utils/parallel_dummy_components.py --runtime $k --core_count $i --size $j --scaling_type strong
    done
  done
done