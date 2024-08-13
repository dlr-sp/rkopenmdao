#! /usr/bin/bash

outer=(1 2 3 4 5 6)
mid=(60 600 6000)
inner=(strong weak)

for i in ${outer[@]}
do
  for j in ${mid[@]}
  do
    for k in ${inner[@]}
    do
      mpirun -np $i python src/rkopenmdao/utils/parallel_dummy_components.py --runtime 1 --core_count $i --size $j --scaling_type $k
    done
  done
done