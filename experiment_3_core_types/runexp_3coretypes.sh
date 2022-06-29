#!/bin/bash

tasknos='10 20 40 80'
for number in $tasknos
do
    for value in {1..10}
    do
        filename="low_n${number}_${value}_biglittle"
        echo "Initiating computation for graph $filename..."      
        python dssched_biglittle_vararea_3coretypes.py ./tasksetsbiglittleadjusted/ ./results/ $filename.csv 91.2
    done
done
