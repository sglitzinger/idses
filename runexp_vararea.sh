#!/bin/bash

tasknos='10 20 40 80'
areas='91.2 72.2 53.2 34.2 15.2'
decstep=3.8
for number in $tasknos
do
    for value in {1..1}
    do
        filename="low_n${number}_${value}_biglittle"
        #for area in $areas
        area=91.2
        while [ true ]
        do
            if (( $(echo "$area < 15.2" |bc -l) )); then
                break
            fi
            echo "Initiating computation for graph $filename, max. chip size $area mm^2..."      
            python dssched_biglittle_vararea.py ./tasksetsbiglittleadjusted/ ./results/ $filename.csv $area
            area=$( echo "$area - $decstep" | bc )
        done
    done
done