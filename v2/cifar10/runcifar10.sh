#!/bin/bash
declare -a MODE=("normal" "poly")
declare -a ACT=("sigmoid" "relu")
declare -a DEG=(10 20 40)

for act in ${ACT[@]}
do
	for mode in ${MODE[@]}
	do
		if [ "$mode" == "normal" ]; then
			python cifar10_cnn.py --mode $mode --activ $act
		else
			for deg in ${DEG[@]}
			do
				python cifar10_cnn.py --mode $mode --activ $act --deg $deg
			done
		fi
	done
done
