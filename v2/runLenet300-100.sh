#!/bin/bash
declare -a MODE=("normal" "poly")
declare -a ACT=("sigmoid" "relu" "linear")

for act in ${ACT[@]}
do
	for mode1 in ${MODE[@]}
	do
		for mode2 in ${MODE[@]}
		do
			python lenet300-100.py --l1 $mode1 --l2 $mode2 --activ $act
		done
	done
done
