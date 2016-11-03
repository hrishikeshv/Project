#!/bin/bash
declare -a MODE=("normal" "poly")
declare -a DEG=(5 10 15)
declare -a LAYERS=(1 2 3)
declare -a ACT=("sigmoid" "relu" "linear")
declare -a FUNC=(1 2 3 4)

for f in ${FUNC[@]}
do
	for hl in ${LAYERS[@]}
	do
		for act in ${ACT[@]}
		do
			for mode in ${MODE[@]}
			do
				if [ "$mode" = "normal" ]; then
					python train_UCI.py $f --mode $mode --hlayers $hl --activ $act
				else
					for deg in ${DEG[@]}
					do
						python train_UCI.py $f --mode $mode --deg $deg --hlayers $hl --activ $act
					done
				fi
			done
		done
	done
done
