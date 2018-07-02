#!/bin/sh

runNetwork() {
	echo	"runNetwork	: $1"
	echo	"log file	: $2"
	echo	"batchsize	: $3"
	echo	"./testing/$1 $3 > $2"
	`./testing/$1 $3 > ../result/$2`
	ret_val=$?
	return $ret_val
}


#define LIVENESS
#define RECOMPUTE_ON
#define LARGER
#define LRU_ON
#define BLASX_MALLOC

testFunc() {
	echo
	echo	"-----------------------------------"
	echo	"network 		: $1"
	echo	"LIVENESS		: $2"
	echo	"RECOMPUTE_ON	: $3"
	echo	"LARGER			: $4"
	echo	"LRU_ON			: $5"
	echo	"BLASX_MALLOC	: $6"
	echo	"batchsize		: $7"
	
	cmd="cmake "
	if [ "$2" = true ]; then
		cmd="${cmd} -DLIVENESS=1"
	fi
	if [ "$3" = true ]; then
		cmd="${cmd} -DRECOMPUTE_ON=1"
	fi
	if [ "$4" = true ]; then
		cmd="${cmd} -DLARGER=1"
	fi
	if [ "$5" = true ]; then
		cmd="${cmd} -DLRU_ON=1"
	fi
	if [ "$6" = true ]; then
		cmd="${cmd} -DBLASX_MALLOC=1"
	fi

	rm -rf /home/ay27/superneurons_ay27/build/*
	cmd="${cmd} .."
	echo $cmd
	`$cmd`
	make release -j

	runNetwork $1 "$1_$2_$3_$4_$5_$6_$7.log" $7
	ret_val=$?

	echo
	echo	"-----------------------------------" 
	return $ret_val
}

AlexNet() {
	batch=128
	ret_val=0
	while : ; do
		testFunc alexnet $1 $2 $3 $4 $5 $batch
		if [ $? -ne 0 ]; then
			break
		fi
		batch=`expr $batch + 128`
	done
}

#define LIVENESS
#define RECOMPUTE_ON
#define LARGER
#define LRU_ON
#define BLASX_MALLOC
AlexNet false false false false true
AlexNet true false false false true