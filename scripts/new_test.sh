#!/bin/sh

runNetwork() {
	echo	"runNetwork	: $1"
	echo	"log file	: $2"
	echo	"batchsize	: $3"
	echo	"./testing/$1 $3 > $2"
	`./testing/$1 $3 > ../result2/$2`
	ret_val=$?
	`grep loss ../result2/$2 > ../result2/$2_loss.log`
	`grep "TOTAL TRACKED" ../result2/$2 > ../result2/$2_mem.log`
	return $ret_val
}

testFunc() {
	echo
	echo	"-----------------------------------"
	echo	"network 	: $1"
	echo	"preftech	: $2"
	echo	"recompute	: $3"
	echo	"batchsize	: $4"

	runNetwork $1 "$1_$2_$3_$4.log" $4
	ret_val=$?

	echo
	echo	"-----------------------------------" 
	return $ret_val
}

makeit() {
	cmd="cmake "
	if [ "$1" = true ]; then
		cmd="${cmd} -DPREFETCH=1"
	fi
	if [ "$2" = true ]; then
		cmd="${cmd} -DRECOMPUTE_ON=1"
	fi
	rm -rf /home/ay27/superneurons_ay271/build/*
	cmd="${cmd} .."
	echo $cmd
	`$cmd`
	make release -j
}

pack() {
	# pfetch + recompute
	makeit $1 $2

	batch1=128
	ret_val=0
	while : ; do
		testFunc alexnet $1 $2 $batch1
		if [ $? -ne 0 ]; then
			break
		fi
		batch1=`expr $batch1 + 128`
	done

	batch=16
	ret_val=0
	while : ;  do
		testFunc residual_imagenet $1 $2 $batch
		if [ $? -ne 0 ]; then
			break
		fi
		batch=`expr $batch + 16`
	done

	batch=16
	ret_val=0
	while  : ;  do
		testFunc residual_imagenet_101 $1 $2 $batch
		if [ $? -ne 0 ]; then
			break
		fi
		batch=`expr $batch + 16`
	done

	batch=16
	ret_val=0
	while : ;  do
		testFunc residual_imagenet_152 $1 $2 $batch
		if [ $? -ne 0 ]; then
			break
		fi
		batch=`expr $batch + 16`
	done
}

pack1() {
	makeit $1 $2
	
	batch=16
	ret_val=0
	while : ;  do
		testFunc vgg16 $1 $2 $batch
		if [ $? -ne 0 ]; then
			break
		fi
		batch=`expr $batch + 16`
	done

	batch=16
	ret_val=0
	while  : ;  do
		testFunc inception_v4 $1 $2 $batch
		if [ $? -ne 0 ]; then
			break
		fi
		batch=`expr $batch + 16`
	done
}

pack1 false false	# no
pack1 false true		# recompute
pack1 true true		# prefetch and recompute