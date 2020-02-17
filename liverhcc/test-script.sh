#!/bin/bash

kf=005
idf=003
rho=1.e-7
rootdir=./conv-test
adir=$rootdir/l2-$rho/$kf/$idf/adv
ldir=$rootdir/l2-$rho/$kf/$idf/lip
mdir=$rootdir/l2-$rho/$kf/$idf/liver/modelunet.h5

options="--gpu=2 --trainmodel --dbfile=../trainingdata.csv --rescon --augment --numepochs=2"
python3 tumorhcc.py $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/l2-$rho --l2reg=$rho

mkdir $adir
mkdir $adir/l2
mkdir $adir/dsc
mkdir $ldir

python3 ../analysis/kernel-analysis-2.py --model=$mdir --outdir=$ldir >> $ldir/lipschitz_log.txt 
python3 ../analysis/adversary.py --model=$mdir --outdir=$adir/


echo done.
