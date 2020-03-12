#!/bin/bash

savedir=/rsrch1/ip/jacctor/livermask/analysis/noisemaker_results
modellocnoreg=/rsrch1/ip/jacctor/livermask/liverhcc/conv/noreg/005/002/liver/modelunet.h5
modellocl1reg=/rsrch1/ip/jacctor/livermask/liverhcc/conv/noaug/onegpu/l1-1.e-5/005/002/liver/modelunet.h5
modellocistareg=/rsrch1/ip/jacctor/livermask/liverhcc/conv/noaug/onegpu/ista-1.e-5/005/002/liver/modelunet.h5
imglist='111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131'
#imglist='111 112'

mkdir -p $savedir/l1
#mkdir -p $savedir/ista

for imgidx in $imglist
do
    echo $imgidx
    python3 noise.py --one_at_a_time --idxOne=$imgidx --model=$modellocnoreg --model_reg=$modellocl1reg --outdir=$savedir/l1/
#    python3 noise.py --one_at_a_time --idxOne=$imgidx --model=$modellocnoreg --model_reg=$modellocistareg --outdir=$savedir/ista/
done

echo done.
