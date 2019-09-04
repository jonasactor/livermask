#!/bin/bash
# script to test different configure options

python3 liverlevelset-analytic.py --builddb --dbfile=trainingdata.csv
python3 liverlevelset-analytic.py --builddb --dbfile=trainingdata_one.csv
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=50 --trainingbatch=4 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt50/batch04/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=5 --trainingbatch=4 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt05/batch04/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=10 --trainingbatch=4 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt10/batch04/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=50 --trainingbatch=1 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt50/batch01/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=5 --trainingbatch=1 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt05/batch01/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=10 --trainingbatch=1 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt10/batch01/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=50 --trainingbatch=20 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt50/batch20/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=5 --trainingbatch=20 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt05/batch20/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=10 --trainingbatch=20 --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt10/batch20/u0ones
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=50 --trainingbatch=4 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt50/batch04/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=5 --trainingbatch=4 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt05/batch04/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=10 --trainingbatch=4 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt10/batch04/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=50 --trainingbatch=1 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt50/batch01/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=5 --trainingbatch=1 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt05/batch01/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=10 --trainingbatch=1 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt10/batch01/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=50 --trainingbatch=20 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt50/batch20/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=5 --trainingbatch=20 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt05/batch20/u0rand
make -f kfold005-000-stats.makefile
python3 liverlevelset-analytic.py --dbfile=trainingdata_one.csv --numepochs=30 --nt=10 --trainingbatch=20 --randinit --kfolds=5 --idfold=0 --outdir=modelout/lsn-analytic/epochs30/nt10/batch20/u0rand
make -f kfold005-000-stats.makefile
