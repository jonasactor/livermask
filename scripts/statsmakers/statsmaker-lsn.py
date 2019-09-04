import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

k = 5
klist = range(k)
klist_done = [0,1,2, 3, 4]
kindices = []
kindices.append(range(0  ,27 ))
kindices.append(range(27 ,53 ))
kindices.append(range(53 ,79 ))
kindices.append(range(79 ,105))
kindices.append(range(105,131))

segresults = []

nummodels  = 1
modeltypes = ["lsn"]
modeldir   = ["./modelout/lsn/TrainingData"]

for n in range(nummodels):
    for kkk in klist:
        if kkk in klist_done:
            for jjj in kindices[kkk]:
                    statfilename = '%s/%03d/%03d/stats-%04d.txt' % (modeldir[n],k,kkk,jjj)
                    fi = open(statfilename, 'r')
                    text = fi.read()
                    fi.close()
                    for ln in text.split("\n"):
                        if ln.startswith(r"Reading #1 from"):
                            file1name = ln.split()[-1]
                        if ln.startswith(r"Reading #2 from"):
                            file2name = ln.split()[-1]
                    for ln in text.split("\n"):
                        if ln.startswith("OVL:"):
                            data = {}
                            lnlist = ln.split()
                            label       =   int(lnlist[1][:-1])
                            vox1        =   int(lnlist[2][:-1])
                            vox2        =   int(lnlist[3][:-1])
                            overlap     =   int(lnlist[4][:-1])
                            Dice        = float(lnlist[5][:-1])
                            int2ratio   = float(lnlist[6])
                            data["indx"]         = jjj
                            data["kfold"]        = kkk
                            data["modeltype"]    = modeltypes[n]
                            data["file1name"]    = file1name
                            data["file2name"]    = file2name
                            data["label"]        = label
                            data["vox1"]         = vox1
                            data["vox2"]         = vox2
                            data["overlap"]      = overlap
                            data["Dice"]         = Dice
                            data["int2ratio"]    = int2ratio
                            segresults.append(data)

plt.rcParams.update({'font.size':24})

numentries = len(segresults)
print(numentries)
dice0 = [data["Dice"] for data in segresults if data["label"]==0]
dice1 = [data["Dice"] for data in segresults if data["label"]==1]
print(min(dice0), max(dice0), sum(dice0)/len(dice0))
print(min(dice1), max(dice1), sum(dice1)/len(dice1))
#plt.figure()
#plt.boxplot([dice0, dice1])
#plt.show()


for kkk in klist_done:
    dice0 = [data["Dice"] for data in segresults if data["label"]==0 and data["kfold"]==kkk and data["modeltype"]==modeltypes[0]]
    dice1 = [data["Dice"] for data in segresults if data["label"]==1 and data["kfold"]==kkk and data["modeltype"]==modeltypes[0]]
    print('kfold ', kkk)
    print('\t', min(dice0), max(dice0), sum(dice0)/len(dice0))
    print('\t', min(dice1), max(dice1), sum(dice1)/len(dice1))

