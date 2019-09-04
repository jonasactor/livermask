import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


segresults = []

nummodels  = 1
modeltypes = ["itksnap"]
modeldir   = ["./modelout/itksnap"]

indices = [3,9,19,23,24,25,37,36,44,49,50,75,76,77,88,95,96,101,103,122,128]

for n in range(nummodels):
        for jjj in indices:
            statfilename = '%s/stats-%04d.txt' % (modeldir[n],jjj)
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
                    print(data)

plt.rcParams.update({'font.size':24})

numentries = len(segresults)
print(numentries)
dice0 = [data["Dice"] for data in segresults if data["label"]==0]
dice1 = [data["Dice"] for data in segresults if data["label"]==1]
dice2 = [data["Dice"] for data in segresults if data["label"]==2]
print(min(dice0), max(dice0))
print(min(dice1), max(dice1))
print(min(dice2), max(dice2))
#plt.figure()
#plt.boxplot([dice0, dice1, dice2])
#plt.show()


for n in range(nummodels):
    dice1 = [data["Dice"] for data in segresults if data["label"]==1 and data["modeltype"]==modeltypes[n]]
    dice2 = [data["Dice"] for data in segresults if data["label"]==2 and data["modeltype"]==modeltypes[n]]
    print(modeltypes[n])
    print('\t', min(dice0), max(dice0), sum(dice0)/len(dice0))
    print('\t', min(dice1), max(dice1), sum(dice1)/len(dice1))
    print('\t', min(dice2), max(dice2), sum(dice2)/len(dice2))
    plt.figure(figsize=(6,6), dpi=200)
    plt.boxplot([dice1, dice2])
    plt.xticks([1,2],['Liver','Tumor'])
    plt.ylabel('DSC')
    plt.ylim([-0.05,1.05])
    plt.savefig('dicebox-%s.pdf'%modeltypes[n],bbox_inches="tight")

print(indices)
print(dice1)
