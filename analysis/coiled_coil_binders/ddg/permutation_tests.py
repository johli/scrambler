#AML - runs permutation tests and saves premutaiton results as npy arrays in outputV2 folder 

from multiprocessing import Pool
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def createScoresDictionary(fileName, fileDir):
    #names are too long, varied, and sizes shift too much.
    #created shortened names in a text file to map both binder 1 & binder 2 scores to these shorthand names
    #fixes up sizes -> either (480,81) or (200, 81) sized score arrays now
    keys = open(fileName, "r")
    keyPairs = [x.strip().split(",") for x in keys.readlines()]
    dictScores = {}
    for pair in keyPairs:
        k = pair[1]
        if k in dictScores:
            dictScores[k].append(pair[0])
        else:
            dictScores[k] = [pair[0]]
    #sort key pairs in dict
    for thing in dictScores:
        dictScores[thing].sort()

    for thing in dictScores:
        #load both, check sizes, look at first row of each
        scores1 = np.load(fileDir + dictScores[thing][0])
        scores2 = np.load(fileDir + dictScores[thing][1])
        #print (scores1.shape)
        #print (scores2.shape)
        #need to sum, then reshape off the 1's
        if scores1.shape[-1] == 20:
            # if shape is (x,1,81,20) -> sum along 20 columns for scores
            # (x,1,81,20) -> (x,1,81,1) -> (x,81)
            sumCol1 = np.reshape(scores1.sum(axis = 3), (scores1.shape[0], 81)) #sum all 20 values
            sumCol2 = np.reshape(scores2.sum(axis=3), (scores2.shape[0], 81))  # sum all 20 values
            #add to dict list
            #print (sumCol1.shape)
            dictScores[thing].append(sumCol1)
            dictScores[thing].append(sumCol2)
        elif scores1.shape[-1] == 1 and len(scores1.shape) != 2:
            # if scores are (x,1,81,1) -> (x,1) (should be able to reshape)
            collapsed = np.reshape(scores1, (scores1.shape[0], 81))
            collapsed2 = np.reshape(scores2, (scores2.shape[0], 81))
            #print(collapsed.shape)
            dictScores[thing].append(collapsed)
            dictScores[thing].append(collapsed2)
        else:
            #do nothing
            #print (scores1.shape)
            dictScores[thing].append(scores1)
            dictScores[thing].append(scores2)
    return dictScores

def makeGraphPermutationTest(allMeans, actualMean, title, saveName, saveNamePlot):
    #makes histogram of means for permutation test
    allArray = np.append(allMeans, [actualMean])
    np.save("./outputV2/" + saveName, allArray) #save the array for the random means, last entry is true mean
    pValue = (sum(allMeans >= actualMean) + 1) / (allMeans.shape[0] + 1)
    plt.hist(allMeans, 20, label='Permutation scores',
             edgecolor='black')
    ylim = plt.ylim()
    print("actual mean: ", actualMean)
    print ("p value: ", pValue)
    plt.plot([actualMean, actualMean], ylim, '--g', linewidth=3,
             label='(p-value %s)' % pValue)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Mean Differences')
    plt.title(title)
    plt.savefig("./outputV2/" + saveNamePlot, dpi=300) #savefig with high dpi
    plt.close()

def mean(toPermute, toCount):
    #shuffles once, does 
    #runs a individual scrambling
    random.shuffle(toPermute)
    return np.mean(toPermute[:toCount]) - np.mean(toPermute[toCount:])

def meanPermutationStarMapDistributor(toPermute, toCount, numberPerms):
    #runs scrambling - using starmap  
    setup = [(toPermute, toCount)] * numberPerms
    with Pool() as p:
        permutations = p.starmap(mean, setup)
    return np.array(permutations)

def recalcPValue(npyAll):
    actualMean = npyAll[-1]
    allMeans = npyAll[:-1]
    pValue = (sum(allMeans >= actualMean) + 1) / (allMeans.shape[0] + 1)
    return pValue

def permutationTests_fixedSize(scoresCsv, scoresB1, scoresB2, scoresName, fixedSize, permutations = 100):
    #run a permutation test on difference of means for each measured metric
    #use random scrambling of all scores and mean of first x method to generate nulls
    #pull fixed number of top nucleotides per dimer (ex: top 8 scored positions in the dimer is 95th percentile)
    #ddGs
    autoscramblerDDGs = []
    otherDDGs = []
    allDDGs = []
    #otherHbonds
    autoscramblerHBonds = []
    otherHBonds = []
    allHBonds = []
    #unsat H bonds
    autoscramblerHBondsUnsat = []
    otherHBondsUnsat = []
    allHBondsUnsat = []
    #fraction H Bond energy
    autoscramblerHBondEn = []
    otherHBondEn = []
    allHBondEns = []
    alaScanInfo = "./NRFalseScores/"
    for i in range(0, scoresB1.shape[0]): #some have 480, some have 200 dimers
        #pull top fixedSize positions by score value 
        row = scoresCsv.iloc[i]
        #getting lengths to slice scores with
        scoresA = scoresB1[i][:row['lenA']] #should be (1, 81)
        scoresB = scoresB2[i][:row['lenB']] #should be (1,81)
        scoresConcat = np.concatenate((scoresA, scoresB))
        
        #use argpartition to get indices of top N positions 
        indsGreater = np.argpartition(scoresConcat, -fixedSize)[-fixedSize:]
        indsLess = np.argpartition(scoresConcat, -fixedSize)[:-fixedSize]
        
        #open ddG values for binder
        ddGValues = np.load(alaScanInfo + row["structure"] + "_ddg.npy")
        allDDGs = allDDGs + ddGValues.flatten().tolist()
        autoscramblerDDGs = autoscramblerDDGs + ddGValues[indsGreater].flatten().tolist()
        otherDDGs = otherDDGs + ddGValues[indsLess].flatten().tolist()
        
        hbonds = np.load(alaScanInfo + row["structure"] + "_hbonds.npy")
        allHBonds = allHBonds + hbonds.flatten().tolist()
        autoscramblerHBonds = autoscramblerHBonds + hbonds[indsGreater].flatten().tolist()
        otherHBonds = otherHBonds + hbonds[indsLess].flatten().tolist()
        
        unsatHBonds = np.load(alaScanInfo + row['structure'] + '_unsat_hbonds.npy')
        allHBondsUnsat = allHBondsUnsat + unsatHBonds.flatten().tolist()
        autoscramblerHBondsUnsat = autoscramblerHBondsUnsat + unsatHBonds[indsGreater].flatten().tolist()
        otherHBondsUnsat = otherHBondsUnsat + unsatHBonds[indsLess].flatten().tolist()
        
        
        hbondsEn = np.load(alaScanInfo + row['structure'] + "_hbonds_energy.npy")
        allHBondEns = allHBondEns + hbondsEn.flatten().tolist()
        autoscramblerHBondEn = autoscramblerHBondEn + hbondsEn[indsGreater].flatten().tolist()
        otherHBondEn = otherHBondEn + hbondsEn[indsLess].flatten().tolist()

    #true means
    print ("number above: ", len(autoscramblerDDGs))
    print ("number below: ", len(otherDDGs))

    setup = [(allDDGs, len(autoscramblerDDGs), permutations), (allHBonds, len(autoscramblerDDGs), permutations), (allHBondsUnsat, len(autoscramblerDDGs), permutations), (allHBondEns, len(autoscramblerDDGs), permutations)]
    results = [meanPermutationStarMapDistributor(x[0],x[1],x[2]) for x in setup]
    
    #randomddGMeans = np.array(meanPermutation(allDDGs, len(autoscramblerDDGs), permutations))
    actualddGMean = np.mean(autoscramblerDDGs) - np.mean(otherDDGs)
    makeGraphPermutationTest(results[0], actualddGMean, "ddG Permutation Test", scoresName + "_top_" + str(fixedSize)  + "_ " + str(permutations) + "_ddG_permutation_test.npy",scoresName + "_top_" + str(fixedSize)  + "_ " + str(permutations) + "_ddG_permutation_test.png")

    #randomHBondMeans = np.array(meanPermutation(allHBonds, len(autoscramblerDDGs), permutations))
    actualHBondMean = np.mean(autoscramblerHBonds) - np.mean(otherHBonds)
    makeGraphPermutationTest(results[1], actualHBondMean, "Delta Interface H Bonds Permutation Test", scoresName + "_top_" + str(fixedSize)   + "_ " + str(permutations) + "_hbonds_permutation_test.npy", scoresName + "_top_" + str(fixedSize)   + "_ " + str(permutations) +"_hbonds_permutation_test.png")

    #randomHBondUnsatMeans = np.array(meanPermutation(allHBondsUnsat, len(autoscramblerDDGs), permutations))
    actualHBondUnsatMean = np.mean(autoscramblerHBondsUnsat) - np.mean(otherHBondsUnsat)
    makeGraphPermutationTest(results[2], actualHBondUnsatMean, "Delta Interface Unsatisfied H Bonds Permutation Test",scoresName + "_top_" + str(fixedSize)   + "_ " + str(permutations) + "_unsat_hbonds_permutation_test.npy", scoresName + "_top_" + str(fixedSize)   + "_ " + str(permutations) + "_unsat_hbonds_permutation_test.png")

    #randomHBondEnMeans = np.array(meanPermutation(allHBondEns, len(autoscramblerDDGs), permutations))
    actualHBondEnMean = np.mean(autoscramblerHBondEn) - np.mean(otherHBondEn)
    makeGraphPermutationTest(results[3], actualHBondEnMean, "Delta Interface H Bond Energy Permutation Test",
                             scoresName + "_top_" + str(fixedSize)   + "_ " + str(permutations) + "_en_hbonds_permutation_test.npy",scoresName + "_top_" + str(fixedSize)   + "_ " + str(permutations) +"_en_hbonds_permutation_test.png")

    
    
#run permutation test for each dimer - do the mean of the top 8 positions per dimer 
#shuffle 10K times and take the mean each time of 8 * 480 positions 
#save each test result in a npy file 
if __name__ == '__main__':
    csvName = "test_set.csv"
    scoresCSV = pd.read_csv(csvName)
    fileDir = "./coiled_coil_binder_scores/"
    #checks out, sorted to binder 1, binder 2 order for the scores in the dict
    dictScores = createScoresDictionary("all_scored_methods.txt", fileDir) 
    
    asDF = {'Model':[], 'DimerN':[],'ddGMean':[], 'ddGPValue':[], 'HBondMeanDifference':[], 'HBondPValue':[], 'HBondUnsatMeanDiff':[], 'HBondUnsatPValue':[], 'HBondEnFractionMeanDiff':[], 'HBondEnFractionPValue':[]}
    toPlot = []
    colorPlot = []
    print (dictScores)
    for thing in dictScores:
        if "tb" in thing:
           #autoscrambler 
            if "inclusion" in thing:
                colorPlot.append("red")
            else:
                colorPlot.append("blue")
        else:
            colorPlot.append("orange")

        print ("ON: ", thing, "OUT OF: ", len(dictScores))
        scores1 = dictScores[thing][2]
        scores2 = dictScores[thing][3]
        permutationTests_fixedSize(scoresCSV, scores1, scores2, thing, 8, 10000)
        
    
    
    
    
    
    