#!/usr/bin/env python
## \file
## \ingroup tutorial_tmva_keras
## \notebook -nodraw
## This tutorial shows how to apply a trained model to new data.
##
## \macro_code
##
## \date 2017
## \author TMVA Team

from ROOT import TMVA, TFile, TString, TH1F, TH2F
from array import array
from subprocess import call
from os.path import isfile

import math

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

# Load data
#if not isfile('tmva_class_example.root'):
#    call(['curl', '-O', 'http://root.cern.ch/files/tmva_class_example.root'])

#data = TFile.Open('tmva_class_example.root')
#signal = data.Get('TreeS')
#background = data.Get('TreeB')

#data = TFile.Open('../sigmaC/treeData/4071_LHC2018_bdefghijklmnop/AnalysisResults.root')
data = TFile.Open('dataNew.root')
signal = data.Get('treeList_0_24_0_24_Sgn')


#branches = {}
#for branch in signal.GetListOfBranches():
#    branchName = branch.GetName()
#    branches[branchName] = array('f', [-999])
#    reader.AddVariable(branchName, branches[branchName])
#    signal.SetBranchAddress(branchName, branches[branchName])
#    background.SetBranchAddress(branchName, branches[branchName])

variables = {}
variableNames = ['massK0S', 'tImpParBach', 'tImpParV0', 'CtK0S', 'cosPAK0S', 'nSigmapr', 'dcaV0', 'asymmPt']
#variableNames = ['massK0S', 'tImpParBach', 'tImpParV0', 'CtK0S', 'cosPAK0S', 'nSigmapr', 'dcaV0']
for i in range(8):
    print(i, variableNames[i])
    varName = variableNames[i];
    variables[varName] = array('f', [-999])
    reader.AddVariable(varName, variables[varName])
    signal.SetBranchAddress(varName, variables[varName])

spectators = {}
spectatorNames =  ['LcPt', 'massLc2K0Sp']
for i in range(2):
    specName = spectatorNames[i];
    spectators[specName] = array('f', [-999])
    signal.SetBranchAddress(specName, spectators[specName])


# Book methods
reader.BookMVA('PyKeras', TString('dataset/weights/TMVAClassification_PyKeras.weights.xml'))

# Print some example classifications

output = TFile.Open('applicationfile_20230923.root', 'RECREATE')
hist = TH1F('hist', 'hist', 1000, 0, 1.0 )
histVsInvMass = TH2F( 'MVA_vs_InvMass', 'MVA_vs_InvMass; pyKeras; m_{inv}(pK^{0}_{S})[GeV/#it{c}^{2}]', 1000, 0, 1, 1000, 2.05, 2.55);
    
print('Processing tree:')
nevents = signal.GetEntries()
print('Number of events: ', nevents)
for i in range(nevents):
#for i in range(10000000):
    if i%10000 == 0:
        print('--- ... Processing event: ', i)
    signal.GetEntry(i)
#    LcPt = spectators['LcPt'][0]
    massLc2K0Sp = spectators['massLc2K0Sp'][0]
#    if LcPt < 1.0 and LcPt > 0.0:
    hist.Fill(reader.EvaluateMVA('PyKeras'))
    histVsInvMass.Fill(reader.EvaluateMVA('PyKeras'), massLc2K0Sp)
#        print(reader.EvaluateMVA('PyKeras'))
print('')

hist.Write()
histVsInvMass.Write()
output.Close()

