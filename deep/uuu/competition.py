import array
import csv
import rootpy.ROOT as ROOT
import rootpy
from rootpy.tree import Tree, TreeChain, TreeModel, FloatCol, IntCol
import time
from rootpy.io import root_open
from random import gauss

import warnings
warnings.filterwarnings( action='ignore', category=RuntimeWarning, message='creating converter.*' )

import sys
import os

sys.path.insert(0, os.path.expanduser('~/code_peters/rootWork/lib'))
sys.path.insert(0, os.path.expanduser('~/.local/lib/python2.7/site-packages/rootpy'))


base_variables = [
    "LifeTime"
    ,"FlightDistance"
    ,"FlightDistanceError"
    ,"VertexChi2"
    ,"pt"
    ,"IP"
    ,"IPSig"
    ,"dira"
    ,"DOCAone"
    ,"DOCAtwo"
    ,"DOCAthree"
    ,"IP_p0p2"
    ,"IP_p1p2"
    ,"isolationa"
    ,"isolationb"
    ,"isolationc"
    ,"isolationd"
    ,"isolatione"
    ,"isolationf"
    ,"iso"
    ,"CDF1"
    ,"CDF2"
    ,"CDF3"
    ,"ISO_SumBDT"
    ,"p0_IsoBDT"
    ,"p1_IsoBDT"
    ,"p2_IsoBDT"
    ,"p0_track_Chi2Dof"
    ,"p1_track_Chi2Dof" 
    ,"p2_track_Chi2Dof" 
    ,"p0_pt"
    ,"p0_p"
    ,"p0_eta"
    ,"p0_IP"
    ,"p0_IPSig"
    ,"p1_pt"
    ,"p1_p"
    ,"p1_eta"
    ,"p1_IP"
    ,"p1_IPSig"
    ,"p2_pt"
    ,"p2_p"
    ,"p2_eta"
    ,"p2_IPSig"
    ,"p2_IP"
    ,"SPDhits"
    ,"signal"
]



usedVariables = [
    "LifeTime"
    ,"FlightDistance"
    ,"FlightDistanceError"
    ,"VertexChi2"
    ,"pt"
    ,"IP"
    ,"IPSig"
    ,"dira"
    ,"DOCAone"
    ,"DOCAtwo"
    ,"DOCAthree"
    ,"IP_p0p2"
    ,"IP_p1p2"
    ,"isolationa"
    ,"isolationb"
    ,"isolationc"
    ,"isolationd"
    ,"isolatione"
    ,"isolationf"
    ,"iso"
    ,"CDF1"
    ,"CDF2"
    ,"CDF3"
    ,"ISO_SumBDT"
    ,"p0_IsoBDT"
    ,"p1_IsoBDT"
    ,"p2_IsoBDT"
    ,"p0_track_Chi2Dof"
    ,"p1_track_Chi2Dof" 
    ,"p2_track_Chi2Dof" 
    ,"p0_pt"
    ,"p0_p"
    ,"p0_eta"
    ,"p0_IP"
    ,"p0_IPSig"
    ,"p1_pt"
    ,"p1_p"
    ,"p1_eta"
    ,"p1_IP"
    ,"p1_IPSig"
    ,"p2_pt"
    ,"p2_p"
    ,"p2_eta"
    ,"p2_IP"
    ,"p2_IPSig"
    ,"SPDhits"
#    ,("frac","SPDhits/(CDF3+1)")
]



spectatorNames = [
    "mass"
    ,"min_ANNmuon"
]


training_filename = "/home/peters/test/kaggle_flavour/flavours-of-physics-start/tau_data/training.root"
default_path = "/home/peters/test/kaggle_flavour/flavours-of-physics-start/tau_data/"



def load (**kwargs):
    print "------------ load ---------------"
    for key,value in kwargs.iteritems ():
        print key," = ",value
        
    input_filenames = kwargs.setdefault ("filenames", None)
    input_treename = kwargs.setdefault ("treename", "data")

    tree = None
    idx = 0
    for filename in input_filenames:
        if not tree:
            print "create chain for "+input_treename
            tree = ROOT.TChain (input_treename)
            tree.Add (filename)
        else:
            print "add friend "+input_treename
            tmp = ROOT.TChain (input_treename)
            tmp.Add (filename)
            friendname = "p"+str(idx)
            print friendname
            idx += 1
            tree.AddFriend (tmp, friendname)

    return tree



def classify (**kwargs):
    print "------------ classification ---------------"
    for key,value in kwargs.iteritems ():
        print key," = ",value
        
    ROOT.TMVA.Tools.Instance ()

    output_filename = kwargs.setdefault ("filename", None)
    input_variables = kwargs.setdefault ("variables", None)
    input_spectators = kwargs.setdefault ("spectators", [])
    signal_cut = ROOT.TCut (kwargs.setdefault ("signal_cut", "signal==1"))
    background_cut = ROOT.TCut (kwargs.setdefault ("background_cut", "signal==0"))
    base_cut = ROOT.TCut (kwargs.setdefault ("base_cut", "(LifeTime >= 0 && FlightDistance >= 0)"))
    input_tree = kwargs.setdefault ("input_tree", None)
    method_suffix = kwargs.setdefault ("method_suffix", "")
    
    outputFile = rootpy.io.File.Open (output_filename, "RECREATE" )

    jobName = "Flavor"
    factory = ROOT.TMVA.Factory( jobName, outputFile, "AnalysisType=Classification:Transformations=I:!V" )
    for var in input_variables:
        if type(var) == str:
            factory.AddVariable (var, 'F')
        else:
            varcomposed = var[0] + ":=" + var[1]
            factory.AddVariable (varcomposed, 'F')
        
    for spec in input_spectators:
        if type(spec) == str:
            factory.AddSpectator (spec, 'F')
        else:
            speccomposed = spec[0] + ":=" + spec[1]
            factory.AddVariable (speccomposed, 'F')

    factory.AddTree (input_tree, "Signal", 1.0, base_cut + signal_cut, "TrainingTesting");
    factory.AddTree (input_tree, "Background", 1.0, base_cut + background_cut, "TrainingTesting");

    mycuts = ROOT.TCut ()
    mycutb = ROOT.TCut ()
    factory.PrepareTrainingAndTestTree (mycuts, mycutb, "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V")


    layoutString = "Layout=TANH|100,TANH|50,LINEAR"

    trainingConfig = [
        "LearningRate=1e-2,Momentum=0.0,Repetitions=1,ConvergenceSteps=50,BatchSize=20,TestRepetitions=7,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True"
        , "LearningRate=1e-3,Momentum=0.0,Repetitions=1,ConvergenceSteps=20,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1"
#        , "LearningRate=1e-4,Momentum=0.0,Repetitions=1,ConvergenceSteps=2,BatchSize=40,TestRepetitions=7,WeightDecay=0.0001,Regularization=L2,Multithreading=True"
        , "LearningRate=1e-5,Momentum=0.0,Repetitions=1,ConvergenceSteps=10,BatchSize=70,TestRepetitions=7,WeightDecay=0.0001,Regularization=NONE,Multithreading=True"
    ]

    trainingStrategy = "TrainingStrategy="
    for idx, conf in enumerate (trainingConfig):
        if idx != 0:
            trainingStrategy += "|"
        trainingStrategy += conf
            
    nnOptions = "!H:!V:ErrorStrategy=CROSSENTROPY:VarTransform=P+G:WeightInitialization=XAVIERUNIFORM"
    nnOptions += ":"+layoutString + ":" + trainingStrategy
        
    methodName = "NNPG"+method_suffix
    factory.BookMethod (ROOT.TMVA.Types.kNN, methodName, nnOptions)

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    outputFile.Close()

    weightFile = "weights/"+jobName+"_"+methodName+".weights.xml"
    return {"method_name" : methodName, "weightfile_name" : weightFile}
        


def setbranch (varname):
    vname = varname
    vtype = 'f'
    if varname == "id":
        vtype = 'i'
        vname = varname.upper ()
    cmd = "%s = array.array ('%s',[0])\n"%(varname,vtype)
    cmd = cmd + 'if "%s" in variablesForFiles[currentFileName]:\n'%(varname)
    cmd = cmd + '    tree.SetBranchAddress ("%s",%s)\n'%(varname,vname)
    return cmd

def branch (varname, alternativeName = ""):
    vname = varname
    vtype = 'f'
    if varname == "id":
        vtype = "i"
        vname = varname.upper ()
    if alternativeName == "":
        alternativeName = varname
    cmd = 'outTree.Branch ("%s",%s,"%s/%s")\n'%(alternativeName,vname,alternativeName,vtype)
    return cmd



                
    
def predict (**kwargs):
    filenames = kwargs.setdefault ("filenames", ["training","test","check_correlation","check_agreement"])
    variableOrder = kwargs.setdefault ("variable_order", ["id", "signal", "mass", "min_ANNmuon", "prediction"])
    prediction_name = kwargs.setdefault ("prediction_name", "prediction")

    execute_tests = kwargs.setdefault ("execute_tests",False)

    
    # default values
    variablesForFiles = {
        "training" : ["prediction","id","signal","mass","min_ANNmuon"],
        "test" : ["prediction","id"],
        "check_agreement" : ["signal","weight","prediction"],
        "check_correlation" : ["mass","prediction"]
        }
    variablesForFiles = kwargs.setdefault ("variablesForFiles", variablesForFiles)
    createForFiles = {
        "training" : ["csv"],
        "test" : ["csv"],
        "check_correlation" : ["csv"],
        "check_agreement" : ["csv"]
        }
    input_variables = kwargs.setdefault ("variables", None)
    input_spectators = kwargs.setdefault ("spectators", [])
    method_name = kwargs.setdefault ("method_name", None)
    weightfile_name = kwargs.setdefault ("weightfile_name", None)
    
    
    print "------------ prediction ---------------"
    for key,value in kwargs.iteritems ():
        print key," = ",value


    ROOT.TMVA.Tools.Instance ()

    reader = ROOT.TMVA.Reader( "!Color:!Silent" )

    variables = []
    varIndex = {}
    for idx, var_name in enumerate (input_variables):
        tmp = array.array('f',[0])
        variables.append (tmp)
        if type(var_name) == str:
            reader.AddVariable (var_name, tmp)
            varIndex[var_name] = idx
        else:
            varcomposed = var_name[0] + ":=" + var_name[1]
            reader.AddVariable (varcomposed, tmp)


        
    spectators = []
    specIndex = {}
    for idx, spec_name in enumerate (input_spectators):
        tmp = array.array('f',[0])
        spectators.append (tmp)
        if type(spec_name) == str:
            reader.AddVariable (spec_name, tmp)
            specIndex[spec_name] = idx
        else:
            speccomposed = spec_name[0] + ":=" + spec_name[1]
            reader.AddVariable (speccomposed, tmp)


    reader.BookMVA (method_name, weightfile_name)

    returnValues = {}
    for currentFileName in filenames:
        print "predict for  file : ",currentFileName
        doCSV = "csv" in createForFiles[currentFileName]
        doROOT = ("root" in createForFiles[currentFileName]) or doCSV

        if not doCSV and not doROOT:
            continue
        
        fileName = default_path + currentFileName + ".root"
        
        # define variables
        ID = array.array ('i',[0])
        outputVariables = variablesForFiles[currentFileName]

        prediction = array.array ('f',[0])
        weight = array.array ('f',[0])
        min_ANNmuon = array.array ('f',[0])
        mass = array.array ('f',[0])
        signal = array.array ('f',[0])
       
        # --- open input file
        f = rootpy.io.File.Open (fileName)
	tree = f.Get("data");


        for v in outputVariables:
            if v != "prediction":
                cmd = setbranch (v)
                #print cmd
                exec (cmd)

      
	# variables for prediction
        #baseVariables = [array.array ('f',[0]) for i in xrange (len (base_variables))]
	# for idx, currentVariableName in enumerate (base_variables):
	#     tree.SetBranchAddress (currentVariableName, baseVariables[idx]);


        # create tree formulas
        formulas = []
        for idx,var in enumerate (input_variables):
            if type(var) == str:
                fml = None
                tree.SetBranchAddress (var, variables[varIndex[var]])
            else:
                fml = ROOT.TTreeFormula (var[0], var[1], tree)
            formulas.append (fml)

            
        # ---- make ROOT file if requested
        outTree = None
        if doROOT:
            rootFileName = currentFileName + "_p_" + method_name + ".root"
            outRootFile = rootpy.io.File (rootFileName, "RECREATE")
            outTree = Tree ("data","data")

            for var in variableOrder:
                if var in variablesForFiles[currentFileName]:
                    altName = ""
                    if var == "prediction":
                        altName = prediction_name
                    cmd = branch (var, altName)
                    exec (cmd)
            
            curr = currentFileName + "_prediction_root"
            returnValues[curr] = rootFileName

        #
        tmstmp = time.time ()
	for ievt in xrange (tree.GetEntries()):
	    tree.GetEntry (ievt)
	    # predict
            for idx,fml in enumerate (formulas):
                if fml != None:
                    variables[idx][0] = fml.EvalInstance ()

            # this will create a harmless warning
            # https://root.cern.ch/phpBB3/viewtopic.php?f=14&t=14213                
	    prediction[0] = reader.EvaluateMVA (method_name)
            prediction[0] = max (0.0, min (1.0, prediction[0]))
            #print prediction
            outTree.fill ()

            if ievt%10000 == 0:
                tmp = tmstmp
                tmstmp = time.time ()
                print ievt,"   t = %f"%(tmstmp-tmp)

        if doROOT:
            outRootFile.Write ()
        
        # ---- prepare csv file
        #csvfile = None
        writer = None
        if doCSV:
            print "prepare csv"
            csvFileName = currentFileName + "_p_" + method_name + ".csv"

            csvFile = open (csvFileName, 'w')
            outTree.csv (",", stream=csvFile);
            csvFile.close ()

            curr = currentFileName + "_prediction_csv"
            returnValues[curr] = csvFileName


            

        f.Close ()

        #if doCSV:
        #    csvfile.close ()
            
        if doROOT:
            outRootFile.Close ()
    if execute_tests:
        cmd = "os.system ('python tests.py %s %s %s')"%(returnValues["check_agreement_prediction_csv"],returnValues["check_correlation_prediction_csv"],returnValues["training_prediction_csv"])
        exec (cmd)
        
    return returnValues

            
    


    

# tree = load (filenames=[training_filename, training_filename])
# classify (filename="classtest.root", variables=variableNames, input_tree=tree)
            

    

def testPrediction ():
    # tree = load (filenames=[training_filename])
    # retClassify = classify (filename="step1.root", variables=usedVariables, input_tree=tree)

    method_name = "NNPG"
    weightfile_name = "weights/Flavor_NNPG.weights.xml"
    
    retPredict = predict (filenames=["training","check_correlation","check_agreement"], method_name=method_name, weightfile_name=weightfile_name, execute_tests=True, variables=usedVariables)


    



def competition ():
    tree = load (filenames=[training_filename])
    retClassify = classify (filename="step1.root", variables=usedVariables, input_tree=tree, method_suffix="1st")

    method_name = retClassify["method_name"]
    weightfile_name = retClassify["weightfile_name"]
    
#    retPredict = predict (filenames=["training","check_agreement","check_correlation"], method_name=method_name, weightfile_name=weightfile_name, execute_tests=True, variables=usedVariables)
    retPredict = predict (filenames=["training","check_agreement","check_correlation","test"], method_name=method_name, weightfile_name=weightfile_name, execute_tests=True, variables=usedVariables)

    twostage = True
    if twostage:
        training_prediction = retPredict["training_prediction_root"]

        tree2nd = load (filenames=[training_filename, training_prediction])
        retClassify2nd = classify (filename="step2.root", variables=usedVariables, input_tree=tree2nd, signal_cut="signal==0 && prediction > 0.6", background_cut="signal==0 && prediction > 0.6", method_suffix="2nd")

        method_name2nd = retClassify2nd["method_name"]
        weightfile_name2nd = retClassify2nd["weightfile_name"]
    
    
        retPredict2nd = predict (filenames=["training","check_agreement","check_correlation","test"], method_name=method_name2nd, weightfile_name=weightfile_name2nd, execute_tests=False, variables=usedVariables, prediction_name="sim")




def loadAgreement (method_name):        
    return load (filenames=[default_path+"check_agreement.root", "check_agreement_p_%s.root"%method_name])

def loadAgreementSim (method_name, method_name2):        
    return load (filenames=[default_path+"check_agreement.root", "check_agreement_p_%s.root"%method_name, "check_agreement_p_%s.root"%method_name2])


def showAgreement (method_name):
    t = loadAgreement (method_name)
    h = t.Draw ("prediction","weight*(signal==0)/1400","")
    t.Draw ("prediction","weight*(signal==1)/500","same")
    return h

        
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    competition ()
    #testPrediction ()
    


