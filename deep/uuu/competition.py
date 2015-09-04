from ROOT import *
import array
import csv


variableNames = [
    "LifeTime"
    ,"FlightDistance"
    ,"FlightDistanceError"
    ,"pt"
    ,"IP"
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
    index = 0
    for filename in input_filenames:
        if not tree:
            print "not tree"
            tree = TChain (input_treename)
            tree.Add (filename)
        else:
            print "has tree"
            tmp = TChain (input_treename)
            tmp.Add (filename)
            friendname = "p"+str(index)
            print friendname
            ++index
            tree.AddFriend (tmp, friendname)

    return tree



def classify (**kwargs):
    print "------------ classification ---------------"
    for key,value in kwargs.iteritems ():
        print key," = ",value
        
    TMVA.Tools.Instance ()

    output_filename = kwargs.setdefault ("filename", None)
    input_variables = kwargs.setdefault ("variables", None)
    input_spectators = kwargs.setdefault ("spectators", [])
    signal_cut = TCut (kwargs.setdefault ("signal_cut", "signal==1"))
    background_cut = TCut (kwargs.setdefault ("background_cut", "signal==0"))
    base_cut = TCut (kwargs.setdefault ("base_cut", "(LifeTime >= 0 && FlightDistance >= 0)"))
    input_tree = kwargs.setdefault ("input_tree", None)
    method_suffix = kwargs.setdefault ("method_suffix", "")
    
    outputFile = TFile.Open (output_filename, "RECREATE" )

    jobName = "Flavor"
    factory = TMVA.Factory( jobName, outputFile, "AnalysisType=Classification:Transformations=I:!V" )
    for var in input_variables:
        factory.AddVariable (var, 'F')
        
    for spec in input_spectators:
        factory.AddSpectator (spec, 'F')

    factory.AddTree (input_tree, "Signal", 1.0, base_cut + signal_cut, "TrainingTesting");
    factory.AddTree (input_tree, "Background", 1.0, base_cut + background_cut, "TrainingTesting");

    mycuts = TCut ()
    mycutb = TCut ()
    factory.PrepareTrainingAndTestTree (mycuts, mycutb, "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V")


    layoutString = "Layout=TANH|100,TANH|50,LINEAR"

    trainingConfig = [
        "LearningRate=1e-2,Momentum=0.0,Repetitions=1,ConvergenceSteps=10,BatchSize=20,TestRepetitions=7,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True",
        "LearningRate=1e-3,Momentum=0.0,Repetitions=1,ConvergenceSteps=2,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1",
        "LearningRate=1e-4,Momentum=0.0,Repetitions=1,ConvergenceSteps=2,BatchSize=40,TestRepetitions=7,WeightDecay=0.0001,Regularization=L2,Multithreading=True",
        "LearningRate=1e-5,Momentum=0.0,Repetitions=1,ConvergenceSteps=3,BatchSize=70,TestRepetitions=7,WeightDecay=0.0001,Regularization=NONE,Multithreading=True"
    ]

    trainingStrategy = "TrainingStrategy="
    for idx, conf in enumerate (trainingConfig):
        if idx != 0:
            trainingStrategy += "|"
        trainingStrategy += conf
            
    nnOptions = "!H:!V:ErrorStrategy=CROSSENTROPY:VarTransform=P+G:WeightInitialization=XAVIERUNIFORM"
    nnOptions += ":"+layoutString + ":" + trainingStrategy
        
    methodName = "NNPG"+method_suffix
    factory.BookMethod (TMVA.Types.kNN, methodName, nnOptions)

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
        vname = varname.toupper ()
    cmd = "%s = array.array ('%s',[0])\n"%(varname,vtype)
    cmd = cmd + 'if "%s" in variablesForFiles[currentFileName]:\n'%(varname)
    cmd = cmd + '    tree.SetBranchAddress ("%1s",%2s)\n'%(varname,vname)
    return cmd

def branch (varname):
    vname = varname
    vtype = 'f'
    if varname == "id":
        vtype = 'i'
        vname = varname.toupper ()
    cmd = 'outTree.Branch ("%1s",%2s,"F")\n'%(varname,vname)
    return cmd



                
    
def predict (**kwargs):
    filenames = kwargs.setdefault ("filenames", ["training","test","check_correlation","check_agreement"])
    variableOrder = kwargs.setdefault ("variable_order", ["id", "signal", "mass", "min_ANNmuon", "prediction"])

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
        "training" : ["root"],
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


    TMVA.Tools.Instance ()

    reader = TMVA.Reader( "!Color:!Silent" )

    variables = [array.array('f',[0])]*len (input_variables)
    for idx, var_name in enumerate (input_variables):
        variables
        reader.AddVariable (var_name, variables[idx])

    spectators = [array.array('f',[0])]*len (input_spectators)
    for idx, spec_name in enumerate (input_spectators):
        reader.AddVariable (spec_name, spectators[idx])


    reader.BookMVA (method_name, weightfile_name)

    returnValues = {}
    for currentFileName in filenames:
        fileName = default_path + currentFileName + ".root"
        
        # define variables
        ID = array.array ('i',[0])
        outputVariables = ["prediction","weight","min_ANNmuon","mass","signal"]

        prediction = array.array ('f',[0])
        weight = array.array ('f',[0])
        min_ANNmuon = array.array ('f',[0])
        mass = array.array ('f',[0])
        signal = array.array ('f',[0])
       
        # --- open input file
        f = TFile.Open (fileName)
	tree = f.Get("data");


        for v in outputVariables:
            if v != "prediction":
                cmd = setbranch (v)
                print cmd
                exec (cmd)

      
	# variables for prediction
        variables = [array.array ('f',[0])]*len (input_variables)
	for idx, currentVariableName in enumerate (input_variables):
	    tree.SetBranchAddress (currentVariableName, variables[idx]);
        
        doCSV = "csv" in createForFiles[currentFileName]
        doROOT = "root" in createForFiles[currentFileName]
            
        # ---- make ROOT file if requested
        doROOT = False
        if doROOT and "root" in createForFiles[currentFileName]:
            rootFileName = currentFileName + "_p_" + method_name + ".root"
            outRootFile = TFile (rootFileName, "RECREATE")
            outTree = TTree ("data","data")

            for v in variablesForFiles[currentFileName]:
                cmd = branch (v)
                exec (cmd)

            curr = currentFileName + "_prediction_root"
            returnValues[curr] = rootFileName

        # ---- prepare csv file
        csvfile = None
        writer = None
        if doCSV and "csv" in createForFiles[currentFileName]:
            csvFileName = currentFileName + "_p_" + method_name + ".csv"
            with open (csvFileName, 'wb') as csvfile:
                writer = csv.writer (csvfile, delimiter = ",")
                vars = []
                for currentVariable in variableOrder:
                    if currentVariable in variablesForFiles[currentFileName]:
                        vars.append (currentVariable)
                writer.writerow (vars)

            curr = currentFileName + "_prediction_csv"
            returnValues[curr] = csvFileName

        # 
	for ievt in xrange (tree.GetEntries()):
	    tree.GetEntry (ievt)
	    # predict
	    prediction = reader.EvaluateMVA (method_name)
            prediction = max (0.0, min (1.0, prediction))
            # prediction = (prediction + 1.0)/2.0;
            if doCSV != None and csvfile != None:
                row = []
                for varName in variableOrder:
                    if varName in variablesForFiles[currentFileName]:
                        cmd = 'row.append (%s)'%varName
                        exec (cmd)
                writer.writerow (row)

            if doROOT:
                outTree.Write ()
                

        f.Close ()

        if doCSV:
            csvfile.close ()
            
        if doROOT:
            outRootFile.Close ()
    if execute_tests:
        import tests.py        
        
    return returnValues

            
    


    

# tree = load (filenames=[training_filename, training_filename])
# classify (filename="classtest.root", variables=variableNames, input_tree=tree)
            

    
    

def competition ():
    tree = load (filenames=[training_filename])
    retClassify = classify (filename="step1.root", variables=variableNames, input_tree=tree)

    method_name = retClassify["method_name"]
    weightfile_name = retClassify["weightfile_name"]
    
    retPredict = predict (filenames=["training","test","check_agreement","check_correlation"], method_name=method_name, weightfile_name=weightfile_name, execute_tests=True, variables=variableNames)

    training_prediction = retPredict["training_prediction_root"]

    # tree = load (filenames=[training_filename, training_prediction])
    # retClassify2nd = classify (filename="step2.root", variables=variableNames, input_tree=tree)

    
    # retPredict = predict (filenames=["training"], method_name=method_name, weightifle_name=weightfile_name)


    
competition ()
#print setbranch ("hallo")

    



