#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include <chrono>
#include <ctime>
#include <algorithm>
 
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

//#if not defined(__CINT__) || defined(__MAKECINT__)
//needs to be included when makecint runs (ACLIC)
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
//#endif

//#include "tmvagui/inc/TMVA/tmvagui.h"

//TString pathToData ("/home/peters/test/kaggle_flavour/flavours-of-physics-start/tau_data/");
TString pathToData ("/home/peter/code/kaggle/flavor/");



template <typename C>
C operator+ (const C& lhs, const C& rhs)
{
    C tmp (lhs);
    tmp.insert (tmp.end (), rhs.begin (), rhs.end ());
    return tmp;
};



std::string now ()
{
    auto _now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(_now);

    struct tm * timeinfo;
    char tmstr [80];

    time (&now_time);
    timeinfo = localtime (&now_time);

    strftime (tmstr,80,"%Y%m%d_%H%M",timeinfo);
    return std::string (tmstr);
}









std::vector<std::string> variableNames = {
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
    ,"p2_track_Chi2Dof" // spoils agreement test
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
    ,"SPDhits" // spoils agreement test
};


std::vector<std::string> spectatorNames = {
    "mass",
    "min_ANNmuon"
};


std::vector<std::string> additionalVariableNames = {
    "signal"
};



TString autoencoder (std::string inputFileName) 
{

    std::string tmstr (now ());
    TString tmstmp (tmstr.c_str ());
   
  
    std::cout << "==> Start Autoencoder " << std::endl;
    std::cout << "-------------------- open input file ---------------- " << std::endl;
    TString fname = pathToData + TString (inputFileName.c_str ()) + TString (".root");
    TFile *input = TFile::Open( fname );

    std::cout << "-------------------- get tree ---------------- " << std::endl;
    TTree *tree     = (TTree*)input->Get("data");
   
    TString outfileName( "TMVAAutoEnc__" );
    outfileName += TString (inputFileName.c_str ()) + TString ("__") + tmstmp + TString (".root");

    std::cout << "-------------------- open output file ---------------- " << std::endl;
    TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

    std::cout << "-------------------- prepare factory ---------------- " << std::endl;
    TMVA::Factory *factory = new TMVA::Factory( "TMVAAutoencoder", outputFile,
						"AnalysisType=Regression:Color:DrawProgressBar" );
    std::cout << "-------------------- add variables ---------------- " << std::endl;


    for (auto varname : variableNames+additionalVariableNames)
    {
	factory->AddVariable (varname.c_str (), 'F');
	factory->AddTarget   (varname.c_str (), 'F');
    }



    std::cout << "-------------------- add tree ---------------- " << std::endl;
    // global event weights per tree (see below for setting event-wise weights)
    Double_t regWeight  = 1.0;   
    factory->AddRegressionTree (tree, regWeight);

   
    std::cout << "-------------------- prepare ---------------- " << std::endl;
    TCut mycut = ""; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";
    factory->PrepareTrainingAndTestTree( mycut, 
					 "nTrain_Regression=0:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V" );


    /* // This would set individual event weights (the variables defined in the  */
    /* // expression need to exist in the original TTree) */
    /* factory->SetWeightExpression( "var1", "Regression" ); */


    if (true)
    {
	TString layoutString ("Layout=TANH|100,TANH|20,TANH|40,LINEAR");

	TString training0 ("LearningRate=1e-5,Momentum=0.5,Repetitions=1,ConvergenceSteps=500,BatchSize=50,TestRepetitions=7,WeightDecay=0.01,Regularization=NONE,DropConfig=0.5+0.5+0.5+0.5,DropRepetitions=2");
	TString training1 ("LearningRate=1e-5,Momentum=0.9,Repetitions=1,ConvergenceSteps=500,BatchSize=30,TestRepetitions=7,WeightDecay=0.01,Regularization=L2,DropConfig=0.1+0.1+0.1,DropRepetitions=1");
	TString training2 ("LearningRate=1e-4,Momentum=0.3,Repetitions=1,ConvergenceSteps=10,BatchSize=40,TestRepetitions=7,WeightDecay=0.1,Regularization=L2");
	TString training3 ("LearningRate=1e-5,Momentum=0.1,Repetitions=1,ConvergenceSteps=10,BatchSize=10,TestRepetitions=7,WeightDecay=0.001,Regularization=NONE");

	TString trainingStrategyString ("TrainingStrategy=");
	trainingStrategyString += training0 + "|" + training1 + "|" + training2 ; //+ "|" + training3;

       
	//       TString trainingStrategyString ("TrainingStrategy=LearningRate=1e-1,Momentum=0.3,Repetitions=3,ConvergenceSteps=20,BatchSize=30,TestRepetitions=7,WeightDecay=0.0,L1=false,DropFraction=0.0,DropRepetitions=5");

	TString nnOptions ("!H:V:ErrorStrategy=SUMOFSQUARES:VarTransform=N:WeightInitialization=XAVIERUNIFORM");
	//       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CHECKGRADIENTS");
	nnOptions.Append (":"); nnOptions.Append (layoutString);
	nnOptions.Append (":"); nnOptions.Append (trainingStrategyString);

	factory->BookMethod( TMVA::Types::kNN, TString("NN_")+tmstmp, nnOptions ); // NN
    }


   
    // --------------------------------------------------------------------------------------------------
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    outputFile->Close();

//    TMVA::TMVARegGui (outfileName);
   
    delete factory;
    return TString("NN_")+tmstmp;
}



TString useAutoencoder (TString method_name)
{
    TMVA::Tools::Instance();

    std::cout << "==> Start useAutoencoder" << std::endl;
    TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

    Float_t signal = 0.0;
    Float_t outSignal = 0.0;
    Float_t inSignal = 0.0;

    std::vector<std::string> localVariableNames (variableNames+additionalVariableNames);
  
    std::vector<Float_t> variables (localVariableNames.size ());
    auto itVar = begin (variables);
    for (auto varName : localVariableNames)
    {
	Float_t* pVar = &(*itVar);
	reader->AddVariable(varName.c_str(), pVar);
	(*itVar) = 0.0;
	++itVar;
    }
    int idxSignal = std::distance (localVariableNames.begin (),
				   std::find (localVariableNames.begin (), localVariableNames.end (),std::string ("signal")));

  
    TString dir    = "weights/";
    TString prefix = "TMVAAutoencoder";
    TString weightfile = dir + prefix + TString("_") + method_name + TString(".weights.xml");
    TString outPrefix = "transformed";
    TString outfilename = pathToData + outPrefix + TString("_") + method_name + TString(".root");
    reader->BookMVA( method_name, weightfile );

  
    TFile* outFile = new TFile (outfilename.Data (), "RECREATE");

  
  
    std::vector<std::string> inputNames = {"training"};
    std::map<std::string,std::vector<std::string>> varsForInput;
    varsForInput["training"].emplace_back ("id");
    varsForInput["training"].emplace_back ("signal");

  
    for (auto inputName : inputNames)
    {
	std::stringstream outfilename;
	outfilename << inputName << "_transformed__" << method_name.Data () << ".root";
	std::cout << outfilename.str () << std::endl;
	/* return; */
      
	std::stringstream infilename;
	infilename << pathToData.Data () << inputName << ".root";

	TTree* outTree = new TTree("transformed","transformed");
      
	std::vector<Float_t> outVariables (localVariableNames.size ());
	itVar = begin (variables);
	auto itOutVar = begin (outVariables);
	for (auto varName : localVariableNames)
        {
	    Float_t* pOutVar = &(*itOutVar);
	    outTree->Branch (varName.c_str (), pOutVar, "F");
	    (*itOutVar) = 0.0;
	    ++itOutVar;

	    Float_t* pVar = &(*itVar);
	    std::stringstream svar;
	    svar << varName << "_in";
	    outTree->Branch (svar.str ().c_str (), pVar, "F");
	    (*itVar) = 0.0;
	    ++itVar;
        }
	Float_t signal_original = 0.0;
	outTree->Branch ("signal_original", &signal_original, "F");

	TFile *input(0);
	std::cout << "infilename = " << infilename.str ().c_str () << std::endl;
	input = TFile::Open (infilename.str ().c_str ());
	TTree* tree = (TTree*)input->Get("data");
  
	Int_t ids;

	// id field if needed
	if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "id") != varsForInput[inputName].end ())
	    tree->SetBranchAddress("id", &ids);

      
	// variables for prediction
	itVar = begin (variables);
	for (auto inputName : localVariableNames)
        {
	    Float_t* pVar = &(*itVar);
	    tree->SetBranchAddress (inputName.c_str(), pVar);
	    ++itVar;
        }
 
	for (Long64_t ievt=0; ievt < tree->GetEntries(); ievt++)
        {
	    tree->GetEntry(ievt);
	    // predict

	    signal_original = variables.at (idxSignal);
	    for (int forcedSignal = 0; forcedSignal <= 1; ++forcedSignal)
            {
		variables.at (idxSignal) = forcedSignal;
		std::vector<Float_t> regressionValues = reader->EvaluateRegression (method_name);
		size_t idx = 0;
		for (auto it = std::begin (regressionValues), itEnd = std::end (regressionValues); it != itEnd; ++it)
                {
		    outVariables.at (idx) = *it;
		    ++idx;
                }
		outTree->Fill ();
            }
          
        }

	outFile->Write ();
	input->Close();
    }
    delete reader;
    return outfilename;
}





void createCDF ()
{
    std::cout << "==> create CDF" << std::endl;

    std::vector<std::string> inputNames = {"training"};

  
    for (auto inputName : inputNames)
    {
	std::stringstream outfilename;
	outfilename << inputName << "_cdf__" << inputName << ".root";
	std::cout << outfilename.str () << std::endl;
	/* return; */
      
	std::stringstream infilename;
	infilename << pathToData.Data () << inputName << ".root";
          

	TFile *input(0);
	std::cout << "infilename = " << infilename.str ().c_str () << std::endl;
	input = TFile::Open (infilename.str ().c_str ());
	TTree* tree  = (TTree*)input->Get("data");
  
      
	// variables for prediction
	std::cout << "prepare variables" << std::endl;
	auto localVariableNames = variableNames+additionalVariableNames;
	std::vector<Float_t> variables (localVariableNames.size ());
	auto itVar = begin (variables);
	for (auto inputName : localVariableNames)
        {
	    Float_t* pVar = &(*itVar);
	    tree->SetBranchAddress(inputName.c_str(), pVar);
	    ++itVar;
        }

	Int_t id;
	// id field 
	tree->SetBranchAddress("id", &id);

	
	Long64_t ievtEnd = tree->GetEntries ();
	ievtEnd = 100;
	std::cout << "process entries #" << ievtEnd << std::endl;
	std::vector<double> sumSmaller (ievtEnd, 0.0);

	struct Vars
	{
	    typedef std::vector<Float_t>::const_iterator iterator;
	    Vars (iterator itBegin, iterator itEnd, Float_t _weight, Int_t _id, Long64_t _order)
	    : variables (itBegin, itEnd)
	    , weight (_weight)
	    , id (_id)
	    , order (_order)
	    {
	    }
	    
	    std::vector<Float_t> variables;
	    Int_t id;
	    Float_t weight;
	    Float_t cdf;
	    Long64_t order;

	    bool operator< (const Vars& other) const
	    {
		for (auto itOther = begin (other.variables), it = begin (variables),
			 itOtherEnd = end (other.variables), itEnd = end (variables);
		     it != itEnd && itOther != itOtherEnd; ++itOther, ++it)
                {
		    //std::cout << "(" << *it << "," << *itOther << ")" << std::flush;
		    if (*it >= *itOther)
		    {
//			std::cout << "X" << std::flush;
			return false;
		    }
		    else
			std::cout << "D" << std::flush;
                }
		std::cout << "U" << std::flush;
		return true;
	    }
	};
	
	Float_t weightSum (0.0);
	std::vector<Vars> vars;
	for (Long64_t ievt=0; ievt < ievtEnd; ievt++)
        {
	    tree->GetEntry (ievt);
	    std::cout << "." << std::flush;
	    Float_t weight = 1.0;
	    vars.emplace_back (begin (variables), end (variables), weight, id, ievt);
	    weightSum += weight;
        }

	
	std::cout << "provide values" << std::endl;
	for (auto it = begin (vars), itEnd = end (vars); it != itEnd; ++it)
	{
	    std::cout << "-" << std::flush;
	    for (auto itCmp = begin (vars), itCmpEnd = end (vars); itCmp != itCmpEnd; ++itCmp)
	    {
		if (*it < *itCmp)
		{
		    std::cout << "!" << std::flush;
		    break;
		}
		else
		{
		    std::cout << "+" << std::flush;
		    (*it).cdf += (*itCmp).weight;
		}
	    }
	}
	
	
	std::cout << "normalize" << std::endl;
	for_each (begin (vars), end (vars), [weightSum](Vars& v) {
		v.cdf /= weightSum;
	    });

	// sort by order
	std::sort (begin (vars), end (vars), [](const Vars& lhs, const Vars& rhs){
		return lhs.order < rhs.order;
	    });
	

	
	input->Close();
	std::cout << "store data" << std::endl;

	TFile* outFile = new TFile (outfilename.str ().c_str (), "RECREATE");
	TTree* outTree = new TTree("cdf_raw","cdf_raw");
	Float_t cdf (0.0);
	outTree->Branch ("id", &id, "F");
	outTree->Branch ("cdf", &cdf, "F");
	for (auto v : vars)
	{
	    id = v.id;
	    cdf = v.cdf;
	    outTree->Fill ();
	}
	outFile->Write ();
	outFile->Close ();
    }
    
}











TString TMVAClassification(TString infilename, bool useTransformed = false)
{
    TMVA::Tools::Instance();

    std::string tmstr (now ());
    TString tmstmp (tmstr.c_str ());
   
  
    std::cout << "==> Start TMVAClassification" << std::endl;
    std::cout << "-------------------- open input file ---------------- " << std::endl;
    TString fname = infilename; //pathToData + infilename + TString (".root");
    if (!useTransformed)
	fname = pathToData + infilename + TString (".root");
    TFile *input = TFile::Open( fname );

    std::cout << "-------------------- get tree ---------------- " << std::endl;
    TString treeName = "data";
    if (useTransformed)
        treeName = "transformed";
    TTree *tree     = (TTree*)input->Get(treeName);
   
    TString outfileName( "TMVA__" );
    outfileName += tmstmp + TString (".root");

    std::cout << "-------------------- open output file ---------------- " << std::endl;
    TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

    std::cout << "-------------------- prepare factory ---------------- " << std::endl;
    TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
						"AnalysisType=Classification" );
    std::cout << "-------------------- add variables ---------------- " << std::endl;


    for (auto varname : variableNames)
    {
	factory->AddVariable (varname.c_str (), 'F');
    }
   
    for (auto varname : spectatorNames)
    {
	factory->AddSpectator (varname.c_str (), 'F');
    }
   
   
    std::cout << "-------------------- add trees ---------------- " << std::endl;
    TCut signalCut ("signal==1");
    TCut backgroundCut ("signal==0");
    if (useTransformed)
    {
        signalCut = "signal_original==1 && signal_in==0";
        backgroundCut = "signal_original==0 && signal_in==0";
    }
    factory->AddTree(tree, "Signal", 1.0, signalCut, "TrainingTesting");
    factory->AddTree(tree, "Background", 1.0, backgroundCut, "TrainingTesting");


    
    TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
    TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

    /* // Set individual event weights (the variables must exist in the original TTree) */
    /* factory->SetSignalWeightExpression( "weight" ); */
    /* factory->SetBackgroundWeightExpression( "weight" ); */

   
    std::cout << "-------------------- prepare ---------------- " << std::endl;
    factory->PrepareTrainingAndTestTree( mycuts, mycutb,
					 "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );


    TString methodName ("");

    if (false)
    {
	// gradient boosting training
        methodName = TString("GBDT__")+tmstmp;
	factory->BookMethod(TMVA::Types::kBDT, methodName,
			    "NTrees=40:BoostType=Grad:Shrinkage=0.01:MaxDepth=7:UseNvars=6:nCuts=20:MinNodeSize=10");
    }
    if (false)
    {
        methodName = TString("Likelihood__")+tmstmp;
	factory->BookMethod( TMVA::Types::kLikelihood, methodName,
			     "H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );
    }
    

    
    if (false)
    {
	TString layoutString ("Layout=TANH|100,LINEAR");

	TString training0 ("LearningRate=1e-1,Momentum=0.0,Repetitions=1,ConvergenceSteps=300,BatchSize=20,TestRepetitions=15,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True");
	TString training1 ("LearningRate=1e-2,Momentum=0.5,Repetitions=1,ConvergenceSteps=300,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1");
	TString training2 ("LearningRate=1e-2,Momentum=0.3,Repetitions=1,ConvergenceSteps=300,BatchSize=40,TestRepetitions=7,WeightDecay=0.0001,Regularization=L2,Multithreading=True");
	TString training3 ("LearningRate=1e-3,Momentum=0.1,Repetitions=1,ConvergenceSteps=200,BatchSize=70,TestRepetitions=7,WeightDecay=0.0001,Regularization=NONE,Multithreading=True");

	TString trainingStrategyString ("TrainingStrategy=");
	trainingStrategyString += training0 + "|" + training1 + "|" + training2 + "|" + training3;
      
	TString nnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=G:WeightInitialization=XAVIERUNIFORM");
	nnOptions.Append (":"); nnOptions.Append (layoutString);
	nnOptions.Append (":"); nnOptions.Append (trainingStrategyString);

        methodName = TString("NNgauss_")+tmstmp;
	factory->BookMethod( TMVA::Types::kNN, methodName, nnOptions ); // NN
    }

    if (true)
    {
	TString layoutString ("Layout=TANH|100,LINEAR");

	TString training0 ("LearningRate=1e-1,Momentum=0.0,Repetitions=1,ConvergenceSteps=300,BatchSize=20,TestRepetitions=15,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True");
	TString training1 ("LearningRate=1e-2,Momentum=0.5,Repetitions=1,ConvergenceSteps=300,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1");
	TString training2 ("LearningRate=1e-2,Momentum=0.3,Repetitions=1,ConvergenceSteps=300,BatchSize=40,TestRepetitions=7,WeightDecay=0.0001,Regularization=L2,Multithreading=True");
	TString training3 ("LearningRate=1e-3,Momentum=0.1,Repetitions=1,ConvergenceSteps=200,BatchSize=70,TestRepetitions=7,WeightDecay=0.0001,Regularization=NONE,Multithreading=True");

	TString trainingStrategyString ("TrainingStrategy=");
	trainingStrategyString += training0 + "|" + training1 + "|" + training2 + "|" + training3;

      
	//       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CROSSENTROPY");
	TString nnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM");
	//       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CHECKGRADIENTS");
	nnOptions.Append (":"); nnOptions.Append (layoutString);
	nnOptions.Append (":"); nnOptions.Append (trainingStrategyString);

        methodName = TString("NNnormalized_")+tmstmp;
        factory->BookMethod( TMVA::Types::kNN, methodName, nnOptions ); // NN
    }


    if (false)
    {
	TString layoutString ("Layout=TANH|100,TANH|50,LINEAR");

	TString training0 ("LearningRate=1e-1,Momentum=0.0,Repetitions=1,ConvergenceSteps=300,BatchSize=20,TestRepetitions=15,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True");
	TString training1 ("LearningRate=1e-2,Momentum=0.5,Repetitions=1,ConvergenceSteps=300,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1");
	TString training2 ("LearningRate=1e-2,Momentum=0.3,Repetitions=1,ConvergenceSteps=300,BatchSize=40,TestRepetitions=7,WeightDecay=0.0001,Regularization=L2,Multithreading=True");
	TString training3 ("LearningRate=1e-3,Momentum=0.1,Repetitions=1,ConvergenceSteps=200,BatchSize=70,TestRepetitions=7,WeightDecay=0.0001,Regularization=NONE,Multithreading=True");

	TString trainingStrategyString ("TrainingStrategy=");
	trainingStrategyString += training0 + "|" + training1 + "|" + training2 + "|" + training3;

      
	//       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CROSSENTROPY");
	TString nnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM");
	//       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CHECKGRADIENTS");
	nnOptions.Append (":"); nnOptions.Append (layoutString);
	nnOptions.Append (":"); nnOptions.Append (trainingStrategyString);

        methodName = TString("NN_normalized_2")+tmstmp;
	factory->BookMethod( TMVA::Types::kNN, methodName, nnOptions ); // NN
    }
   
   
   
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    //input->Close();
    outputFile->Close();

//    TMVA::TMVAGui (outfileName);
   
    delete factory;
    return methodName;
}


void TMVAPredict(TString method_name)
{
    TMVA::Tools::Instance();

    std::cout << "==> Start TMVAPredict" << std::endl;
    TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );  


    std::vector<Float_t> variables (variableNames.size ());
    auto itVar = begin (variables);
    for (auto varName : variableNames)
    {
	Float_t* pVar = &(*itVar);
	reader->AddVariable(varName.c_str(), pVar);
	(*itVar) = 0.0;
	++itVar;
    }

    for (auto varName : spectatorNames)
    {
	Float_t var;
	reader->AddSpectator (varName.c_str(), &var);
	++itVar;
    }

    TString dir    = "weights/";
    TString prefix = "TMVAClassification";
    TString weightfile = dir + prefix + TString("_") + method_name + TString(".weights.xml");
    reader->BookMVA( method_name, weightfile ); 

  

  
    std::vector<std::string> inputNames = {"training","test","check_correlation","check_agreement"};
    std::map<std::string,std::vector<std::string>> varsForInput;
    varsForInput["training"].emplace_back ("id");
    varsForInput["training"].emplace_back ("signal");
    varsForInput["training"].emplace_back ("mass");
    varsForInput["training"].emplace_back ("min_ANNmuon");
    varsForInput["training"].emplace_back ("prediction");

    varsForInput["test"].emplace_back ("id");
    varsForInput["test"].emplace_back ("prediction");

    varsForInput["check_agreement"].emplace_back ("signal");
    varsForInput["check_agreement"].emplace_back ("weight");
    varsForInput["check_agreement"].emplace_back ("prediction");

    varsForInput["check_correlation"].emplace_back ("mass");
    varsForInput["check_correlation"].emplace_back ("prediction");


  
    for (auto inputName : inputNames)
    {
	std::stringstream outfilename;
	outfilename << inputName << "_prediction__" << method_name.Data () << ".csv";
	std::cout << outfilename.str () << std::endl; 
	/* return; */
      
	std::stringstream infilename;
	infilename << pathToData.Data () << inputName << ".root";
          
	std::ofstream outfile (outfilename.str ());
	bool isFirst = true;
	for (auto inputName : varsForInput[inputName])
        {
	    if (!isFirst)
            {
		outfile << ",";
            }
	    else
		isFirst = false;
	    outfile << inputName;
        }
	outfile << "\n";


	TFile *input(0);
	std::cout << "infilename = " << infilename.str ().c_str () << std::endl;
	input = TFile::Open (infilename.str ().c_str ());
	TTree* tree = (TTree*)input->Get("data");
  
	Int_t ids;
	Float_t prediction;
	Float_t weight;
	Float_t min_ANNmuon;
	Float_t mass;
	Float_t signal;

	// id field if needed
	if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "id") != varsForInput[inputName].end ())
	    tree->SetBranchAddress("id", &ids);

	// signal field if needed
	if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "signal") != varsForInput[inputName].end ())
	    tree->SetBranchAddress("signal", &signal);

	// min_ANNmuon field if needed
	if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "min_ANNmuon") != varsForInput[inputName].end ())
	    tree->SetBranchAddress("min_ANNmuon", &min_ANNmuon);

	// mass field if needed
	if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "mass") != varsForInput[inputName].end ())
	    tree->SetBranchAddress("mass", &mass);

	// weight field if needed
	if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "weight") != varsForInput[inputName].end ())
	    tree->SetBranchAddress("weight", &weight);

      
	// variables for prediction
	itVar = begin (variables);
	for (auto inputName : variableNames)
        {
	    Float_t* pVar = &(*itVar);
	    tree->SetBranchAddress(inputName.c_str(), pVar);
	    ++itVar;
        }
	
 
	for (Long64_t ievt=0; ievt < tree->GetEntries(); ievt++)
        {
	    tree->GetEntry(ievt);
	    // predict
	    prediction = reader->EvaluateMVA (method_name);

	    if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "id") != varsForInput[inputName].end ())
		outfile << ids << ",";

	    if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "signal") != varsForInput[inputName].end ())
		outfile << signal << ",";

	    if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "min_ANNmuon") != varsForInput[inputName].end ())
		outfile << min_ANNmuon << ",";

	    if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "mass") != varsForInput[inputName].end ())
		outfile << mass << ",";

	    if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "weight") != varsForInput[inputName].end ())
		outfile << weight << ",";

	    if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "prediction") != varsForInput[inputName].end ())
		outfile << (prediction + 1.) / 2.;

          
	    outfile << "\n";
        }

	outfile.close();
	input->Close();
    }
    delete reader;

    TString cmd (".! python tests.py ");
    cmd += method_name;
    gROOT->ProcessLine (cmd);
}


int competitionAutoEnc ()
{
    TString methodNameAutoEncoder = autoencoder ("training");
    TString trainingFileName = useAutoencoder (methodNameAutoEncoder);
    TString methodNameClassification = TMVAClassification (trainingFileName, true);
    TMVAPredict (methodNameClassification);

    return 0;
}


int competitionDirect ()
{
    TString trainingFileName ("training");
    TString methodNameClassification = TMVAClassification (trainingFileName, false);
    TMVAPredict (methodNameClassification);

    return 0;
}

