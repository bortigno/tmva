#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include <chrono>
#include <ctime>
 
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


TString pathToData ("/home/peters/test/kaggle_flavour/flavours-of-physics-start/tau_data/");



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
//   ,"p2_track_Chi2Dof" // spoils agreement test
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
//   ,"SPDhits" // spoils agreement test
  };






void TMVAClassification()
{
   TMVA::Tools::Instance();

   std::string tmstr (now ());
   TString tmstmp (tmstr.c_str ());
   
  
   std::cout << "==> Start TMVAClassification" << std::endl;
   std::cout << "-------------------- open input file ---------------- " << std::endl;
   TString fname = pathToData + TString ("training.root");
   TFile *input = TFile::Open( fname );

   std::cout << "-------------------- get tree ---------------- " << std::endl;
   TTree *tree     = (TTree*)input->Get("data");
   
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
   
   
   std::cout << "-------------------- add trees ---------------- " << std::endl;
   factory->AddTree(tree, "Signal", 1.0, TCut("signal==1"), "TrainingTesting");
   factory->AddTree(tree, "Background", 1.0, TCut("signal==0"), "TrainingTesting");

   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";
   
   std::cout << "-------------------- prepare ---------------- " << std::endl;
   factory->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );


   // gradient boosting training
   factory->BookMethod(TMVA::Types::kBDT, TString ("GBDT_")+tmstmp,
                       "NTrees=40:BoostType=Grad:Shrinkage=0.01:MaxDepth=7:UseNvars=6:nCuts=20:MinNodeSize=10");

   factory->BookMethod( TMVA::Types::kLikelihood, TString ("Likelihood_")+tmstmp,
                        "H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );


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

       factory->BookMethod( TMVA::Types::kNN, TString("NNgauss_")+tmstmp, nnOptions ); // NN
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

       factory->BookMethod( TMVA::Types::kNN, TString ("NNnormalized_")+tmstmp, nnOptions ); // NN
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

       factory->BookMethod( TMVA::Types::kNN, "NN_normalized_2", nnOptions ); // NN
   }
   
   
   
   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();

   //input->Close();
   outputFile->Close();

   TMVA::TMVAGui (outfileName);
   
   delete factory;
}


void TMVAPredict(TString method_name)
{
  TMVA::Tools::Instance();

  std::cout << "==> Start TMVAPredict" << std::endl;
  TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );  


//  Float_t variables[3];
  std::vector<Float_t> variables (variableNames.size ());
  auto itVar = begin (variables);
  for (auto varName : variableNames)
  {
      Float_t* pVar = &(*itVar);
      reader->AddVariable(varName.c_str(), pVar);
      (*itVar) = 0.0;
      ++itVar;
  }

  TString dir    = "weights/";
  TString prefix = "TMVAClassification";
  TString weightfile = dir + prefix + TString("_") + method_name + TString(".weights.xml");
  reader->BookMVA( method_name, weightfile ); 

  

  
  std::vector<std::string> inputNames = {"test","check_correlation","check_agreement"};
  std::map<std::string,std::vector<std::string>> varsForInput;
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
      Float_t mass;
      Float_t signal;

      // id field if needed
      if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "id") != varsForInput[inputName].end ())
          tree->SetBranchAddress("id", &ids);

      // signal field if needed
      if (std::find (varsForInput[inputName].begin (), varsForInput[inputName].end (), "signal") != varsForInput[inputName].end ())
          tree->SetBranchAddress("signal", &signal);

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


int competition()
{
  TMVAClassification();
  cout << "Classifier have been trained\n";
//  TMVAPredict();
//  cout << "Submission is ready: baseline_c.csv; send it\n";
  return 0;
}
