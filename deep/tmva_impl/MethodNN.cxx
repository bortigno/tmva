// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodNN                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A neural network implementation                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer      <peter.speckmayer@gmx.ch> - CERN, Switzerland       *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// neural network implementation
//_______________________________________________________________________

#include "TString.h"
#include "TTree.h"
#include "TFile.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodNN.h"
#include "TMVA/Timer.h"
#include "TMVA/Types.h"
#include "TMVA/Tools.h"
#include "TMVA/Config.h"

#include "TMVA/NeuralNet.h"
#include "TMVA/Monitoring.h"

REGISTER_METHOD(NN)

ClassImp(TMVA::MethodNN)


//______________________________________________________________________________
TMVA::MethodNN::MethodNN( const TString& jobName,
                          const TString& methodTitle,
                          DataSetInfo& theData,
                          const TString& theOption,
                          TDirectory* theTargetDir )
: MethodBase( jobName, Types::kNN, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
}

//______________________________________________________________________________
TMVA::MethodNN::MethodNN( DataSetInfo& theData,
                          const TString& theWeightFile,
                          TDirectory* theTargetDir )
   : MethodBase( Types::kNN, theData, theWeightFile, theTargetDir )
{
   // constructor from a weight file
}

//______________________________________________________________________________
TMVA::MethodNN::~MethodNN()
{
   // destructor
   // nothing to be done
}

//_______________________________________________________________________
Bool_t TMVA::MethodNN::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // MLP can handle classification with 2 classes and regression with one regression-target
   if (type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   if (type == Types::kMulticlass ) return kTRUE;
   if (type == Types::kRegression ) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
void TMVA::MethodNN::Init()
{
   // default initializations
}

//_______________________________________________________________________
void TMVA::MethodNN::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // know options:
   // TrainingMethod  <string>     Training method
   //    available values are:         BP   Back-Propagation <default>
   //                                  GA   Genetic Algorithm (takes a LONG time)
   //
   // LearningRate    <float>      NN learning rate parameter
   // DecayRate       <float>      Decay rate for learning parameter
   // TestRate        <int>        Test for overtraining performed at each #th epochs
   //
   // BPMode          <string>     Back-propagation learning mode
   //    available values are:         sequential <default>
   //                                  batch
   //
   // BatchSize       <int>        Batch size: number of events/batch, only set if in Batch Mode,
   //                                          -1 for BatchSize=number_of_events

   // DeclareOptionRef(fTrainMethodS="SD", "TrainingMethod",
   //                  "Train with back propagation steepest descend");
   // AddPreDefVal(TString("SD"));

   DeclareOptionRef(fLayout="TANH|(N+2)*2,TANH|(N+10),LINEAR",    "Layout",    "neural network layout");
   DeclareOptionRef(fErrorStrategy="MUTUALEXCLUSIVE",    "ErrorStrategy",    "error strategy (regression: sum of squares; classification: crossentropy; multiclass: crossentropy/mutual exclusive cross entropy");
   AddPreDefVal(TString("CROSSENTROPY"));
   AddPreDefVal(TString("SUMOFSQUARES"));
   AddPreDefVal(TString("MUTUALEXCLUSIVE"));

   DeclareOptionRef(fTrainingStrategy="LearningRate=1e-4,Momentum=0.3,Repetitions=3,ConvergenceSteps=100,BatchSize=70,TestRepetitions=7,WeightDecay=0.0,L1=false,DropFraction=0.4,DropRepetitions=5|LearningRate=1e-4,Momentum=0.3,Repetitions=3,ConvergenceSteps=100,BatchSize=70,TestRepetitions=7,WeightDecay=0.0,L1=false,DropFraction=0.4,DropRepetitions=5",    "TrainingStrategy",    "defines the training strategies");


}


std::vector<std::pair<int,NN::EnumFunction>> TMVA::MethodANNBase::ParseLayoutString(TString layerSpec)
{
    // parse layout specification string and return a vector, each entry
    // containing the number of neurons to go in each successive layer
    std::vector<std::pair<int,NN::EnumFunction>> layout;
    const TString delim_Layer (",");
    const TString delim_Sub ("|");

    const inputSize = GetNvar ();

    TObjArray* layerStrings = layerSpec.Tokenize (delim_Layer);
    TIter nextLayer (layerStrings);
    TString* layerString;
    for (layerString = (TString*)nextLayer ())
    {
        int numNodes = 0;
        NN::EnumFunction eActivationFunction = NN::TANH;

        TObjArray* subStrings = layerString->Tokenize (delim_Sub);
        TIter nextToken (subStrings);
        TString* token;
       
        int idxToken = 0;
        for (token = (TString*)nextToken ())
        {
            switch (idxToken)
            {
            case 0:
            {
                TString strActFnc (*token);
                if (strActFnc == "RELU")
                    eActivationFunction = NN::RELU;
                else if (strActFnc == "TANH")
                    eActivationFunction = NN::TANH;
                else if (strActFnc == "SYMMRELU")
                    eActivationFunction = NN::SYMMRELU;
                else if (strActFnc == "SOFTSIGN")
                    eActivationFunction = NN::SOFTSIGN;
                else if (strActFnc == "SIGMOID")
                    eActivationFunction = NN::SIGMOID;
                else if (strActFnc == "LINEAR")
                    eActivationFunction = NN::LINEAR;
                else if (strActFnc == "GAUSS")
                    eActivationFunction = NN::GAUSS;
            }
            case 1: // number of nodes
            {
                TString strNumNodes (*token);
                TString strN ("x");
                strNumNodes.ReplaceAll ("N", strN);
                strNumNodes.ReplaceAll ("n", strN);
                TFormulat fml ("tmp",strNumNodes);
                numNodes = fml.Eval (inputSize);
            }
            }
            layout.push_back (std::make_pair (numNodes,eActivationFunction));
            ++idxToken;
        }
    }
    return layout;
}



// parse key value pairs in blocks -> return vector of blocks with map of key value pairs
std::vector<std::map<TString,TString>> TMVA::MethodANNBase::ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim)
{
    std::vector<std::map<TString,TString>> blockKeyValues;
    const TString keyValueDelim ("=");

    const inputSize = GetNvar ();

    TObjArray* blockStrings = parseString.Tokenize (blockDelim);
    TIter nextBlock (blockStrings);
    TString* blockString;
    for (blockString = (TString*)nextBlock ())
    {
        blockKeyValues.push_back (std::map<TString,TString> ()); // new block
        std::map<TString,TString>& currentBlock = blockKeyValues.back ();

        TObjArray* subStrings = layerString->Tokenize (tokenDelim);
        TIter nextToken (subStrings);
        TString* token;
       
        for (token = (TString*)nextToken ())
        {
            TString strKeyValue = (*token);
            int delimPos = strKey.First (keyValueDelim.Data ());
            if (delimPos <= 0)
                continue;

            TString strKey = TString (strKeyValue (0, delimPos)).ToUpper ();
            TString strValue = TString (strKeyValue (delimPos+1, strKeyValue.Length ()));

            strKey.Strip (TString::kBoth, ' ');
            strValue.Strip (TString::kBoth, ' ');

            currentBlock.insert (std::make_pair (strKey, strValue));
        }
    }
    return blockKeyValues;
}


TString fetchValue (const std::map<TString, TString>& keyValueMap, TString key)
{
    std::map<TString, TString>::const_iterator it = keyValueMap.find (key.ToUpper ());
    if (it == keyValueMap.end ())
        return TString ("");
    return it->second;
}

template <typename T>
T fetchValue (const std::map<TString,TString>& keyValueMap, TString key, T defaultValue);

template <>
int fetchValue (const std::map<TString,TString>& keyValueMap, TString key, int defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    return value.Atoi ();
}

template <>
double fetchValue (const std::map<TString,TString>& keyValueMap, TString key, double defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    return value.Atof ();
}

template <>
TString fetchValue (const std::map<TString,TString>& keyValueMap, TString key, TString defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    return value;
}

template <>
bool fetchValue (const std::map<TString,TString>& keyValueMap, TString key, bool defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    if (value.ToUpper () == "TRUE" ||
        value.ToUpper () == "T" ||
        value.ToUpper () == "1")
        return true;
    return false;
}


//_______________________________________________________________________
void TMVA::MethodNN::ProcessOptions()
{
   // process user options
   MethodBase::ProcessOptions();

   
   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kINFO 
            << "Will ignore negative events in training!"
            << Endl;
   }


   //                                                                                         block-delimiter  token-delimiter
   std::vector<std::map<TString,TString>> strategyKeyValues = ParseKeyValueString (fStrategy, TString ("|"), TString (","));

   // create settings
   if (fAnalysisType == Types::kClassification)
   {

       if (fErrorStrategy == "SUMOFSQUARES") fModeErrorFunction = ErrorModeFunction::SUMOFSQUARES;
       if (fErrorStrategy == "CROSSENTROPY") fModeErrorFunction = ErrorModeFunction::CROSSENTROPY;
       if (fErrorStrategy == "MUTUALEXCLUSIVE") fModeErrorFunction = ErrorModeFunction::CROSSENTROPY_MUTUALEXCLUSIVE;

       for (auto& block : strategyKeyValues)
       {
           size_t convergenceSteps = 100;
           
           int convergenceSteps = fetchValue (stringKeyValues, "ConvergenceSteps", 100);
           int batchSize = fetchValue (stringKeyValues, "BatchSize", 30);
           int testRepetitions = fetchValue (stringKeyValues, "TestRepetitions", 7);
           double factorWeightDecay = fetchValue (stringKeyValues, "BatchSize", 0.0);
           bool isL1 = fetchValue (stringKeyValues, "isL1", false);
           double dropFraction = fetchValue (stringKeyValues, "DropFraction", 0.0);
           int dropRepetitions = fetchValue (stringKeyValues, "DropRepetitions", 7);
           
           

           std::shared_ptr<ClassificationSettings> ptrSettings = std::make_unique <ClassificationSettings> (
               convergenceSteps, batchSize, 
               testRepetitions, factorWeightDecay,
               isL1, dropFraction, dropRepetitions,
               fScaleToNumEvents);

           fSettings.push_back (ptrSettings);
       }
   }
   // else if (fAnalysisType == Types::kMulticlass)
   // {
   //     ptrSettings = std::make_unique <MulticlassSettings> ((*itSetting).convergenceSteps, (*itSetting).batchSize, 
   //                                                          (*itSetting).testRepetitions, (*itSetting).factorWeightDecay,
   //                                                          (*itSetting).isL1, (*itSetting).dropFraction, (*itSetting).dropRepetitions,
   //                                                          fScaleToNumEvents);
   // }
   // else if (fAnalysisType == Types::kRegression)
   // {
   //     ptrSettings = std::make_unique <RegressionSettings> ((*itSetting).convergenceSteps, (*itSetting).batchSize, 
   //                                                          (*itSetting).testRepetitions, (*itSetting).factorWeightDecay,
   //                                                          (*itSetting).isL1, (*itSetting).dropFraction, (*itSetting).dropRepetitions,
   //                                                          fScaleToNumEvents);
   // }

   settings.setWeightSums (fSumOfSigWeights_test, fSumOfBkgWeights_test);


}

//______________________________________________________________________________
void TMVA::MethodNN::Train()
{
    
    fMonitoring= NULL;
    if (fMonitoring)
    {
        fMonitoring = new Monitoring ();
        fMonitoring->Start ();
    }

    // INITIALIZATION
    // create pattern
    std::vector<Pattern> trainPattern;
    std::vector<Pattern> testPattern;

    const std::vector<TMVA::Event*>& eventCollectionTraining = GetEventCollection (Types::kTraining);
    const std::vector<TMVA::Event*>& eventCollectionTesting  = GetEventCollection (Types::kTesting);

    for (size_t iEvt = 0, iEvtEnd = eventCollectionTraining.size (); iEvt < iEvtEnd; ++iEvt)
    {
        const TMVA::Event* event = eventCollectionTraining.at (iEvt);
        std::vector<Float_t>& values  = event.GetValues  ();
        std::vector<Float_t>& targets = event.GetTargets ();
        trainPattern.push_back (Pattern (values.begin  (), values.end (), targets.begin (), targets.end (), event.GetWeight ());
    }

    for (size_t iEvt = 0, iEvtEnd = eventCollectionTesting.size (); iEvt < iEvtEnd; ++iEvt)
    {
        const TMVA::Event* event = eventCollectionTesting.at (iEvt);
        std::vector<Float_t>& values  = event.GetValues  ();
        std::vector<Float_t>& targets = event.GetTargets ();
        trainPattern.push_back (Pattern (values.begin  (), values.end (), targets.begin (), targets.end (), event.GetWeight ());
    }


    if (trainPattern.empty () || testPattern.empty ())
        return;

    // create net and weights
    fNet.clear ();
    fWeights.clear ();

    // if "resume" from saved weights
    if (fResume)
    {
        std::tie (fNet, fWeights) = Read (fFileName);
    }
    else // initialize weights and net
    {
        size_t inputSize = trainPattern.front ().input ().size ();
        size_t outputSize = trainPattern.front ().output ().size ();

        // configure neural net
        for (auto itLayout = begin (fLayout), itLayoutEnd = end (fLayout); itLayout != itLayoutEnd; ++itLayout)
        {
            //net.addLayer (NN::Layer (50, NN::EnumFunction::TANH)); 
            fNet.addLayer (NN::Layer ((*itLayout).numNodes, (*itLayout).activationFunction)); 
        }
        fNet.addLayer (NN::Layer (outputSize, (*itLayout).activationFunction, NN::ModeOutputValues::SIGMOID)); 
        fNet.setErrorFunction (fModeErrorFunction); 

        size_t numWeights = net.numWeights (inputSize);
        fWeights.resize (numWeights, 0.0);

        // initialize weights
        gaussDistribution (fWeights, 0.1, 1.0/sqrt(inputSize));
    }


    // loop through settings 
    // and create "settings" and minimizer 
    for (auto itSetting = begin (fSettings), itSettingEnd = end (fSettings); itSetting != itSettingEnd; ++itSetting)
    {
        std::unique_ptr<Settings> ptrSettings = *itSetting;
        ptrSetting->setMonitoring (monitoring);
        NN::Steepest minimizer (ptrSetting->learningRate, ptrSetting->momentum, ptrSetting->repetitions);

        double E = 0;
        if ((*itSetting).minimizer == MinimizerType::fSteepest)
        {
            NN::Steepest minimizer ((*itSetting).learningRate, (*itSetting).momentum, (*itSetting).repetitions);
            E = fNet.train (fWeights, trainPattern, testPattern, minimizer, &ptrSettings.get ());
        }
    }
    delete fMonitoring;
}





//_______________________________________________________________________
Double_t TMVA::MethodMLP::GetMvaValue( Double_t* errLower, Double_t* errUpper )
{
    if (fWeights.empty ())
        return 0.0;

    std::vector<Float_t> inputValues = GetEvent ();
    std::vector<double> input (inputValues.begin (), inputValues.end ());
    std::vector<double> output = fNet.compute (input, fWeights);
    if (output.empty ())
        return 0.0;

    return output.at (0);
}



//_______________________________________________________________________
void TMVA::MethodMLP::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   MethodANNBase::MakeClassSpecific(fout, className);
}

//_______________________________________________________________________
void TMVA::MethodMLP::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   TString col    = gConfig().WriteOptionsReference() ? TString() : gTools().Color("bold");
   TString colres = gConfig().WriteOptionsReference() ? TString() : gTools().Color("reset");

   Log() << Endl;
   Log() << col << "--- Short description:" << colres << Endl;
   Log() << Endl;
   Log() << "The MLP artificial neural network (ANN) is a traditional feed-" << Endl;
   Log() << "forward multilayer perceptron impementation. The MLP has a user-" << Endl;
   Log() << "defined hidden layer architecture, while the number of input (output)" << Endl;
   Log() << "nodes is determined by the input variables (output classes, i.e., " << Endl;
   Log() << "signal and one background). " << Endl;
   Log() << Endl;
   Log() << col << "--- Performance optimisation:" << colres << Endl;
   Log() << Endl;
   Log() << "Neural networks are stable and performing for a large variety of " << Endl;
   Log() << "linear and non-linear classification problems. However, in contrast" << Endl;
   Log() << "to (e.g.) boosted decision trees, the user is advised to reduce the " << Endl;
   Log() << "number of input variables that have only little discrimination power. " << Endl;
   Log() << "" << Endl;
   Log() << "In the tests we have carried out so far, the MLP and ROOT networks" << Endl;
   Log() << "(TMlpANN, interfaced via TMVA) performed equally well, with however" << Endl;
   Log() << "a clear speed advantage for the MLP. The Clermont-Ferrand neural " << Endl;
   Log() << "net (CFMlpANN) exhibited worse classification performance in these" << Endl;
   Log() << "tests, which is partly due to the slow convergence of its training" << Endl;
   Log() << "(at least 10k training cycles are required to achieve approximately" << Endl;
   Log() << "competitive results)." << Endl;
   Log() << Endl;
   Log() << col << "Overtraining: " << colres
         << "only the TMlpANN performs an explicit separation of the" << Endl;
   Log() << "full training sample into independent training and validation samples." << Endl;
   Log() << "We have found that in most high-energy physics applications the " << Endl;
   Log() << "avaliable degrees of freedom (training events) are sufficient to " << Endl;
   Log() << "constrain the weights of the relatively simple architectures required" << Endl;
   Log() << "to achieve good performance. Hence no overtraining should occur, and " << Endl;
   Log() << "the use of validation samples would only reduce the available training" << Endl;
   Log() << "information. However, if the perrormance on the training sample is " << Endl;
   Log() << "found to be significantly better than the one found with the inde-" << Endl;
   Log() << "pendent test sample, caution is needed. The results for these samples " << Endl;
   Log() << "are printed to standard output at the end of each training job." << Endl;
   Log() << Endl;
   Log() << col << "--- Performance tuning via configuration options:" << colres << Endl;
   Log() << Endl;
   Log() << "The hidden layer architecture for all ANNs is defined by the option" << Endl;
   Log() << "\"HiddenLayers=N+1,N,...\", where here the first hidden layer has N+1" << Endl;
   Log() << "neurons and the second N neurons (and so on), and where N is the number  " << Endl;
   Log() << "of input variables. Excessive numbers of hidden layers should be avoided," << Endl;
   Log() << "in favour of more neurons in the first hidden layer." << Endl;
   Log() << "" << Endl;
   Log() << "The number of cycles should be above 500. As said, if the number of" << Endl;
   Log() << "adjustable weights is small compared to the training sample size," << Endl;
   Log() << "using a large number of training samples should not lead to overtraining." << Endl;
}

