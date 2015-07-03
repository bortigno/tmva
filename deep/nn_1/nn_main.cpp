
#include "neuralNet.h"
#include "../mnist/readnist.hpp" 





// hilfsfunktion um auf einen tastenDruck zu warten
void wait_for_key ()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)  // every keypress registered, also arrow keys
    std::cout << std::endl << "Press any key to continue..." << std::endl;

    FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
    _getch();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    std::cout << std::endl << "Press ENTER to continue..." << std::endl;

    std::cin.clear();
    std::cin.ignore(std::cin.rdbuf()->in_avail());
    std::cin.get();
#endif
    return;
}




void writeKaggleHiggs (std::string fileName, const NN::Net& net, const std::vector<double>& weights, 
		       std::vector<Pattern>& patternContainer, double cutValue);




class KaggleClassificationSettings : public NN::ClassificationSettings
{
public:
    KaggleClassificationSettings (size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
                                  double _factorWeightDecay = 1e-5, NN::EnumRegularization eRegularization = NN::EnumRegularization::NONE, 
                                  size_t _scaleToNumEvents = 0, NN::Monitoring* pMonitoring = NULL)
        : NN::ClassificationSettings (_convergenceSteps, _batchSize, _testRepetitions, _factorWeightDecay, eRegularization, _scaleToNumEvents, pMonitoring)
    {
    }

    virtual void computeResult (const NN::Net& net, std::vector<double>& weights) 
    {
	write (m_fileNameNetConfig, net, weights);
	if (!m_fileNameResult.empty () && m_pResultPatternContainer && !m_pResultPatternContainer->empty ())
	    writeKaggleHiggs (m_fileNameResult, net, weights, *m_pResultPatternContainer, m_cutValue);
    }
};







void writeKaggleHiggs (std::string fileName, const NN::Net& net, const std::vector<double>& weights, 
		       std::vector<Pattern>& patternContainer, double cutValue)
{
    //                     mva,    label, id,    rank
    std::vector<std::tuple<double, char, size_t, size_t> > data;
 
    for (const Pattern& pattern : patternContainer)
    {
	double value;
	char label;
	size_t id;
	size_t rank;
	
	value = net.compute (pattern.input (), weights).at (0);
	id = pattern.getID ();
	label = (value > cutValue ? 's' : 'b');
	rank = 0;

	data.push_back (std::make_tuple (value, label, id, rank));
    }
    std::sort (begin (data), end (data));
    size_t idx = 1;
    for_each (begin (data), end (data), [&idx](std::tuple<double, char, size_t, size_t>& row){
	    size_t& rank = std::get<3>(row);
	    rank = idx;
	    ++idx;
	} );

    std::ofstream file (fileName, std::ios::trunc);	
    file << "EventId,RankOrder,Class" << std::endl;
    for_each (begin (data), end (data), [&file](std::tuple<double, char, size_t, size_t>& row){
	    char& label = std::get<1>(row);
	    size_t& id = std::get<2>(row);
	    size_t& rank = std::get<3>(row);
	    file << id << "," << rank << "," << label << std::endl;
	} );
    file << std::endl;
 }










void checkGradients ()
{
    NN::Net net;

    size_t inputSize = 1;
    size_t outputSize = 1;


    net.setInputSize (inputSize);
    net.setOutputSize (outputSize);
    net.addLayer (NN::Layer (30, NN::EnumFunction::SOFTSIGN)); 
//    net.addLayer (NN::Layer (30, NN::EnumFunction::SOFTSIGN)); 
//    net.addLayer (Layer (outputSize, EnumFunction::LINEAR)); 
    net.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR, NN::ModeOutputValues::SIGMOID)); 
    net.setErrorFunction (NN::ModeErrorFunction::CROSSENTROPY);
//    net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);
    //weights.at (0) = 1000213.2;

    // test dropWeightFactor

    NN::uniform (weights, 0.7);

    std::cout << "number of weights = " << numWeights << std::endl;

    std::vector<double> weightsCopy (weights);
    std::vector<double> dropFractions = {0.3, 0.4, 0.7, 0.2};
    net.dropOutWeightFactor (weightsCopy, dropFractions);
    int idx = 0;
    for (auto itWC = weightsCopy.begin (), itWCEnd = weightsCopy.end (), itW = weights.begin (); itWC != itWCEnd; ++itW, ++itWC)
    {
	std::cout << "index " << idx << "  :  w = " << *itW << " , wcopy = " << *itWC << std::endl;
	assert (*itWC == 0 || *itWC != *itW);
	++idx;
    }
    net.dropOutWeightFactor (weightsCopy, dropFractions, true);
    idx = 0;
    for (auto itWC = weightsCopy.begin (), itWCEnd = weightsCopy.end (), itW = weights.begin (); itWC != itWCEnd; ++itW, ++itWC)
    {
	std::cout << "index " << idx << "  :  w = " << *itW << " , wcopy = " << *itWC << std::endl;
	assert (std::fabs (*itWC - *itW) < 1e-5);
	++idx;
    }

    weights.assign (numWeights, 0.0);

    std::vector<Pattern> pattern;
    for (size_t iPat = 0, iPatEnd = 10; iPat < iPatEnd; ++iPat)
    {
        std::vector<double> input;
        std::vector<double> output;
        for (size_t i = 0; i < inputSize; ++i)
        {
            input.push_back (NN::uniformDouble (-1.5, 1.5));
        }
        for (size_t i = 0; i < outputSize; ++i)
        {
            output.push_back (NN::uniformDouble (-1.5, 1.5));
        }
        pattern.push_back (Pattern (input,output));
    }


    NN::Settings settings (/*_convergenceSteps*/ 15, /*_batchSize*/ 1, /*_testRepetitions*/ 7, /*_factorWeightDecay*/ 0,
                           /*regularization*/ NN::EnumRegularization::NONE, /* use multithreading */ true);

    size_t improvements = 0;
    size_t worsenings = 0;
    size_t smallDifferences = 0;
    size_t largeDifferences = 0;
    for (size_t iTest = 0; iTest < 1000; ++iTest)
    {
        NN::uniform (weights, 0.7);
        std::vector<double> gradients (numWeights, 0);
        NN::Batch batch (begin (pattern), end (pattern));
        NN::DropContainer dropContainer;
        std::tuple<NN::Settings&, NN::Batch&, NN::DropContainer&> settingsAndBatch (settings, batch, dropContainer);
        double E = net (settingsAndBatch, weights, gradients);
        std::vector<double> changedWeights;
        changedWeights.assign (weights.begin (), weights.end ());

        int changeWeightPosition = NN::randomInt (numWeights);
        double dEdw = gradients.at (changeWeightPosition);
        while (dEdw == 0.0)
        {
            changeWeightPosition = NN::randomInt (numWeights);
            dEdw = gradients.at (changeWeightPosition);
        }

        const double gamma = 0.01;
        double delta = gamma*dEdw;
        changedWeights.at (changeWeightPosition) += delta;
        if (dEdw == 0.0)
        {
            std::cout << "dEdw == 0.0 ";
            continue;
        }
        
        assert (dEdw != 0.0);
        double Echanged = net (settingsAndBatch, changedWeights);

//	double difference = fabs((E-Echanged) - delta*dEdw);
        double difference = fabs ((E+delta - Echanged)/E);
	bool direction = (E-Echanged)>0 ? true : false;
//	bool directionGrad = delta>0 ? true : false;
        bool isOk = difference < 0.3 && difference != 0;

	if (direction)
	    ++improvements;
	else
	    ++worsenings;

	if (isOk)
	    ++smallDifferences;
	else
	    ++largeDifferences;

        if (true || !isOk)
        {
	    if (!direction)
		std::cout << "=================" << std::endl;
            std::cout << "E = " << E << " Echanged = " << Echanged << " delta = " << delta << "   pos=" << changeWeightPosition << "   dEdw=" << dEdw << "  difference= " << difference << "  dirE= " << direction << std::endl;
        }
        if (isOk)
        {
        }
        else
        {
//            for_each (begin (weights), end (weights), [](double w){ std::cout << w << ", "; });
//            std::cout << std::endl;
//            assert (isOk);
        }
    }
    std::cout << "improvements = " << improvements << std::endl;
    std::cout << "worsenings = " << worsenings << std::endl;
    std::cout << "smallDifferences = " << smallDifferences << std::endl;
    std::cout << "largeDifferences = " << largeDifferences << std::endl;

    std::cout << "check gradients done" << std::endl;
}






void testXOR ()
{
//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)


    NN::Net net;

    size_t inputSize = 2;
    size_t outputSize = 1;

    net.setInputSize (inputSize);
    net.setOutputSize (outputSize);
    net.addLayer (NN::Layer (4, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR)); 

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);

    std::vector<Pattern> patterns;
    patterns.push_back (Pattern ({0, 0}, {0}));
    patterns.push_back (Pattern ({1, 1}, {0}));
    patterns.push_back (Pattern ({1, 0}, {1}));
    patterns.push_back (Pattern ({0, 1}, {1}));

    NN::uniform (weights, 0.7);
    
//    StochasticCG minimizer;
    NN::Steepest minimizer (/*learningRate*/ 1e-5);
    NN::Settings settings (/*_convergenceSteps*/ 50, /*_batchSize*/ 4, /*_testRepetitions*/ 7);
    /*double E = */net.train (weights, patterns, patterns, minimizer, settings);

}





void testClassification ()
{
//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)


    std::vector<Pattern> trainPattern;
    std::vector<Pattern> testPattern;

    std::default_random_engine generator;
    std::normal_distribution<double> distX0 (1.0, 2.0);
    std::normal_distribution<double> distY0 (1.0, 2.0);
    std::normal_distribution<double> distX1 (-1.0, 3.0);
    std::normal_distribution<double> distY1 (-1.0, 3.0);
    for (size_t i = 0, iEnd = 5000; i < iEnd; ++i)
    {
        trainPattern.push_back (Pattern ({distX0 (generator), distY0 (generator)}, {0.9}));
        trainPattern.push_back (Pattern ({distX1 (generator), distY1 (generator)}, {0.1}));
        testPattern.push_back (Pattern ({distX0 (generator), distY0 (generator)}, {0.9}));
        testPattern.push_back (Pattern ({distX1 (generator), distY1 (generator)}, {0.1}));
    }

    assert (!trainPattern.empty ());
    assert (!testPattern.empty ());


    NN::Net net;

    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.setInputSize (inputSize);
    net.setOutputSize (outputSize);
    net.addLayer (NN::Layer (10, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (10, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (10, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR, NN::ModeOutputValues::SIGMOID)); 
//    net.addLayer (Layer (outputSize, EnumFunction::LINEAR, ModeOutputValues::DIRECT)); 
    net.setErrorFunction (NN::ModeErrorFunction::CROSSENTROPY);
//    net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);

    NN::uniform (weights, 0.2);
    
    NN::Steepest minimizer (1e-6, 0.0, 3);
//    MaxGradWeight minimizer;
    NN::ClassificationSettings settings (/*_convergenceSteps*/ 150, /*_batchSize*/ 30, /*_testRepetitions*/ 7, 
                                         /*factorWeightDecay*/ 1.0e-5, /*regularization*/ NN::EnumRegularization::NONE, 
				     /*scaleToNumEvents*/0);
    /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);


}




void testWriteRead ()
{
//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)

    std::vector<Pattern> trainPattern;
    std::vector<Pattern> testPattern;

    std::default_random_engine generator;
    std::normal_distribution<double> distX0 (1.0, 2.0);
    std::normal_distribution<double> distY0 (1.0, 2.0);
    std::normal_distribution<double> distX1 (-1.0, 3.0);
    std::normal_distribution<double> distY1 (-1.0, 3.0);
    for (size_t i = 0, iEnd = 1000; i < iEnd; ++i)
    {
        trainPattern.push_back (Pattern ({distX0 (generator), distY0 (generator)}, {1.0}));
        trainPattern.push_back (Pattern ({distX1 (generator), distY1 (generator)}, {0.0}));
        testPattern.push_back (Pattern ({distX0 (generator), distY0 (generator)}, {1.0}));
        testPattern.push_back (Pattern ({distX1 (generator), distY1 (generator)}, {0.0}));
    }

    assert (!trainPattern.empty ());
    assert (!testPattern.empty ());


    NN::Net net;

    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.setInputSize (inputSize);
    net.setOutputSize (outputSize);
    net.addLayer (NN::Layer (3, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR)); 

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);

    NN::uniform (weights, 0.2);
    
    NN::Steepest minimizer;
    NN::ClassificationSettings settings (/*_convergenceSteps*/ 2, /*_batchSize*/ 30, /*_testRepetitions*/ 7, /*factorWeightDecay*/ 1.0e-5);
    //double E = net.train (weights, trainPattern, testPattern, minimizer, settings);


    std::cout << "BEFORE" << std::endl;
    std::cout << net << std::endl;
    std::cout << "WEIGHTS BEFORE" << std::endl;
    for_each (begin (weights), end (weights), [](double w){ std::cout << w << " "; });
    std::cout << std::endl << std::endl;

    // writing
    write ("testfile.nn", net, weights);

    // reading
    NN::Net readNet;
    std::vector<double> readWeights;
    std::tie (readNet, readWeights) = NN::read ("testfile.nn");


    std::cout << "READ" << std::endl;
    std::cout << readNet << std::endl;
    std::cout << "WEIGHTS READ" << std::endl;
    for_each (begin (readWeights), end (readWeights), [](double w){ std::cout << w << " "; });
    std::cout << std::endl << std::endl;
}







void Higgs ()
{
//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)


    // std::string filenameTrain ("/home/peter/code/kaggle_Higgs/training.csv");
    // std::string filenameTest ("/home/peter/code/kaggle_Higgs/training.csv");
    // std::string filenameSubmission ("/home/peter/code/kaggle_Higgs/test.csv");

    std::string filenameTrain ("/home/peters/test/kaggle_Higgs/training.csv");
    std::string filenameTest ("/home/peters/test/kaggle_Higgs/training.csv");
    std::string filenameSubmission ("/home/peters/test/kaggle_Higgs/test.csv");

    std::vector<std::string> fieldNamesTrain; 
    std::vector<std::string> fieldNamesTest; 
    size_t skipTrain = 0;
    size_t numberTrain = 100000;
    size_t skipTest  =  100000;
    size_t numberTest  =  100000;
    size_t numberSubmission  = 5;
    
    double sumOfSigWeights_train (0);
    double sumOfBkgWeights_train (0);
    double sumOfSigWeights_test (0);
    double sumOfBkgWeights_test (0);
    double sumOfSigWeights_sub (0);
    double sumOfBkgWeights_sub (0);

    std::vector<Pattern> trainPattern = readCSV (filenameTrain, fieldNamesTrain, "EventId", "Label", "Weight", 
                                                 sumOfSigWeights_train, sumOfBkgWeights_train, numberTrain, skipTrain);
    std::vector<Pattern> testPattern = readCSV (filenameTest, fieldNamesTest, "EventId", "Label", "Weight", 
                                                sumOfSigWeights_test, sumOfBkgWeights_test, numberTest, skipTest);
    std::vector<Pattern> submissionPattern = readCSV (filenameSubmission, fieldNamesTest, "EventId", "Label", "Weight", 
                                                      sumOfSigWeights_sub, sumOfBkgWeights_sub, numberSubmission);

    std::cout << "read " << trainPattern.size () << " training pattern from CSV file" << std::endl;
    std::cout << "read " << testPattern.size () <<  " test pattern from CSV file" << std::endl;
    std::cout << "read " << submissionPattern.size () <<  " submission pattern from CSV file" << std::endl;

    assert (!trainPattern.empty ());
    assert (!testPattern.empty ());


    // reading
    NN::Net net;
    std::vector<double> weights;
    std::vector<double> dropConfig;
    std::vector<double> dropConfig2;

#if false // read from saved file
    std::tie (net, weights) = read ("higgs.net");

    // net.layers ().back ().modeOutputValues (ModeOutputValues::DIRECT); 
    // net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);
    
#else
    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.setInputSize (inputSize);
    net.setOutputSize (outputSize);
    net.addLayer (NN::Layer (50, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (30, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (10, NN::EnumFunction::TANH)); 
    net.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR, NN::ModeOutputValues::SIGMOID)); 
    net.setErrorFunction (NN::ModeErrorFunction::CROSSENTROPY);

    net.initializeWeights (NN::WeightInitializationStrategy::XAVIERUNIFORM, 
			   trainPattern.begin (),
			   trainPattern.end (), 
			   std::back_inserter (weights));

    dropConfig = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    dropConfig2 = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    double dropRepetitions = 1;
    

#endif

    NN::Monitoring monitoring;
    std::vector<size_t> layerSizesForMonitoring;

    for (auto& layer : net.layers ())
    {
        layerSizesForMonitoring.push_back (layer.numWeights (inputSize));
        inputSize = layer.numNodes ();
    }

    


    int scaleToNumEvents = 550000;
    
//    typedef NN::SteepestThreaded LocalMinimizer;
    typedef NN::Steepest LocalMinimizer;
    {
        LocalMinimizer minimizer (1e-1, 0.0, 1, &monitoring, layerSizesForMonitoring);
	KaggleClassificationSettings settings (/*_convergenceSteps*/ 100, /*_batchSize*/ 10, /*_testRepetitions*/ 7, 
                                                   /*factorWeightDecay*/ 1e-3, /*regularization*/NN::EnumRegularization::NONE,
                                                   /*scaleToNumEvents*/ scaleToNumEvents, &monitoring);
        settings.setDropOut (std::begin (dropConfig), std::end (dropConfig), dropRepetitions);
        settings.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);

        settings.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);
    }
    {
        LocalMinimizer minimizer (1e-2, 0.5, 1, &monitoring, layerSizesForMonitoring);
        KaggleClassificationSettings settings (/*_convergenceSteps*/ 100, /*_batchSize*/ 20, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.001, /*regularization*/NN::EnumRegularization::NONE,
                                              /*scaleToNumEvents*/ scaleToNumEvents, &monitoring);
        settings.setDropOut (std::begin (dropConfig2), std::end (dropConfig2), dropRepetitions);
        settings.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);
    }
    {
        LocalMinimizer minimizer (1e-2, 0.3, 1, &monitoring, layerSizesForMonitoring);
        KaggleClassificationSettings settings (/*_convergenceSteps*/ 50, /*_batchSize*/ 30, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.001, /*regularization*/NN::EnumRegularization::L2,
                                              /*scaleToNumEvents*/ scaleToNumEvents, &monitoring);
//        settings2.setDropOut (std::begin (dropConfig2), std::end (dropConfig2), dropRepetitions);
        settings.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);
    }
    {
        LocalMinimizer minimizer (1e-3, 0.1, 1, &monitoring, layerSizesForMonitoring);
        KaggleClassificationSettings settings (/*_convergenceSteps*/ 50, /*_batchSize*/ 40, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.0001, /*regularization*/NN::EnumRegularization::L1,
                                              /*scaleToNumEvents*/ scaleToNumEvents, &monitoring);
        settings.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);
    }


    
    wait_for_key();
}




void createChessData (int numPattern)
{
   const int nvar = 2;
   double xvar[nvar];

   // output files
   std::ofstream file_data;

   file_data.open ("chess_data.csv");

   double sigma=0.3;
   double meanX;
   double meanY;
   int xtype=1, ytype=1;
   int iCurrent=0;
   int m_nDim = 2; // actually the boundary, there is a "bump" for every interger value
                     // between in the Inteval [-m_nDim,m_nDim]

   file_data << "id,x,y,weight,label" << std::endl;

   while (iCurrent < numPattern)
   {
       xtype=1;
       for (int i=-m_nDim; i <=  m_nDim; i++)
       {
           ytype  =  1;
           for (int j=-m_nDim; j <=  m_nDim; j++)
           {
               meanX=double(i);
               meanY=double(j);
               xvar[0]=NN::gaussDouble(meanY,sigma);
               xvar[1]=NN::gaussDouble(meanX,sigma);
               int type   = xtype*ytype;
               file_data << iCurrent << "," << xvar[0] << "," << xvar[1] << "," << 1.0 << "," << (type==1 ? "s" : "b") << std::endl;
               iCurrent++;
               ytype *= -1;
           }
           xtype *= -1;
       }
   }

   file_data.close ();
}






void Chess ()
{



//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)


    std::string filenameTrain ("chess_data.csv");
    std::string filenameTest ("chess_data.csv");

    std::vector<std::string> fieldNamesTrain; 
    std::vector<std::string> fieldNamesTest; 
    size_t skipTrain = 0;
    size_t numberTrain = 10000;
    size_t skipTest  =  10000;
    size_t numberTest  =  10000;
    
    double sumOfSigWeights_train (0);
    double sumOfBkgWeights_train (0);
    double sumOfSigWeights_test (0);
    double sumOfBkgWeights_test (0);

    std::vector<Pattern> trainPattern = readCSV (filenameTrain, fieldNamesTrain, "id", "label", "weight", 
                                                 sumOfSigWeights_train, sumOfBkgWeights_train, numberTrain, skipTrain);
    std::vector<Pattern> testPattern = readCSV (filenameTest, fieldNamesTest, "id", "label", "weight", 
                                                sumOfSigWeights_test, sumOfBkgWeights_test, numberTest, skipTest);

    std::cout << "read " << trainPattern.size () << " training pattern from CSV file" << std::endl;
    std::cout << "read " << testPattern.size () <<  " test pattern from CSV file" << std::endl;

    assert (!trainPattern.empty ());
    assert (!testPattern.empty ());


    // reading
    NN::Net net;
    std::vector<double> weights;
    std::vector<double> dropConfig;
    std::vector<double> dropConfig2;
    
#if false // read from saved file
    std::tie (net, weights) = read ("chess.net");

    // net.layers ().back ().modeOutputValues (ModeOutputValues::DIRECT); 
    // net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);
    
#else
    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.setInputSize (inputSize);
    net.setOutputSize (outputSize);
    NN::EnumFunction myActFnc = NN::EnumFunction::SOFTSIGN;
    net.addLayer (NN::Layer (4, myActFnc)); 
    net.addLayer (NN::Layer (3, myActFnc)); 
    net.addLayer (NN::Layer (2, myActFnc)); 
    net.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR, NN::ModeOutputValues::SIGMOID)); 
    net.setErrorFunction (NN::ModeErrorFunction::CROSSENTROPY);

    net.initializeWeights (NN::WeightInitializationStrategy::XAVIERUNIFORM, 
			   trainPattern.begin (),
			   trainPattern.end (), 
			   std::back_inserter (weights));

    dropConfig = {0.0, 0.5, 0.5, 0.5, 0.5};
    dropConfig2 = {0.0, 0.1, 0.1, 0.1, 0.1};
    double dropRepetitions = 1;
    
#endif
    
    NN::Monitoring monitoring;
    std::vector<size_t> layerSizesForMonitoring;

    for (auto& layer : net.layers ())
    {
        layerSizesForMonitoring.push_back (layer.numWeights (inputSize));
        inputSize = layer.numNodes ();
    }


    std::cout << "number of weights : " << net.numWeights () << std::endl;



    bool mulithreading = true;
    typedef NN::Steepest LocalMinimizer;
    {
        LocalMinimizer minimizer (1e-1, 0.2, 1, &monitoring, layerSizesForMonitoring);
	NN::ClassificationSettings settings (/*_convergenceSteps*/ 500, /*_batchSize*/ 50, /*_testRepetitions*/ 7, 
                                             /*factorWeightDecay*/ 1e-3, /*regularization*/NN::EnumRegularization::NONE,
                                             /*scaleToNumEvents*/ 10000,
                                             /* use multithreading */ mulithreading, 
                                             &monitoring);
        settings.setDropOut (std::begin (dropConfig), std::end (dropConfig), dropRepetitions);

        settings.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);
    }
    {
        LocalMinimizer minimizer2 (1e-2, 0.5, 1, &monitoring, layerSizesForMonitoring);
        NN::ClassificationSettings settings2 (/*_convergenceSteps*/ 300, /*_batchSize*/ 40, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.001, /*regularization*/NN::EnumRegularization::L2,
                                              /*scaleToNumEvents*/ 10000,
                                             /* use multithreading */ mulithreading, 
                                              &monitoring);
        settings2.setDropOut (std::begin (dropConfig2), std::end (dropConfig2), dropRepetitions);
        settings2.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer2, settings2);
    }
    {
        LocalMinimizer minimizer2 (1e-2, 0.3, 1, &monitoring, layerSizesForMonitoring);
        NN::ClassificationSettings settings2 (/*_convergenceSteps*/ 150, /*_batchSize*/ 40, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.001, /*regularization*/NN::EnumRegularization::L2,
                                              /*scaleToNumEvents*/ 10000,
                                              /* use multithreading */ mulithreading, 
                                              &monitoring);
//        settings2.setDropOut (std::begin (dropConfig2), std::end (dropConfig2), dropRepetitions);
        settings2.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer2, settings2);
    }
    {
        LocalMinimizer minimizer2 (1e-3, 0.1, 1, &monitoring, layerSizesForMonitoring);
        NN::ClassificationSettings settings2 (/*_convergenceSteps*/ 200, /*_batchSize*/ 100, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.0001, /*regularization*/NN::EnumRegularization::L1,
                                              /*scaleToNumEvents*/ 10000,
                                             /* use multithreading */ mulithreading, 
                                              &monitoring);
        settings2.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer2, settings2);
    }
    
    wait_for_key();
    
}



void mnist ()
{



//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)


    // std::string fileNameTrainImg ("/home/peter/code/MNIST/train-images-idx3-ubyte");
    // std::string fileNameTestImg ("/home/peter/code/MNIST/t10k-images-idx3-ubyte");
    // std::string fileNameTrainLabels ("/home/peter/code/MNIST/train-labels-idx1-ubyte");
    // std::string fileNameTestLabels ("/home/peter/code/MNIST/t10k-labels-idx1-ubyte");

    std::string fileNameTrainImg    ("/home/peters/test/MNIST_handWrittenDigits/train-images-idx3-ubyte");
    std::string fileNameTrainLabels ("/home/peters/test/MNIST_handWrittenDigits/train-labels-idx1-ubyte");
    std::string fileNameTestImg     ("/home/peters/test/MNIST_handWrittenDigits/t10k-images-idx3-ubyte");
    std::string fileNameTestLabels  ("/home/peters/test/MNIST_handWrittenDigits/t10k-labels-idx1-ubyte");

    std::vector<std::string> fieldNamesTrain; 
    std::vector<std::string> fieldNamesTest; 
    // size_t skipTrain = 0;
    // size_t numberTrain = 10000;
    // size_t skipTest  =  10000;
    // size_t numberTest  =  10000;
    
    // double sumOfSigWeights_train (0);
    // double sumOfBkgWeights_train (0);
    double sumOfSigWeights_test (0);
    double sumOfBkgWeights_test (0);


    std::vector<Pattern> trainPattern = read_Mnist (fileNameTrainImg, fileNameTrainLabels, -1);
    std::vector<Pattern> testPattern = read_Mnist (fileNameTestImg, fileNameTestLabels, -1);


    std::cout << "read " << trainPattern.size () << " training pattern from MNIST files" << std::endl;
    std::cout << "read " << testPattern.size () <<  " test pattern from MNIST files" << std::endl;

    assert (!trainPattern.empty ());
    assert (!testPattern.empty ());


    // reading
    NN::Net net;
    std::vector<double> weights;
    std::vector<double> dropConfig;
    std::vector<double> dropConfig2;
    
#if false // read from saved file
    std::tie (net, weights) = read ("mnist.net");

    // net.layers ().back ().modeOutputValues (ModeOutputValues::DIRECT); 
    // net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);
    
#else
    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.setInputSize (inputSize);
    net.setOutputSize (outputSize);
    NN::EnumFunction myActFnc = NN::EnumFunction::SOFTSIGN;
    // net.addLayer (NN::Layer (100, NN::EnumFunction::SOFTSIGN)); 
    // net.addLayer (NN::Layer (30, NN::EnumFunction::SOFTSIGN)); 
    // net.addLayer (NN::Layer (20, NN::EnumFunction::SOFTSIGN)); 
    net.addLayer (NN::Layer (1000, myActFnc)); 
    net.addLayer (NN::Layer (100, myActFnc)); 
    net.addLayer (NN::Layer (10, myActFnc)); 
    net.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR, NN::ModeOutputValues::SIGMOID)); 
    net.setErrorFunction (NN::ModeErrorFunction::CROSSENTROPY);

//    size_t numWeightsFirstLayer = net.layers ().front ().numWeights (inputSize);

//    size_t numWeights = net.numWeights (inputSize);

//    gaussDistribution (weights, 0.1, 1.0/sqrt(inputSize));
    net.initializeWeights (NN::WeightInitializationStrategy::XAVIERUNIFORM, 
			   trainPattern.begin (),
			   trainPattern.end (), 
			   std::back_inserter (weights));

    dropConfig = {0.0, 0.5, 0.5, 0.5, 0.5, 0.5};
    dropConfig2 = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    double dropRepetitions = 1;
    
#endif
    
    NN::Monitoring monitoring;
    std::vector<size_t> layerSizesForMonitoring;

    for (auto& layer : net.layers ())
    {
        layerSizesForMonitoring.push_back (layer.numWeights (inputSize));
        inputSize = layer.numNodes ();
    }

    std::cout << "number of weights : " << net.numWeights () << std::endl;

    bool mulithreading = true;
    typedef NN::Steepest LocalMinimizer;
    {
        LocalMinimizer minimizer (1e-1, 0.0, 1, &monitoring, layerSizesForMonitoring);
	NN::ClassificationSettings settings (/*_convergenceSteps*/ 200, /*_batchSize*/ 50, /*_testRepetitions*/ 7, 
                                             /*factorWeightDecay*/ 1e-3, /*regularization*/NN::EnumRegularization::NONE,
                                             /*scaleToNumEvents*/ 10000,
                                             /* use multithreading */ mulithreading, 
                                             &monitoring);
        settings.setDropOut (std::begin (dropConfig), std::end (dropConfig), dropRepetitions);

        settings.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);
    }
    {
        LocalMinimizer minimizer2 (1e-2, 0.3, 1, &monitoring, layerSizesForMonitoring);
        NN::ClassificationSettings settings2 (/*_convergenceSteps*/ 300, /*_batchSize*/ 40, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.001, /*regularization*/NN::EnumRegularization::L2,
                                              /*scaleToNumEvents*/ 10000,
                                             /* use multithreading */ mulithreading, 
                                              &monitoring);
        settings2.setDropOut (std::begin (dropConfig2), std::end (dropConfig2), dropRepetitions);
        settings2.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer2, settings2);
    }
    {
        LocalMinimizer minimizer2 (1e-2, 0.3, 1, &monitoring, layerSizesForMonitoring);
        NN::ClassificationSettings settings2 (/*_convergenceSteps*/ 70, /*_batchSize*/ 40, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.001, /*regularization*/NN::EnumRegularization::L2,
                                              /*scaleToNumEvents*/ 10000,
                                              /* use multithreading */ mulithreading, 
                                              &monitoring);
//        settings2.setDropOut (std::begin (dropConfig2), std::end (dropConfig2), dropRepetitions);
        settings2.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer2, settings2);
    }
    {
        LocalMinimizer minimizer2 (1e-3, 0.1, 1, &monitoring, layerSizesForMonitoring);
        NN::ClassificationSettings settings2 (/*_convergenceSteps*/ 200, /*_batchSize*/ 100, /*_testRepetitions*/ 7, 
                                              /*factorWeightDecay*/ 0.0001, /*regularization*/NN::EnumRegularization::L1,
                                              /*scaleToNumEvents*/ 10000,
                                             /* use multithreading */ mulithreading, 
                                              &monitoring);
        settings2.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
//    settings2.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
        /*double E = */net.train (weights, trainPattern, testPattern, minimizer2, settings2);
    }
    
    wait_for_key();
    
}




int main ()
{ 
//    createChessData (20000);
//    return 1;
//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)


//   checkGradients ();
//    testXOR ();
//    Higgs ();
    Chess ();
//    mnist ();
//    testClassification ();
//    testWriteRead ();

//    wait_for_key();
    return 0;
} 


