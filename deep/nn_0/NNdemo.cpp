

#include <fstream>
#include <iostream>
#include <cstddef> 
#include <iomanip> 
#include "../pattern/pattern.hpp"
#include "neuralNet.hpp"

#include <fenv.h>

#include "../mnist/readnist.hpp"

const double pi = 3.1415927;


// hilfsfunktion um auf einen tastendruck zu warten
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




// Formel fuer das regressions-beispiel
double formula_0 (double& a, double& b)
{
    double x = randomNumber (0.0,1.0);
    double frac = randomNumber (0.0,1.0);
    a = frac * x;
    b = (1.0-frac)*x;
    double result = sin ((a+b)*2.0*pi);
    result *= result;
    result = result*2 -1;
    return result;
}


// alternative formel fuer das regressions-beispiel
double formula_1 (double& a, double& b)
{
    double x = randomNumber (-1.0,1.0);
    double frac = randomNumber (0.0,1.0);
    a = frac * x;
    b = (1.0-frac)*x;
    double result = (a+b);
    result = sin(result*2*pi);
    result = (result/fabs(result))/(sqrt(fabs(40*x))+1)*0.99*cos (x*2*3.1415*2.1); // + noise
    result = std::min (0.9, std::max (-0.9, result));
//    result = result*2 -1;
    return result;
}

// alternative formel fuer das regressions-beispiel
double formula_2 (double& a, double& b)
{
    double x = randomNumber (-1.0,1.0);
    double frac = randomNumber (0.0,1.0);
    a = frac * x;
    b = (1.0-frac)*x;
    double result = sin ((a+b)*2.0*pi)/sqrt (fabs(a+b));
    result *= result;
    result = result*2 -1;
    return result;
}


// hier erfolgt die auswahl welche der formeln fuer das regressions-beispiel verwendet wird
double formula (double& a, double& b)
{
    return formula_2 (a, b);
}


// vorbereitung einer shape fuer ein klassifikations-beispiel
void shape (double& x, double& y, int& category)
{
    x = randomNumber (-1.0,1.0);
    y = randomNumber (-1.0,1.0);

    category = 0;
    
    double center0[2] = {0.8,-0.3};
    double center1[2] = {-0.7,0.5};
    double radius0 = 0.3;
    double radius1 = 0.1;

    double diff_0[2] = {x-center0[0], y-center0[1]};
    double diff_1[2] = {x-center1[0], y-center1[1]};

    double dist[2] = {sqrt (std::pow(diff_0[0],2.0) + std::pow(diff_0[1],2.0)), sqrt (std::pow(diff_1[0],2.0) + std::pow(diff_1[1],2.0))};

    if (dist[0] < radius0)
        category = 1;

    if (dist[1] < radius1)
        category = 2;

}





// enthaelt additional zu den settings die plot-kommandos fuer die graphischen
// ausgaben. 
struct RegressionSettings : public Settings
{
    virtual void startTestCycle () 
    {
        resetPlot ("curve");
        clearData ("datCurve");
        clearData ("datCurveTgt");
    }

    virtual void endTestCycle () 
    {
        plot ("curve", "curvePointsTgt", "datCurveTgt", "points", "");
        plot ("curve", "curvePoints", "datCurve", "points", "");
    }

    virtual void drawSample (const std::vector<NNTYPE>& input, const std::vector<NNTYPE>& output, const std::vector<NNTYPE>& target, NNTYPE patternWeight) 
    {
        NNTYPE inVal = input.at(0) + input.at (1);
        NNTYPE outVal = output.at (0);
        NNTYPE outTgt = target.at (0);

        addPoint ("datCurveTgt", inVal, outTgt);
        addPoint ("datCurve", inVal, outVal);
    }
};




// enthaelt additional zu den settings die plot-kommandos fuer die graphischen
// ausgaben. 
struct ClassificationSettings : public Settings
{
    virtual void startTestCycle () 
    {
        m_out.clear ();
        m_tgt.clear ();
        m_weights.clear ();
        resetPlot ("roc");
        clearData ("datRoc");
    }

    virtual void endTestCycle () 
    {
        double minVal = *std::min_element (begin (m_out), end (m_out));
        double maxVal = *std::max_element (begin (m_out), end (m_out));
        const size_t numBins = 100;

        std::vector<double> truePositives (numBins, 0);
        std::vector<double> falsePositives (numBins, 0);
        std::vector<double> trueNegatives (numBins, 0);
        std::vector<double> falseNegatives (numBins, 0);

        double binSize = (maxVal - minVal)/(double)numBins;

        if (fabs(binSize) < 0.0001)
            return;

        for (size_t i = 0, iEnd = m_out.size (); i < iEnd; ++i)
        {
            double val = m_out.at (i);
            double truth = m_tgt.at (i);
            double weight = m_weights.at (i);

            size_t bin = (val-minVal)/binSize;

            if (truth > 0.5)
            {
                for (size_t n = 0; n < bin; ++n)
                {
                    falseNegatives.at (bin) += weight;
                }
                for (size_t n = bin; n < numBins; ++n)
                {
                    truePositives.at (bin) += weight;
                }
            }
            else
            {
                for (size_t n = 0; n < bin; ++n)
                {
                    trueNegatives.at (bin) += weight;
                }
                for (size_t n = bin; n < numBins; ++n)
                {
                    falsePositives.at (bin) += weight;
                }
            }
        }

        std::vector<double> sigEff;
        std::vector<double> backRej;

        for (size_t i = 0; i < numBins; ++i)
        {
            double tp = truePositives.at (i);
            double fp = falsePositives.at (i);
            double tn = trueNegatives.at (i);
            double fn = falseNegatives.at (i);

            double seff = tp / (tp+fp);
            double br = 1.0 - (fn / (tn+fn));

            sigEff.push_back (seff);
            backRej.push_back (br);
            
            addPoint ("datRoc", seff, br);
        }


        plot ("curveRoc", "curvePoints", "datRoc", "points", "");
    }

    virtual void drawSample (const std::vector<NNTYPE>& input, const std::vector<NNTYPE>& output, const std::vector<NNTYPE>& target, NNTYPE patternWeight) 
    {
        m_out.push_back (output.at (0));
        m_tgt.push_back (target.at (0));
        m_weights.push_back (patternWeight);
    }

    std::vector<double> m_out;
    std::vector<double> m_tgt;
    std::vector<double> m_weights;
};




// funktioniert noch nicht wie gewuenscht. Soll die kommandos fuer die 
// graphische ausgabe im klassifikationsbeispiel enthalten
struct ShapeSettings : public Settings
{
    virtual void startTestCycle () 
    {
        resetPlot ("curve");
        clearData ("datCurve");
        clearData ("datCurveTgt");
    }

    virtual void endTestCycle () 
    {
        plot ("curve", "curvePointsTgt", "datCurveTgt", "points", "");
        plot ("curve", "curvePoints", "datCurve", "points", "");
    }

    virtual void drawSample (const std::vector<NNTYPE>& input, const std::vector<NNTYPE>& output, const std::vector<NNTYPE>& target, NNTYPE patternWeight) 
    {
        NNTYPE inVal = input.at(0) + input.at (1);
        NNTYPE outVal = output.at (0);
        NNTYPE outTgt = target.at (0);

        addPoint ("datCurveTgt", inVal, outTgt);
        addPoint ("datCurve", inVal, outVal);
    }
};





// dieses beispiel funktioniert. Macht regression y= f(a+b)
int testRegression ()
{
    std::cout << "TEST REGRESSION" << std::endl;
    RegressionSettings settings;
    settings.weightDecay = 1e-5; // weight-decay. Reduziert die weights mit jedem schritt. Sorgt fuer stabilere weights
    settings.maxInitWeight = 0.9; // maximalwert den ein weight bei der initialisierung einnemhen kann. (wird gauss-gewuerfelt)
    settings.repetitions = 10; // repetitions fuer den minimisier-algorithmus
    settings.maxRepetitions = 50; // nach dieser anzahl der repetitions wird der minimisier-algorithmus abgebrochen und mit 
// dem naechsten batch weitergemacht.

    settings.testRepetitions = 10; // nach testRepetitions wird ein test mit dem test-sample durchgefuehrt und die 
// graphischen ausgaben fuer die test samples koennen sich veraendern. 

    NeuralNet neuralNet (&settings); // initialisierung des neuronalen netzes


    // konfiguration des neuronalen netzes
    size_t layers[] = {2,30,30,20,1};
    size_t numLayers = (sizeof (layers)/sizeof(size_t));
    neuralNet.mlp (&layers[0], &layers[numLayers], eVarying, eLinear); 

    std::vector<Pattern> trainPattern;
    std::vector<Pattern> testPattern;

 
    int numEvents = 5000; // anzahl der trainings samples und test samples
    for (int i = 0, iEnd = numEvents; i < iEnd; ++i) 
    {
        double a = 0;
        double b = 0;
        double result = formula (a, b);
        double input[] = {a,b};
        double output[] = {result};
        Pattern pattern (&input[0], &input[2], &output[0], &output[1]); // bauen eines training patterns
	trainPattern.push_back (pattern);// hinzufuegen zu den pattern (training)
    }


    for (int i = 0, iEnd = numEvents; i < iEnd; ++i)
    {
        double a = 0;
        double b = 0;
	double result = formula (a, b);
	double input[] = {a,b};
	double output[] = {result};
	Pattern pattern (&input[0], &input[2], &output[0], &output[1]); // bauen eines test patterns
	testPattern.push_back (pattern); // hinzufuegen zu dem pattern vektor (test)
    }


    neuralNet.training_SCG (trainPattern.begin (), trainPattern.end (), testPattern.begin (), testPattern.end ()
			    , 10 // batch size
    			    , 400); // convergence steps



    // erstellen eines files mit dem output der test-samples
    std::ofstream testFile; 
    testFile.open ("test.csv");
    testFile << "a:b:s:o:t" << std::endl;
    for (auto it = testPattern.begin (), itEnd= testPattern.end (); it != itEnd; ++it)
    {
	Pattern& pattern = *it;
	NNTYPE sumInput = 0.0;
	std::vector<NNTYPE> inputValues;
	bool isFirst = true;
	for (auto itP = pattern.beginInput (), itPEnd = pattern.endInput (); itP != itPEnd; ++itP)
	{
	    NNTYPE value = *itP;
	    inputValues.push_back (value);
	    sumInput += value;
	    if (!isFirst)
	    {
		testFile << ",";
	    }
	    testFile<< value;
	    isFirst = false;
	}
	testFile << "," << sumInput;

	neuralNet.calculateSample (inputValues.begin (), inputValues.end ());
	NNTYPE result = (*neuralNet.beginNodes (2))->value;
	testFile << "," << result;

	for (auto itP = pattern.beginOutput (), itPEnd = pattern.endOutput (); itP != itPEnd; ++itP)
	{
	    NNTYPE value = *itP;
	    testFile << "," << value;
	}
	testFile << std::endl;
    }
    testFile.close();
    

    // erstellen eines files mit dem output der training-samples
    std::ofstream trainFile;
    trainFile.open ("train.csv");
    trainFile << "a:b:s:o:t" << std::endl;
    for (auto it = trainPattern.begin (), itEnd= trainPattern.end (); it != itEnd; ++it)
    {
	Pattern& pattern = *it;
	NNTYPE sumInput = 0.0;
	std::vector<NNTYPE> inputValues;
	bool isFirst = true;
	for (auto itP = pattern.beginInput (), itPEnd = pattern.endInput (); itP != itPEnd; ++itP)
	{
	    NNTYPE value = *itP;
	    inputValues.push_back (value);
	    sumInput += value;
	    if (!isFirst)
	    {
		trainFile << ",";
	    }
	    trainFile << value;
	    isFirst = false;
	}
	trainFile << "," << sumInput;

	neuralNet.calculateSample (inputValues.begin (), inputValues.end ());
	NNTYPE result = (*neuralNet.beginNodes (2))->value;
	trainFile << "," << result;


	for (auto itP = pattern.beginOutput (), itPEnd = pattern.endOutput (); itP != itPEnd; ++itP)
	{
	    NNTYPE value = *itP;
	    trainFile << "," << value;
	}
	trainFile << std::endl;
    }
    trainFile.close();

    wait_for_key();

    return 0;
}








int testCategories ()
{
    std::cout << "TEST CATEGORIES" << std::endl;
    ClassificationSettings settings;
    settings.weightDecay = 1e-5; // weight-decay. Reduziert die weights mit jedem schritt. Sorgt fuer stabilere weights
    settings.maxInitWeight = 0.9; // maximalwert den ein weight bei der initialisierung einnemhen kann. (wird gauss-gewuerfelt)
    settings.repetitions = 20; // repetitions fuer den minimisier-algorithmus
    settings.maxRepetitions = 50; // nach dieser anzahl der repetitions wird der minimisier-algorithmus abgebrochen und mit 
// dem naechsten batch weitergemacht.

    settings.testRepetitions = 10; // nach testRepetitions wird ein test mit dem test-sample durchgefuehrt und die 
// graphischen ausgaben fuer die test samples koennen sich veraendern. 

    NeuralNet neuralNet (&settings);



    // std::string filenameTrain ("/home/peter/code/kaggle_Higgs/training.csv");
    // std::string filenameTest ("/home/peter/code/kaggle_Higgs/test.csv");
    std::string filenameTrain ("/home/developer/test/kaggle_Higgs/training.csv");
    std::string filenameTest ("/home/developer/test/kaggle_Higgs/test.csv");

    std::vector<std::string> fieldNamesTrain; 
    std::vector<std::string> fieldNamesTest; 
    std::vector<Pattern> trainPattern = readCSV (filenameTrain, fieldNamesTrain, "Label", "Weight", 50000);
    std::vector<Pattern> testPattern = readCSV (filenameTrain, fieldNamesTest, "Label", "Weight", 50000, 50000);

    assert (!trainPattern.empty ());
    assert (!testPattern.empty ());

//    for_each (pattern.begin (), pattern.end (), [](const Pattern& p){ std::cout << p; } );



    size_t layers[] = {testPattern.front ().input ().size (),30,30, testPattern.front ().output ().size ()};
    size_t numLayers = (sizeof (layers)/sizeof(size_t));
    neuralNet.mlp (&layers[0], &layers[numLayers], eVarying, eLinear);



    std::cout << "start training" << std::endl;
    neuralNet.training_SCG (trainPattern.begin (), trainPattern.end (), testPattern.begin (), testPattern.end ()
			    , 50 // batch size
    			    , 30); // convergence steps



    // std::ofstream testFile;
    // testFile.open ("test.csv");
    // testFile << "a:b:s:o:t" << std::endl;
    // for (auto it = testPattern.begin (), itEnd= testPattern.end (); it != itEnd; ++it)
    // {
    //     Pattern& pattern = *it;
    //     NNTYPE sumInput = 0.0;
    //     std::vector<NNTYPE> inputValues;
    //     bool isFirst = true;
    //     for (auto itP = pattern.beginInput (), itPEnd = pattern.endInput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         inputValues.push_back (value);
    //         sumInput += value;
    //         if (!isFirst)
    //         {
    //     	testFile << ",";
    //         }
    //         testFile<< value;
    //         isFirst = false;
    //     }
    //     testFile << "," << sumInput;

    //     neuralNet.calculateSample (inputValues.begin (), inputValues.end ());
    //     NNTYPE result = (*neuralNet.beginNodes (2))->value;
    //     testFile << "," << result;

    //     for (auto itP = pattern.beginOutput (), itPEnd = pattern.endOutput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         testFile << "," << value;
    //     }
    //     testFile << std::endl;
    // }
    // testFile.close();
    

    // std::ofstream trainFile;
    // trainFile.open ("train.csv");
    // trainFile << "a:b:s:o:t" << std::endl;
    // for (auto it = trainPattern.begin (), itEnd= trainPattern.end (); it != itEnd; ++it)
    // {
    //     Pattern& pattern = *it;
    //     NNTYPE sumInput = 0.0;
    //     std::vector<NNTYPE> inputValues;
    //     bool isFirst = true;
    //     for (auto itP = pattern.beginInput (), itPEnd = pattern.endInput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         inputValues.push_back (value);
    //         sumInput += value;
    //         if (!isFirst)
    //         {
    //     	trainFile << ",";
    //         }
    //         trainFile << value;
    //         isFirst = false;
    //     }
    //     trainFile << "," << sumInput;

    //     neuralNet.calculateSample (inputValues.begin (), inputValues.end ());
    //     NNTYPE result = (*neuralNet.beginNodes (2))->value;
    //     trainFile << "," << result;


    //     for (auto itP = pattern.beginOutput (), itPEnd = pattern.endOutput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         trainFile << "," << value;
    //     }
    //     trainFile << std::endl;
    // }
    // trainFile.close();

    std::cout << "finished training" << std::endl;
    wait_for_key();

    return 0;
}








// funktionscheck ob die gradientenberechnung sinnvoll ist
int checkGradientCalculation ()
{
    std::cout << "CHECK GRADIENT CALCULATION" << std::endl;
    RegressionSettings settings;
//    settings.weightDecay = 1e-5;
    settings.weightDecay = 0.0;
    settings.maxInitWeight = 20.0;
    settings.repetitions = 10;
    settings.maxRepetitions = 30;
    NeuralNet neuralNet (&settings);

    size_t layers[] = {1,1};
    size_t numLayers = (sizeof (layers)/sizeof(size_t));
    neuralNet.mlp (&layers[0], &layers[numLayers], eLinear, eLinear);


    double a = 0;
    double b = 0;

    // double result = formula (a, b);
    // double input[] = {a+b};
    // double output[] = {result};
    // Pattern pattern (&input[0], &input[1], &output[0], &output[1]);

    int numEvents = 5;
    for (int i = 0, iEnd = numEvents; i < iEnd; ++i)
    {
        double result = formula (a, b);
        double input[] = {a+b};
        double output[] = {result};
        Pattern pattern (&input[0], &input[1], &output[0], &output[1]);


        NNTYPE E = 0, E_delta = 0, E_delta_gradient = 0;
        bool gradOK = neuralNet.checkGradient (E, E_delta, E_delta_gradient, pattern, 1.0, 0.5);
        NNTYPE relDiff = (E_delta - E_delta_gradient)/E;
        const int col = 10;
        std::cout << "E = " << std::setw (col) << E << "  E_d = " << std::setw (col)  << E_delta << "  E_dg = " << std::setw (col) << E_delta_gradient << "    diff = " << std::setw (col) << relDiff << "         " << (gradOK ? "OK" : "WRONG") << std::endl;

    }


    std::cout << "finished training" << std::endl;
    wait_for_key();

    return 0;
}







int testMNISTLabeling ()
{
    std::cout << "TEST MNIST LABELLING" << std::endl;
    RegressionSettings settings;
    settings.weightDecay = 1e-5;
    settings.maxInitWeight = 0.9;
    settings.repetitions = 10;
    settings.maxRepetitions = 50;
    settings.testRepetitions = 10;
    NeuralNet neuralNet (&settings);


    auto trainPattern = read_Mnist ("/home/developer/test/MNIST_handWrittenDigits/train-images.idx3-ubyte", 
                                                   "/home/developer/test/MNIST_handWrittenDigits/train-labels.idx1-ubyte", 1000);
    auto testPattern = read_Mnist ("/home/developer/test/MNIST_handWrittenDigits/t10k-images.idx3-ubyte", 
                                                  "/home/developer/test/MNIST_handWrittenDigits/t10k-labels.idx1-ubyte", 100);

    size_t inputSize = trainPattern.at (0).inputSize ();
    size_t outputSize = trainPattern.at (0).outputSize ();

    size_t layers[] = {inputSize,100,100,outputSize};
    size_t numLayers = (sizeof (layers)/sizeof(size_t));
    neuralNet.mlp (&layers[0], &layers[numLayers], eVarying, eLinear);



    neuralNet.training_SCG (trainPattern.begin (), trainPattern.end (), testPattern.begin (), testPattern.end ()
			    , 10 // batch size
    			    , 40); // convergence steps



    // std::ofstream testFile;
    // testFile.open ("test.csv");
    // testFile << "a:b:s:o:t" << std::endl;
    // for (auto it = testPattern.begin (), itEnd= testPattern.end (); it != itEnd; ++it)
    // {
    //     Pattern& pattern = *it;
    //     NNTYPE sumInput = 0.0;
    //     std::vector<NNTYPE> inputValues;
    //     bool isFirst = true;
    //     for (auto itP = pattern.beginInput (), itPEnd = pattern.endInput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         inputValues.push_back (value);
    //         sumInput += value;
    //         if (!isFirst)
    //         {
    //     	testFile << ",";
    //         }
    //         testFile<< value;
    //         isFirst = false;
    //     }
    //     testFile << "," << sumInput;

    //     neuralNet.calculateSample (inputValues.begin (), inputValues.end ());
    //     NNTYPE result = (*neuralNet.beginNodes (2))->value;
    //     testFile << "," << result;

    //     for (auto itP = pattern.beginOutput (), itPEnd = pattern.endOutput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         testFile << "," << value;
    //     }
    //     testFile << std::endl;
    // }
    // testFile.close();
    

    // std::ofstream trainFile;
    // trainFile.open ("train.csv");
    // trainFile << "a:b:s:o:t" << std::endl;
    // for (auto it = trainPattern.begin (), itEnd= trainPattern.end (); it != itEnd; ++it)
    // {
    //     Pattern& pattern = *it;
    //     NNTYPE sumInput = 0.0;
    //     std::vector<NNTYPE> inputValues;
    //     bool isFirst = true;
    //     for (auto itP = pattern.beginInput (), itPEnd = pattern.endInput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         inputValues.push_back (value);
    //         sumInput += value;
    //         if (!isFirst)
    //         {
    //     	trainFile << ",";
    //         }
    //         trainFile << value;
    //         isFirst = false;
    //     }
    //     trainFile << "," << sumInput;

    //     neuralNet.calculateSample (inputValues.begin (), inputValues.end ());
    //     NNTYPE result = (*neuralNet.beginNodes (2))->value;
    //     trainFile << "," << result;


    //     for (auto itP = pattern.beginOutput (), itPEnd = pattern.endOutput (); itP != itPEnd; ++itP)
    //     {
    //         NNTYPE value = *itP;
    //         trainFile << "," << value;
    //     }
    //     trainFile << std::endl;
    // }
    // trainFile.close();

    std::cout << "finished training" << std::endl;
    wait_for_key();

    return 0;
}





// obsolete test
void testPowForSmallValues ()
{
    double val = 2.3;
    while (true)
    {
        val /= 11.23234234;
        double result = std::pow (val,2.0);
        std::cout << "val " << val << " = " << result << std::endl;
    }
}




int main ()
{
//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)
//    testRegression ();
//    testMNISTLabeling ();
//    testPowForSmallValues ();
//    checkGradientCalculation ();
    testCategories ();
}
    


// int main ()
// {
//     std::vector<Pattern> vecA;
// //    vecA.reserve (10);

//     for (int i = 0; i < 5; ++i)
//     {
// 	std::cout << "=== iteration " << i << std::endl;
// 	auto v = { 1+i, 2+i, 3+i, 4+i};
// 	auto v2 = { 1-i, 2-i, 3-i, 4-i};
// 	vecA.emplace_back (Pattern (begin (v), end (v), begin (v2), end (v2), i));
// //	vecA.emplace_back (Pattern ());
//     }
    
// }

