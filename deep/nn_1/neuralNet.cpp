
#include "neuralNet.h"

#include "../gnuplotWrapper/gnuplot_i.hpp" //Gnuplot class handles POSIX-Pipe-communikation with Gnuplot


namespace NN
{



double gaussDouble (double mean, double sigma)
{
    static std::default_random_engine generator;
    std::normal_distribution<double> distribution (mean, sigma);
    return distribution (generator);
}


double uniformDouble (double minValue, double maxValue)
{
    static std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(minValue, maxValue);
    return distribution(generator);
}


    
int randomInt (int maxValue)
{
    static std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,maxValue-1);
    return distribution(generator);
}


double studenttDouble (double distributionParameter)
{
    static std::default_random_engine generator;
    std::student_t_distribution<double> distribution (distributionParameter);
    return distribution (generator);
}



Steepest::Steepest (double learningRate, 
                    double momentum, 
                    size_t repetitions, 
                    Monitoring* pMonitoring, 
                    std::vector<size_t> layerSizes) 
    : MinimizerMonitoring (pMonitoring, layerSizes)
    , m_repetitions (repetitions)
    , m_alpha (learningRate)
    , m_beta (momentum)
{
}



    SteepestThreaded::SteepestThreaded (double learningRate, 
                                        double momentum, 
                                        size_t repetitions, 
                                        Monitoring* pMonitoring, 
                                        std::vector<size_t> layerSizes) 
        : MinimizerMonitoring (pMonitoring, layerSizes)
        , m_repetitions (repetitions)
        , m_alpha (learningRate)
        , m_beta (momentum)
    {
    }



    

typename Batch::const_iterator begin (const Batch& batch)
{
    return batch.begin ();
}

typename Batch::const_iterator end (const Batch& batch)
{
    return batch.end ();
}




    LayerData::LayerData (const_iterator_type itInputBegin, const_iterator_type itInputEnd, ModeOutputValues eModeOutput)
	: m_isInputLayer (true)
	, m_hasWeights (false)
	, m_hasGradients (false)
	, m_eModeOutput (eModeOutput) 
    {
	m_itInputBegin = itInputBegin;
	m_itInputEnd   = itInputEnd;
	m_size = std::distance (itInputBegin, itInputEnd);
	m_deltas.assign (m_size, 0);
    }




    LayerData::LayerData (size_t _size, 
	       const_iterator_type itWeightBegin, 
	       iterator_type itGradientBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       const_function_iterator_type itInverseFunctionBegin,
	       ModeOutputValues eModeOutput)
	: m_size (_size)
	, m_itConstWeightBegin   (itWeightBegin)
	, m_itGradientBegin (itGradientBegin)
	, m_itFunctionBegin (itFunctionBegin)
	, m_itInverseFunctionBegin (itInverseFunctionBegin)
	, m_isInputLayer (false)
	, m_hasWeights (true)
	, m_hasGradients (true)
	, m_eModeOutput (eModeOutput) 
    {
	m_values.assign (_size, 0);
	m_deltas.assign (_size, 0);
	m_valueGradients.assign (_size, 0);
    }




    LayerData::LayerData (size_t _size, const_iterator_type itWeightBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       ModeOutputValues eModeOutput)
	: m_size (_size)
	, m_itConstWeightBegin   (itWeightBegin)
	, m_itFunctionBegin (itFunctionBegin)
	, m_isInputLayer (false)
	, m_hasWeights (true)
	, m_hasGradients (false)
	, m_eModeOutput (eModeOutput) 
    {
	m_values.assign (_size, 0);
    }



    typename LayerData::container_type LayerData::computeProbabilities ()
    {
	container_type probabilitiesContainer;
	switch (m_eModeOutput)
	{
	case ModeOutputValues::SIGMOID:
        {
	    std::transform (begin (m_values), end (m_values), std::back_inserter (probabilitiesContainer), Sigmoid);
	    break;
        }
	case ModeOutputValues::SOFTMAX:
        {
            double sum = 0;
            probabilitiesContainer = m_values;
            std::for_each (begin (probabilitiesContainer), end (probabilitiesContainer), [&sum](double& p){ p = std::exp (p); sum += p; });
            if (sum != 0)
                std::for_each (begin (probabilitiesContainer), end (probabilitiesContainer), [sum ](double& p){ p /= sum; });
	    break;
        }
	case ModeOutputValues::DIRECT:
	default:
	    probabilitiesContainer.assign (begin (m_values), end (m_values));
	}
	return probabilitiesContainer;
    }






    Layer::Layer (size_t _numNodes, EnumFunction _activationFunction, ModeOutputValues eModeOutputValues) 
	: m_numNodes (_numNodes) 
	, m_eModeOutputValues (eModeOutputValues)
    {
	for (size_t iNode = 0; iNode < _numNodes; ++iNode)
	{
	    auto actFnc = Linear;
	    auto invActFnc = InvLinear;
	    m_activationFunction = EnumFunction::LINEAR;
	    switch (_activationFunction)
	    {
	    case EnumFunction::ZERO:
		actFnc = ZeroFnc;
		invActFnc = ZeroFnc;
		m_activationFunction = EnumFunction::ZERO;
		break;
	    case EnumFunction::LINEAR:
		actFnc = Linear;
		invActFnc = InvLinear;
		m_activationFunction = EnumFunction::LINEAR;
		break;
	    case EnumFunction::TANH:
		actFnc = Tanh;
		invActFnc = InvTanh;
		m_activationFunction = EnumFunction::TANH;
		break;
	    case EnumFunction::RELU:
		actFnc = ReLU;
		invActFnc = InvReLU;
		m_activationFunction = EnumFunction::RELU;
		break;
	    case EnumFunction::SYMMRELU:
		actFnc = SymmReLU;
		invActFnc = InvSymmReLU;
		m_activationFunction = EnumFunction::SYMMRELU;
		break;
	    case EnumFunction::TANHSHIFT:
		actFnc = TanhShift;
		invActFnc = InvTanhShift;
		m_activationFunction = EnumFunction::TANHSHIFT;
		break;
	    case EnumFunction::SOFTSIGN:
		actFnc = SoftSign;
		invActFnc = InvSoftSign;
		m_activationFunction = EnumFunction::SOFTSIGN;
		break;
	    case EnumFunction::SIGMOID:
		actFnc = Sigmoid;
		invActFnc = InvSigmoid;
		m_activationFunction = EnumFunction::SIGMOID;
		break;
	    case EnumFunction::GAUSS:
		actFnc = Gauss;
		invActFnc = InvGauss;
		m_activationFunction = EnumFunction::GAUSS;
		break;
	    case EnumFunction::GAUSSCOMPLEMENT:
		actFnc = GaussComplement;
		invActFnc = InvGaussComplement;
		m_activationFunction = EnumFunction::GAUSSCOMPLEMENT;
		break;
	    case EnumFunction::DOUBLEINVERTEDGAUSS:
		actFnc = DoubleInvertedGauss;
		invActFnc = InvDoubleInvertedGauss;
		m_activationFunction = EnumFunction::DOUBLEINVERTEDGAUSS;
		break;
	    }
	    m_vecActivationFunctions.push_back (actFnc);
	    m_vecInverseActivationFunctions.push_back (invActFnc);
	}
    }



    std::string Layer::write () const
    {
	std::stringstream signature;
	signature << "---LAYER---" << std::endl;
	signature << "LAYER=FULL" << std::endl;
	signature << "NODES=" << numNodes () << std::endl;
	signature << "ACTFNC=" << (char)m_activationFunction << std::endl;
	signature << "OUTMODE=" << (char)m_eModeOutputValues << std::endl;
	signature << "---LAYER-END---";
	return signature.str ();
    }


static Layer readLayer (std::istream& ss)
{
    std::string key, value, line;

    size_t numNodes (0);
    EnumFunction actFnc (EnumFunction::LINEAR);
    ModeOutputValues modeOut (ModeOutputValues::DIRECT);
    std::string layerType;
    std::string endLayerString ("---LAYER-END---");
    while(std::getline(ss, line))
    {
	if (line.compare(0, endLayerString.length (), endLayerString) == 0)
	    break;

	// Create an istringstream instance to parse the key and the value
	std::istringstream ss_line (line);
	std::getline(ss_line, key, '=');
 
	if (key == "LAYER")
	{
	    ss_line >> layerType;
	}
	else if (key == "NODES") 
	{
	    ss_line >> numNodes;
	}
	else if (key == "ACTFNC") 
	{
	    char actFncVal;
	    ss_line >> actFncVal;
	    actFnc = EnumFunction (actFncVal);
	}
	else if (key == "OUTMODE") 
	{
	    char modeOutputValues;
	    ss_line >> modeOutputValues;
	    modeOut = ModeOutputValues (modeOutputValues);
	}
    }
    if (layerType == "FULL")
    {
	return Layer (numNodes, actFnc, modeOut);
    }
    return Layer (0, EnumFunction::LINEAR, ModeOutputValues::DIRECT);
}


    
    


    
Settings::Settings (size_t _convergenceSteps, size_t _batchSize, size_t _testRepetitions, 
                    double _factorWeightDecay, EnumRegularization eRegularization,
                    bool _useMultithreading, Monitoring* pMonitoring)
    : m_convergenceSteps (_convergenceSteps)
    , m_batchSize (_batchSize)
    , m_testRepetitions (_testRepetitions)
    , m_factorWeightDecay (_factorWeightDecay)
    , count_E (0)
    , count_dE (0)
    , count_mb_E (0)
    , count_mb_dE (0)
    , m_regularization (eRegularization)
    , m_useMultithreading (_useMultithreading)
    , m_pMonitoring (pMonitoring)
    {
    }
    
Monitoring::~Monitoring () 
{
    for (PlotMap::iterator it = plots.begin (), itEnd = plots.end (); it != itEnd; ++it)
    {
        delete it->second;
    }
}




void Monitoring::clearData (std::string dataName)
{
    DataMap::mapped_type& data = getData (dataName);
    data.clear ();
}


void Settings::clearData (std::string dataName)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->clearData (dataName);
}


void Monitoring::addPoint (std::string dataName, double x, double y)
{
    DataMap::mapped_type& data = getData (dataName);
    data.useErr = false;

    data.x.push_back (x);
    data.y.push_back (y);
}

void Monitoring::addPoint (std::string dataName, double x, double y, double err)
{
    DataMap::mapped_type& data = getData (dataName);
    data.useErr = true;

    data.x.push_back (x);
    data.y.push_back (y);
    data.err.push_back (err);
}


void Settings::addPoint (std::string dataName, double x, double y)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->addPoint (dataName, x, y);
}



Gnuplot* Monitoring::plot (std::string plotName, std::string subName, std::string dataName, std::string style, std::string smoothing)
{
    Gnuplot* pPlot = getPlot (plotName);
    DataMap::mapped_type& data = getData (dataName);

    std::vector<double>& vecX = data.x;
    std::vector<double>& vecY = data.y;
    std::vector<double>& vecErr = data.err;

    if (vecX.empty ())
        return pPlot;

    if (data.useErr)
        pPlot->set_style(style).set_smooth(smoothing).plot_xy_err(vecX,vecY,vecErr,subName);
    else
        pPlot->set_style(style).set_smooth(smoothing).plot_xy(vecX,vecY,subName);
    pPlot->unset_smooth ();

    return pPlot;
}



Gnuplot* Settings::plot (std::string plotName, std::string subName, std::string dataName, std::string style, std::string smoothing)
{
    if (!m_pMonitoring)
        return NULL;
    return m_pMonitoring->plot (plotName, subName, dataName, style, smoothing);
}


void Monitoring::resetPlot (std::string plotName)
{
    Gnuplot* pPlot = getPlot (plotName);
    pPlot->reset_plot ();
}

void Settings::resetPlot (std::string plotName)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->resetPlot (plotName);
}


inline Gnuplot* Monitoring::getPlot (std::string plotName)
{
    PlotMap::iterator itPlot = plots.find (plotName);
    if (itPlot == plots.end ())
    {
        std::cout << "create new gnuplot" << std::endl;
        std::pair<PlotMap::iterator, bool> result = plots.insert (std::make_pair (plotName, new Gnuplot));
        itPlot = result.first;
    }

    Gnuplot* pPlot = itPlot->second;
    return pPlot;
}


inline Monitoring::DataMap::mapped_type& Monitoring::getData (std::string dataName)
{
    DataMap::iterator itData = data.find (dataName);
    if (itData == data.end ())
    {
        std::pair<DataMap::iterator, bool> result = data.insert (std::make_pair (dataName, DataMap::mapped_type (false)));
        itData = result.first;
    }

    return itData->second;
}














void ClassificationSettings::testSample (double error, double output, double target, double weight)
    {
        m_output.push_back (output);
        m_targets.push_back (target);
        m_weights.push_back (weight);
    }


void ClassificationSettings::startTestCycle () 
    {
        m_output.clear ();
        m_targets.clear ();
        m_weights.clear ();
        clearData ("datRoc");
        clearData ("datOutputSig");
        clearData ("datOutputBkg");
        clearData ("datAms");
        clearData ("datSignificance");
        resetPlot ("roc");
        resetPlot ("output");
        resetPlot ("amsSig");
    }

    void ClassificationSettings::endTestCycle () 
    {
        if (m_output.empty ())
            return;
        double minVal = *std::min_element (begin (m_output), end (m_output));
        double maxVal = *std::max_element (begin (m_output), end (m_output));
        const size_t numBinsROC = 1000;
        const size_t numBinsData = 100;

        std::vector<double> truePositives (numBinsROC+1, 0);
        std::vector<double> falsePositives (numBinsROC+1, 0);
        std::vector<double> trueNegatives (numBinsROC+1, 0);
        std::vector<double> falseNegatives (numBinsROC+1, 0);

        std::vector<double> x (numBinsData, 0);
        std::vector<double> datSig (numBinsData+1, 0);
        std::vector<double> datBkg (numBinsData+1, 0);

        double binSizeROC = (maxVal - minVal)/(double)numBinsROC;
        double binSizeData = (maxVal - minVal)/(double)numBinsData;

        double sumWeightsSig = 0.0;
        double sumWeightsBkg = 0.0;

        for (size_t b = 0; b < numBinsData; ++b)
        {
            double binData = minVal + b*binSizeData;
            x.at (b) = binData;
        }

        if (fabs(binSizeROC) < 0.0001)
            return;

        for (size_t i = 0, iEnd = m_output.size (); i < iEnd; ++i)
        {
            double val = m_output.at (i);
            double truth = m_targets.at (i);
            double weight = m_weights.at (i);

            bool isSignal = (truth > 0.5 ? true : false);

            if (m_sumOfSigWeights != 0 && m_sumOfBkgWeights != 0)
            {
                if (isSignal)
                    weight *= m_sumOfSigWeights;
                else
                    weight *= m_sumOfBkgWeights;
            }

            size_t binROC = (val-minVal)/binSizeROC;
            size_t binData = (val-minVal)/binSizeData;

            if (isSignal)
            {
                for (size_t n = 0; n <= binROC; ++n)
                {
                    truePositives.at (n) += weight;
                }
                for (size_t n = binROC+1; n < numBinsROC; ++n)
                {
                    falseNegatives.at (n) += weight;
                }

                datSig.at (binData) += weight;
                sumWeightsSig += weight;
            }
            else
            {
                for (size_t n = 0; n <= binROC; ++n)
                {
                    falsePositives.at (n) += weight;
                }
                for (size_t n = binROC+1; n < numBinsROC; ++n)
                {
                    trueNegatives.at (n) += weight;
                }

                datBkg.at (binData) += weight;
                sumWeightsBkg += weight;
            }
        }

        std::vector<double> sigEff;
        std::vector<double> backRej;

        double bestSignificance = 0;
        double bestCutAMS = 0;
        double bestCutSignificance = 0;

        double bestAMS = 0;

	double numEventsScaleFactor = 1.0;
	if (m_scaleToNumEvents > 0)
	{
	    size_t numEvents = m_output.size ();
	    numEventsScaleFactor = double (m_scaleToNumEvents)/double (numEvents);
	}

        for (size_t i = 0; i < numBinsROC; ++i)
        {
            double tp = truePositives.at (i) * numEventsScaleFactor;
            double fp = falsePositives.at (i) * numEventsScaleFactor;
            double tn = trueNegatives.at (i) * numEventsScaleFactor;
            double fn = falseNegatives.at (i) * numEventsScaleFactor;

            double seff = (tp+fn == 0.0 ? 1.0 : (tp / (tp+fn)));
	    double brej = (tn+fp == 0.0 ? 0.0 : (tn / (tn+fp)));

            sigEff.push_back (seff);
            backRej.push_back (brej);
            
            addPoint ("datRoc", seff, brej); // x, y


	    double currentCut = (i * binSizeROC)+minVal;

            double sig = tp;
            double bkg = fp;
            double significance = sig / sqrt (sig + bkg);
            if (significance > bestSignificance)
            {
                bestSignificance = significance;
                bestCutSignificance = currentCut;
            }

            double br = 10.0;
            double s = tp;
            double b = fp;
            double radicand = 2 *( (s+b+br) * log (1.0 + s/(b+br)) -s);
            if (radicand < 0)
                std::cout << "radicand is negative." << std::endl;
            else
                radicand = sqrt (radicand);

            addPoint ("datAms", currentCut, radicand); // x, y
	    addPoint ("datSignificance", currentCut, significance);

            if (radicand > bestAMS) 
	    {
                bestAMS = radicand;
		bestCutAMS = currentCut;
	    }
        }

        m_significances.push_back (bestSignificance);
        static size_t testCycle = 0;

        for (size_t i = 0; i < numBinsData; ++i)
        {
            addPoint ("datOutputSig", x.at (i), datSig.at (i)/sumWeightsSig);
            addPoint ("datOutputBkg", x.at (i), datBkg.at (i)/sumWeightsBkg);
        }


        
        m_ams.push_back (bestAMS);

        ++testCycle;

//inline Gnuplot* Settings::plot (std::string plotName, std::string subName, std::string dataName, std::string style, std::string smoothing)
        plot ("roc", "curvePoints", "datRoc", "lines", "cspline");
        plot ("output", "outPtsSig", "datOutputSig", "lines", "cspline");
        plot ("output", "outPtsBkg", "datOutputBkg", "lines", "cspline");
        plot ("amsSig", "curveAms", "datAms", "lines", "cspline");
        plot ("amsSig", "curveSignificance", "datSignificance", "lines", "cspline");

        std::cout << "bestCutAMS = " << bestCutAMS << "  ams = " << bestAMS
		  << "      bestCutSignificance = " << bestCutSignificance << "  significance = " << bestSignificance << std::endl;
	m_cutValue = bestCutAMS;
    }



    void ClassificationSettings::setWeightSums (double sumOfSigWeights, double sumOfBkgWeights) { m_sumOfSigWeights = sumOfSigWeights; m_sumOfBkgWeights = sumOfBkgWeights; }
    void ClassificationSettings::setResultComputation (std::string _fileNameNetConfig, std::string _fileNameResult, std::vector<Pattern>* _resultPatternContainer)
    {
	m_pResultPatternContainer = _resultPatternContainer;
	m_fileNameResult = _fileNameResult;
	m_fileNameNetConfig = _fileNameNetConfig;
    }






    

    size_t Net::numWeights (size_t numInputNodes, size_t trainingStartLayer) const 
    {
	size_t num (0);
	size_t index (0);
	size_t prevNodes (numInputNodes);
	for (auto& layer : m_layers)
	{
	    if (index >= trainingStartLayer)
		num += layer.numWeights (prevNodes);
	    prevNodes = layer.numNodes ();
	    ++index;
	}
	return num;
    }





    std::ostream& Net::write (std::ostream& ostr) const
    {
	ostr << "===NET===" << std::endl;
	ostr << "ERRORFUNCTION=" << (char)m_eErrorFunction << std::endl;
	for (const Layer& layer: m_layers)
	{
	    ostr << layer.write () << std::endl;
	}
	return ostr;
    }




void MinimizerMonitoring::resetPlot (std::string plotName)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->resetPlot (plotName);
}


void MinimizerMonitoring::clearData (std::string dataName)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->clearData (dataName);
}



void MinimizerMonitoring::addPoint (std::string dataName, double x, double y)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->addPoint (dataName, x, y);
}

void MinimizerMonitoring::addPoint (std::string dataName, double x, double y, double err)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->addPoint (dataName, x, y, err);
}



std::ostream& operator<< (std::ostream& ostr, Net const& net)
{
    ostr << "NET" << std::endl;
    for (Layer const& layer : net.m_layers)
    {
	ostr << layer.write ();
	ostr << std::endl;
    }
    ostr << std::endl;
    return ostr;
}


std::istream& read (std::istream& istr, Net& net)
{
    // net
    std::string line, key;
    if (!getline (istr, line)) // "NET"
        return istr;

    if (line != "===NET===")
	return istr;

    while (istr.good ())
    {
	if (!getline (istr, line))
	    return istr;

	std::istringstream ss_line (line);
	std::getline(ss_line, key, '=');
 
	if (key == "ERRORFUNCTION")
	{
	    char errorFnc;
	    ss_line >> errorFnc;
	    net.setErrorFunction (ModeErrorFunction (errorFnc));
	}
	else if (line == "---LAYER---")
	    net.addLayer (readLayer (istr));
	else
	    return istr;
    }
    return istr;
}







void write (std::string fileName, const Net& net, const std::vector<double>& weights) 
{
    std::ofstream file (fileName, std::ios::trunc);	
    net.write (file);
    file << "===WEIGHTS===" << std::endl;
    for (double w : weights)
    {
	file << w << " ";
    }
    file << std::endl;
}


std::tuple<Net, std::vector<double>> read (std::string fileName) 
{
    std::vector<double> weights;
    Net net;

    std::ifstream infile (fileName);

    // net
    if (infile.is_open () && infile.good ())
    {
	read (infile, net);
    }


    // weights
    std::string line;
    if (!getline (infile, line))
        return std::make_tuple (net, weights);

    std::stringstream ssline (line);

    while (ssline)
    {
        double value;
        std::string token;
        if (!getline (ssline, token, ' ')) 
            break;

	std::stringstream tr;
	tr << token;
	tr >> value;

        weights.push_back (value);
    }
    return std::make_tuple (net, weights);
}


    
    

}; // namespace NN

    



