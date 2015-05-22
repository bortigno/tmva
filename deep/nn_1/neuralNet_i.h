

namespace NN
{



inline double gaussDouble (double mean, double sigma)
{
    static std::default_random_engine generator;
    std::normal_distribution<double> distribution (mean, sigma);
    return distribution (generator);
}


template <typename ItSource, typename ItWeight, typename ItTarget>
void applyWeights (ItSource itSourceBegin, ItSource itSourceEnd, ItWeight itWeight, ItTarget itTargetBegin, ItTarget itTargetEnd)
{
    for (auto itSource = itSourceBegin; itSource != itSourceEnd; ++itSource)
    {
        for (auto itTarget = itTargetBegin; itTarget != itTargetEnd; ++itTarget)
        {
            (*itTarget) += (*itSource) * (*itWeight);
            ++itWeight;
        }
    }
}



template <typename ItSource, typename ItWeight, typename ItPrev>
void applyWeightsBackwards (ItSource itCurrBegin, ItSource itCurrEnd, ItWeight itWeight, ItPrev itPrevBegin, ItPrev itPrevEnd)
{
    for (auto itPrev = itPrevBegin; itPrev != itPrevEnd; ++itPrev)
    {
	for (auto itCurr = itCurrBegin; itCurr != itCurrEnd; ++itCurr)
	{
            (*itPrev) += (*itCurr) * (*itWeight);
            ++itWeight;
        }
    }
}





template <typename ItValue, typename ItFunction>
void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction)
{
    while (itValue != itValueEnd)
    {
        auto& value = (*itValue);
        value = (*itFunction) (value);

        ++itValue; ++itFunction;
    }
}


template <typename ItValue, typename ItFunction, typename ItInverseFunction, typename ItGradient>
void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction, ItInverseFunction itInverseFunction, ItGradient itGradient)
{
    while (itValue != itValueEnd)
    {
        auto& value = (*itValue);
        value = (*itFunction) (value);
        (*itGradient) = (*itInverseFunction) (value);
        
        ++itValue; ++itFunction; ++itInverseFunction; ++itGradient;
    }
}



template <typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient)
{
    while (itSource != itSourceEnd)
    {
        auto itTargetDelta = itTargetDeltaBegin;
        auto itTargetGradient = itTargetGradientBegin;
        while (itTargetDelta != itTargetDeltaEnd)
        {
            (*itGradient) += - (*itTargetDelta) * (*itSource) * (*itTargetGradient);
            ++itTargetDelta; ++itTargetGradient; ++itGradient;
        }
        ++itSource; 
    }
}



template <bool isL1, typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient, typename ItWeight>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient, 
	     ItWeight itWeight, double weightDecay)
{
    while (itSource != itSourceEnd)
    {
        auto itTargetDelta = itTargetDeltaBegin;
        auto itTargetGradient = itTargetGradientBegin;
        while (itTargetDelta != itTargetDeltaEnd)
        {
            //                                                                                       L1 regularization                   L2 regularization
	    (*itGradient) -= + (*itTargetDelta) * (*itSource) * (*itTargetGradient) + (isL1 ? std::copysign (weightDecay,(*itWeight)) : (*itWeight) * weightDecay);
            ++itTargetDelta; ++itTargetGradient; ++itGradient; ++itWeight;
        }
        ++itSource; 
    }
}




inline MinimizerMonitoring::MinimizerMonitoring (Monitoring* pMonitoring, std::vector<size_t> layerSizes)
    : m_pMonitoring (pMonitoring)
    , m_layerSizes (layerSizes)
    , m_xGrad (0)
    , m_xWeights (0)
    , m_countGrad (0)
    , m_countWeights (0)
{
}


inline Gnuplot* MinimizerMonitoring::plot (std::string plotName, 
					   std::string subName, 
					   std::string dataName, 
					   std::string style, 
					   std::string smoothing)
{
    if (!m_pMonitoring)
        return NULL;
    return m_pMonitoring->plot (plotName, subName, dataName, style, smoothing);
}



void MinimizerMonitoring::resetPlot (std::string plotName)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->resetPlot (plotName);
}


inline void MinimizerMonitoring::clearData (std::string dataName)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->clearData (dataName);
}



inline void MinimizerMonitoring::addPoint (std::string dataName, double x, double y)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->addPoint (dataName, x, y);
}

inline void MinimizerMonitoring::addPoint (std::string dataName, double x, double y, double err)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->addPoint (dataName, x, y, err);
}


template <typename Gradients>
inline void MinimizerMonitoring::plotGradients (const Gradients& gradients)
{
    if (m_countGrad % 100 == 0)
    {
    int index = 0;
    for (size_t size : m_layerSizes)
    {
        std::pair<double,double> meanVariance = computeMeanVariance (begin (gradients), begin (gradients) + size);

        std::stringstream sstrGrad;
        sstrGrad << "grad_" << index;
//        addPoint (sstrGrad.str (), m_xGrad, meanVariance.first, sqrt (meanVariance.second));
        addPoint (sstrGrad.str (), m_xGrad, sqrt (meanVariance.second));
        ++index;
    }
    ++m_xGrad;

    resetPlot ("gradients");
    for (int index = 0, indexEnd = m_layerSizes.size (); index < indexEnd; ++index)
    {
        std::stringstream sstrGrad;
        sstrGrad << "grad_" << index;
        plot ("gradients", sstrGrad.str (), sstrGrad.str (), "lines", "cspline");
    }
    }
    ++m_countGrad;
}

template <typename Weights>
inline void MinimizerMonitoring::plotWeights (const Weights& weights)
{
    if (m_countWeights % 100 == 0)
    {
    int index = 0;
    for (size_t size : m_layerSizes)
    {
        std::pair<double,double> meanVariance = computeMeanVariance (begin (weights), begin (weights) + size);

        std::stringstream sstrWeights;
        sstrWeights << "weights_" << index;
//        addPoint (sstrWeights.str (), m_xWeights, meanVariance.first, sqrt (meanVariance.second));
        addPoint (sstrWeights.str (), m_xWeights, sqrt (meanVariance.second));
        ++index;
    }
    ++m_xWeights;

    resetPlot ("weights");
    for (int index = 0, indexEnd = m_layerSizes.size (); index < indexEnd; ++index)
    {
        std::stringstream sstrWeights;
        sstrWeights << "weights_" << index;
        plot ("weights", sstrWeights.str (), sstrWeights.str (), "lines", "cspline");
    }
    }
    ++m_countWeights;
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



    template <typename Function, typename Weights, typename PassThrough>
        double Steepest::operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough) 
    {
	size_t numWeights = weights.size ();
	std::vector<NNTYPE> gradients (numWeights, 0.0);
	std::vector<NNTYPE> localWeights (begin (weights), end (weights));
        if (m_prevGradients.empty ())
            m_prevGradients.assign (weights.size (), 0);


        double Ebase = fitnessFunction (passThrough, weights, gradients);
        double Emin = Ebase;
        double E = Ebase;

        /* plotWeights (weights); */
        /* plotGradients (gradients); */


        bool success = true;
        size_t currentRepetition = 0;
        while (success)
        {
            if (currentRepetition >= m_repetitions)
                break;

            double alpha = m_alpha; //gaussDouble (m_alpha, m_alpha/2.0);

            auto itLocW = begin (localWeights);
            auto itLocWEnd = end (localWeights);
            auto itG = begin (gradients);
            auto itPrevG = begin (m_prevGradients);
            for (; itLocW != itLocWEnd; ++itLocW, ++itG, ++itPrevG)
            {
                (*itG) *= alpha;
                (*itG) += m_beta * (*itPrevG);
                (*itLocW) += (*itG);
                (*itPrevG) = (*itG);
            }
            gradients.assign (numWeights, 0.0);
            E = fitnessFunction (passThrough, localWeights, gradients);

            itLocW = begin (localWeights);
            itLocWEnd = end (localWeights);
            auto itW = begin (weights);
            for (; itLocW != itLocWEnd; ++itLocW, ++itW)
            {
                (*itW) = (*itLocW);
            }

            if (E < Emin)
            {
                Emin = E;
                std::cout << ".";
            }
            else
                std::cout << "X";
            ++currentRepetition;
        }
        return E;
    }





    template <typename Function, typename Weights, typename Gradients, typename PassThrough>
        double SteepestThreaded::fitWrapper (Function& function, PassThrough& passThrough, Weights weights)
    {
	return fitnessFunction (passThrough, weights);
    }



    template <typename Function, typename Weights, typename PassThrough>
        double SteepestThreaded::operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough) 
    {
	size_t numWeights = weights.size ();
	std::vector<double> gradients (numWeights, 0.0);
	std::vector<double> localWeights (begin (weights), end (weights));
        if (m_prevGradients.empty ())
            m_prevGradients.assign (weights.size (), 0);


        fitnessFunction (passThrough, weights, gradients);

        std::vector<std::future<double> > futures;
        std::vector<std::pair<double,double> > factors;
        for (size_t i = 0; i < m_repetitions; ++i)
        {
            std::vector<double> tmpWeights (weights);
            double alpha = gaussDouble (m_alpha, m_beta);
            double beta  = gaussDouble (m_alpha, m_beta);
            auto itGradient = begin (gradients);
            auto itPrevGradient = begin (m_prevGradients);
            std::for_each (begin (tmpWeights), end (tmpWeights), [alpha,beta,&itGradient,&itPrevGradient](double& w) 
                           { 
                               w += alpha * (*itGradient) + beta * (*itPrevGradient);
                               ++itGradient; ++itPrevGradient;
                           }
                );

	    // fitnessFunction is a function template which turns into a function at invocation
	    // if we call fitnessFunction directly in async, the templat parameters
	    // cannot be deduced correctly. Through the lambda function, the types are 
            // already deduced correctly for the lambda function and the async. The deduction for 
	    // the template function is then done from within the lambda function. 
	    futures.push_back (std::async (std::launch::async, [&fitnessFunction, &passThrough, tmpWeights]() mutable 
					   {  
					       return fitnessFunction (passThrough, tmpWeights); 
					   }) );

            factors.push_back (std::make_pair (alpha,beta));
        }

        // select best
        double bestAlpha = m_alpha, bestBeta = 0.0;
        auto itE = begin (futures);
        double bestE = 1e100;
        for (auto& alphaBeta : factors)
        {
            double E = (*itE).get ();
            if (E < bestE)
            {
                bestAlpha = alphaBeta.first;
                bestBeta = alphaBeta.second;
                bestE = E;
            }
            ++itE;
        }

        // walk this way
        auto itGradient = begin (gradients);
        auto itPrevGradient = begin (m_prevGradients);
        std::for_each (begin (weights), end (weights), [bestAlpha,bestBeta,&itGradient,&itPrevGradient](double& w) 
                       { 
                           double grad = bestAlpha * (*itGradient) + bestBeta * (*itPrevGradient);
                           w += grad;
                           (*itPrevGradient) = grad;
                           ++itGradient; ++itPrevGradient;
                       }
            );
        return bestE;
    }








    template <typename Function, typename Weights, typename PassThrough>
        double MaxGradWeight::operator() (Function& fitnessFunction, const Weights& weights, PassThrough& passThrough) 
    {
	double alpha = m_learningRate;

	size_t numWeights = weights.size ();
	std::vector<double> gradients (numWeights, 0.0);
	std::vector<double> localWeights (begin (weights), end (weights));


        double Ebase = fitnessFunction (passThrough, weights, gradients);
        double Emin = Ebase;

        bool success = true;
        size_t currentRepetition = 0;
        while (success)
        {
            if (currentRepetition >= m_repetitions)
                break;

	    auto itMaxGradElement = std::max_element (begin (gradients), end (gradients));
	    auto idx = std::distance (begin (gradients), itMaxGradElement);
	    localWeights.at (idx) += alpha*(*itMaxGradElement);
            gradients.assign (numWeights, 0.0);
            double E = fitnessFunction (passThrough, localWeights, gradients);

            if (E < Emin)
            {
                Emin = E;

                auto itLocW = begin (localWeights);
                auto itLocWEnd = end (localWeights);
                auto itW = begin (weights);
                for (; itLocW != itLocWEnd; ++itLocW, ++itW)
                {
                    (*itW) = (*itLocW);
                }
            }
            ++currentRepetition;
        }
        return Emin;
    }









template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
double sumOfSquares (ItOutput itOutputBegin, ItOutput itOutputEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight) 
{
    double errorSum = 0.0;

    // output - truth
    ItTruth itTruth = itTruthBegin;
    bool hasDeltas = (itDelta != itDeltaEnd);
    for (ItOutput itOutput = itOutputBegin; itOutput != itOutputEnd; ++itOutput, ++itTruth)
    {
	assert (itTruth != itTruthEnd);
	double output = (*itOutput);
	double error = output - (*itTruth);
	if (hasDeltas)
	{
	    (*itDelta) = (*itInvActFnc)(output) * error * patternWeight;
	    ++itDelta; ++itInvActFnc;
	}
	errorSum += error*error  * patternWeight;
    }

    return 0.5*errorSum;
}



template <typename ItProbability, typename ItTruth, typename ItDelta, typename ItInvActFnc>
double crossEntropy (ItProbability itProbabilityBegin, ItProbability itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight) 
{
    bool hasDeltas = (itDelta != itDeltaEnd);
    
    double errorSum = 0.0;
    for (ItProbability itProbability = itProbabilityBegin; itProbability != itProbabilityEnd; ++itProbability)
    {
        double probability = *itProbability;
        double truth = *itTruthBegin;
        truth = truth < 0.1 ? 0.1 : truth;
        truth = truth > 0.9 ? 0.9 : truth;
        if (hasDeltas)
        {
            double delta = probability - truth;
	    (*itDelta) = delta*patternWeight;
//	    (*itDelta) = (*itInvActFnc)(probability) * delta * patternWeight;
            ++itDelta;
        }
        double error (0);
        if (probability == 0) // protection against log (0)
        {
            if (truth >= 0.5)
                error += 1.0;
        }
        else if (probability == 1)
        {
            if (truth < 0.5)
                error += 1.0;
        }
        else
            error += - (truth * log (probability) + (1.0-truth) * log (1.0-probability)); // cross entropy function
        errorSum += error * patternWeight;
        
    }
    return errorSum;
}




template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
double softMaxCrossEntropy (ItOutput itProbabilityBegin, ItOutput itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight) 
{
    double errorSum = 0.0;

    bool hasDeltas = (itDelta != itDeltaEnd);
    // output - truth
    ItTruth itTruth = itTruthBegin;
    for (auto itProbability = itProbabilityBegin; itProbability != itProbabilityEnd; ++itProbability, ++itTruth)
    {
	assert (itTruth != itTruthEnd);
	double probability = (*itProbability);
	double truth = (*itTruth);
	if (hasDeltas)
	{
            (*itDelta) = probability - truth;
//	    (*itDelta) = (*itInvActFnc)(sm) * delta * patternWeight;
	    ++itDelta; //++itInvActFnc;
	}
        double error (0);

	error += truth * log (probability);
	errorSum += error;
    }

    return -errorSum * patternWeight;
}







template <typename ItWeight>
double weightDecay (double error, ItWeight itWeight, ItWeight itWeightEnd, double factorWeightDecay)
{

    // weight decay (regularization)
    double w = 0;
    double sumW = 0;
    for (; itWeight != itWeightEnd; ++itWeight)
    {
	double weight = (*itWeight);
	w += weight*weight;
        sumW += fabs (weight);
    }
    return error + 0.5 * w * factorWeightDecay / sumW;
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




    LayerData::LayerData (size_t size, 
	       const_iterator_type itWeightBegin, 
	       iterator_type itGradientBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       const_function_iterator_type itInverseFunctionBegin,
	       ModeOutputValues eModeOutput)
	: m_size (size)
	, m_itConstWeightBegin   (itWeightBegin)
	, m_itGradientBegin (itGradientBegin)
	, m_itFunctionBegin (itFunctionBegin)
	, m_itInverseFunctionBegin (itInverseFunctionBegin)
	, m_isInputLayer (false)
	, m_hasWeights (true)
	, m_hasGradients (true)
	, m_eModeOutput (eModeOutput) 
    {
	m_values.assign (size, 0);
	m_deltas.assign (size, 0);
	m_valueGradients.assign (size, 0);
    }




    LayerData::LayerData (size_t size, const_iterator_type itWeightBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       ModeOutputValues eModeOutput)
	: m_size (size)
	, m_itConstWeightBegin   (itWeightBegin)
	, m_itFunctionBegin (itFunctionBegin)
	, m_isInputLayer (false)
	, m_hasWeights (true)
	, m_hasGradients (false)
	, m_eModeOutput (eModeOutput) 
    {
	m_values.assign (size, 0);
    }



    typename LayerData::container_type LayerData::computeProbabilities ()
    {
	container_type probabilities;
	switch (m_eModeOutput)
	{
	case ModeOutputValues::SIGMOID:
        {
	    std::transform (begin (m_values), end (m_values), std::back_inserter (probabilities), Sigmoid);
	    break;
        }
	case ModeOutputValues::SOFTMAX:
        {
            double sum = 0;
            probabilities = m_values;
            std::for_each (begin (probabilities), end (probabilities), [&sum](double& p){ p = std::exp (p); sum += p; });
            if (sum != 0)
                std::for_each (begin (probabilities), end (probabilities), [sum ](double& p){ p /= sum; });
	    break;
        }
	case ModeOutputValues::DIRECT:
	default:
	    probabilities.assign (begin (m_values), end (m_values));
	}
	return probabilities;
    }



std::ostream& operator<< (std::ostream& ostr, LayerData const& data)
{
    ostr << "---LAYER---";
    ostr << "size= " << data.m_size << "   ";
    if (data.m_isInputLayer)
    {
        ostr << "input layer, nodes: {";
        for (auto it = data.m_itInputBegin; it != data.m_itInputEnd; ++it)
            ostr << (*it) << ", ";
        ostr << "}   ";
    }
    else
    {
        ostr << "nodes: {";
        for (auto it = begin (data.m_values), itEnd = end (data.m_values); it != itEnd; ++it)
            ostr << (*it) << ", ";
        ostr << "}   ";
    }
    ostr << "deltas: {";
    for (auto it = begin (data.m_deltas), itEnd = end (data.m_deltas); it != itEnd; ++it)
        ostr << (*it) << ", ";
    ostr << "}   ";
    if (data.m_hasWeights)
    {
        ostr << "weights: {" << (*data.weightsBegin ()) << ", ...}   ";
    }
    if (data.m_hasGradients)
    {
        ostr << "gradients: {" << (*data.gradientsBegin ()) << ", ...}   ";
    }
    return ostr;
}




    Layer::Layer (size_t numNodes, EnumFunction activationFunction, ModeOutputValues eModeOutputValues) 
	: m_numNodes (numNodes) 
	, m_eModeOutputValues (eModeOutputValues)
    {
	for (size_t iNode = 0; iNode < numNodes; ++iNode)
	{
	    auto actFnc = Linear;
	    auto invActFnc = InvLinear;
	    m_activationFunction = EnumFunction::LINEAR;
	    switch (activationFunction)
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






template <typename LAYERDATA>
void forward (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		  currLayerData.weightsBegin (), 
		  currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin ());
}

template <typename LAYERDATA>
void forward_training (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		  currLayerData.weightsBegin (), 
		  currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin (), 
		    currLayerData.inverseFunctionBegin (), currLayerData.valueGradientsBegin ());
}


template <typename LAYERDATA>
void backward (LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    applyWeightsBackwards (currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			   currLayerData.weightsBegin (), 
			   prevLayerData.deltasBegin (), prevLayerData.deltasEnd ());
}



template <typename LAYERDATA>
void update (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData, double weightDecay, bool isL1)
{
    if (weightDecay != 0.0) // has weight regularization
	if (isL1)  // L1 regularization ( sum(|w|) )
	{
	    update<true> (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
			  currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			  currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin (), 
			  currLayerData.weightsBegin (), weightDecay);
	}
	else // L2 regularization ( sum(w^2) )
	{
	    update<false> (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
			   currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			   currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin (), 
			   currLayerData.weightsBegin (), weightDecay);
	}
    else
    { // no weight regularization
	update (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
		currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin ());
    }
}



Settings::Settings (size_t _convergenceSteps, size_t _batchSize, size_t _testRepetitions, 
                    double _factorWeightDecay, bool isL1Regularization, double dropFraction,
                    size_t dropRepetitions, Monitoring* pMonitoring)
    : m_convergenceSteps (_convergenceSteps)
    , m_batchSize (_batchSize)
    , m_testRepetitions (_testRepetitions)
    , m_factorWeightDecay (_factorWeightDecay)
    , count_E (0)
    , count_dE (0)
    , count_mb_E (0)
    , count_mb_dE (0)
    , m_isL1Regularization (isL1Regularization)
    , m_dropFraction (dropFraction)
    , m_dropRepetitions (dropRepetitions)
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




inline void Monitoring::clearData (std::string dataName)
{
    DataMap::mapped_type& data = getData (dataName);
    data.clear ();
}


inline void Settings::clearData (std::string dataName)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->clearData (dataName);
}


inline void Monitoring::addPoint (std::string dataName, double x, double y)
{
    DataMap::mapped_type& data = getData (dataName);
    data.useErr = false;

    data.x.push_back (x);
    data.y.push_back (y);
}

inline void Monitoring::addPoint (std::string dataName, double x, double y, double err)
{
    DataMap::mapped_type& data = getData (dataName);
    data.useErr = true;

    data.x.push_back (x);
    data.y.push_back (y);
    data.err.push_back (err);
}


inline void Settings::addPoint (std::string dataName, double x, double y)
{
    if (!m_pMonitoring)
        return;
    m_pMonitoring->addPoint (dataName, x, y);
}



inline Gnuplot* Monitoring::plot (std::string plotName, std::string subName, std::string dataName, std::string style, std::string smoothing)
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



inline Gnuplot* Settings::plot (std::string plotName, std::string subName, std::string dataName, std::string style, std::string smoothing)
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




    template <typename ItPat, typename OutIterator>
    void Net::initializeWeights (WeightInitializationStrategy eInitStrategy, 
				     ItPat itPatternBegin, 
				     ItPat itPatternEnd, 
				     OutIterator itWeight)
    {
        if (eInitStrategy == WeightInitializationStrategy::XAVIER)
        {
            // input and output properties
            int numInput = (*itPatternBegin).inputSize ();

            // compute variance and mean of input and output
            //...
	

            // compute the weights
            for (auto& layer: layers ())
            {
                double nIn = numInput;
                for (size_t iWeight = 0, iWeightEnd = layer.numWeights (numInput); iWeight < iWeightEnd; ++iWeight)
                {
                    (*itWeight) = NN::gaussDouble (0.0, sqrt (2.0/nIn)); // factor 2.0 for ReLU
                    ++itWeight;
                }
                numInput = layer.numNodes ();
            }
            return;
        }

        if (eInitStrategy == WeightInitializationStrategy::TEST)
        {
            // input and output properties
            int numInput = (*itPatternBegin).inputSize ();

            // compute variance and mean of input and output
            //...
	

            // compute the weights
            for (auto& layer: layers ())
            {
//                double nIn = numInput;
                for (size_t iWeight = 0, iWeightEnd = layer.numWeights (numInput); iWeight < iWeightEnd; ++iWeight)
                {
                    (*itWeight) = NN::gaussDouble (0.0, 0.1);
                    ++itWeight;
                }
                numInput = layer.numNodes ();
            }
            return;
        }

        if (eInitStrategy == WeightInitializationStrategy::LAYERSIZE)
        {
            // input and output properties
            int numInput = (*itPatternBegin).inputSize ();

            // compute variance and mean of input and output
            //...
	

            // compute the weights
            for (auto& layer: layers ())
            {
                double nIn = numInput;
                for (size_t iWeight = 0, iWeightEnd = layer.numWeights (numInput); iWeight < iWeightEnd; ++iWeight)
                {
                    (*itWeight) = NN::gaussDouble (0.0, sqrt (layer.numWeights (nIn))); // factor 2.0 for ReLU
                    ++itWeight;
                }
                numInput = layer.numNodes ();
            }
            return;
        }

    }





    template <typename WeightsType>
        void Net::dropOutWeightFactor (const DropContainer& dropContainer, WeightsType& weights, double factor)
    {
//        return;
	// reduce weights because of dropped nodes
	// if dropOut enabled
	if (dropContainer.empty ())
	    return;

	// fill the dropOut-container
	auto itWeight = begin (weights);
	auto itDrop = begin (dropContainer);
	for (auto itLayer = begin (m_layers), itLayerEnd = end (m_layers)-1; itLayer != itLayerEnd; ++itLayer)
//	for (auto itLayer = begin (m_layers), itLayerEnd = end (m_layers); itLayer != itLayerEnd; ++itLayer)
	{
	    auto& layer = *itLayer;
	    auto& nextLayer = *(itLayer+1);
	    /* // in the first and last layer, all the nodes are always on */
	    /* if (itLayer == begin (m_layers)) // is first layer */
	    /* { */
	    /*     itDrop += layer.numNodes (); */
	    /*     itWeight += layer.numNodes () * nextLayer.numNodes (); */
	    /*     continue; */
	    /* } */

	    auto itLayerDrop = itDrop;
	    for (size_t i = 0, iEnd = layer.numNodes (); i < iEnd; ++i)
	    {
		auto itNextDrop = itDrop + layer.numNodes ();
	    
		bool drop = (*itLayerDrop);
		for (size_t j = 0, jEnd = nextLayer.numNodes (); j < jEnd; ++j)
		{
		    if (drop && (*itNextDrop))
		    {
			(*itWeight) *= factor;
		    }
		    ++itWeight;
		    ++itNextDrop;
		}
		++itLayerDrop;
	    }
	}
        std::cout << std::endl;
        std::cout << "drop out fraction " << factor << std::endl;
        for (char c : dropContainer)
        {
            std::cout << ((short)c);
        }
        std::cout << std::endl;
        std::copy (dropContainer.begin (), dropContainer.end (), std::ostream_iterator<short>(std::cout, ""));
    }



    template <typename Minimizer>
        double Net::train (std::vector<double>& weights, 
		  std::vector<Pattern>& trainPattern, 
		  const std::vector<Pattern>& testPattern, 
                  Minimizer& minimizer, Settings& settings)
    {
        settings.clearData ("trainErrors");
        settings.clearData ("testErrors");
        std::cout << "START TRAINING" << std::endl;
        size_t convergenceCount = 0;
        double minError = 1e10;

        size_t cycleCount = 0;
        size_t testCycleCount = 0;
        double testError = 1e20;
        size_t dropOutChangeCount = 0;

	DropContainer dropContainer;

        // until convergence
        do
        {
            std::cout << "train cycle " << cycleCount << std::endl;
            ++cycleCount;

	    // shuffle training pattern
//            std::random_shuffle (begin (trainPattern), end (trainPattern)); // is done in the training cycle
	    double dropFraction = settings.dropFraction ();
            std::cout << "train cycle drop fraction = " << dropFraction << std::endl;

	    // if dropOut enabled
            if (dropFraction > 0 && dropOutChangeCount % settings.dropRepetitions () == 0)
	    {
		if (dropOutChangeCount > 0)
		    dropOutWeightFactor (dropContainer, weights, dropFraction);

		// fill the dropOut-container
		dropContainer.clear ();
		for (auto itLayer = begin (m_layers), itLayerEnd = end (m_layers); itLayer != itLayerEnd; ++itLayer)
		{
		    auto& layer = *itLayer;
		    // in the first and last layer, all the nodes are always on
		    if (itLayer == begin (m_layers) || itLayer == end (m_layers)-1) // is first layer or is last layer
		    {
			dropContainer.insert (end (dropContainer), layer.numNodes (), true);
			continue;
		    }
		    // how many nodes have to be dropped
		    size_t numDrops = settings.dropFraction () * layer.numNodes ();
		    dropContainer.insert (end (dropContainer), layer.numNodes ()-numDrops, true); // add the markers for the nodes which are enabled
		    dropContainer.insert (end (dropContainer), numDrops, false); // add the markers for the disabled nodes
		    // shuffle 
		    std::random_shuffle (end (dropContainer)-layer.numNodes (), end (dropContainer)); // shuffle enabled and disabled markers
		}
		if (dropOutChangeCount > 0)
                    dropOutWeightFactor (dropContainer, weights, 1.0/dropFraction);
	    }

	    // execute training cycle
            double trainError = trainCycle (minimizer, weights, begin (trainPattern), end (trainPattern), settings, dropContainer);

	    

	    // check if we execute a test
            if (testCycleCount % settings.testRepetitions () == 0)
            {
		if (dropOutChangeCount > 0)
		    dropOutWeightFactor (dropContainer, weights, dropFraction);

		dropContainer.clear (); // execute test on all the nodes
                testError = 0;
                double weightSum = 0;
                settings.startTestCycle ();
                std::vector<double> output;
                for (auto it = begin (testPattern), itEnd = end (testPattern); it != itEnd; ++it)
                {
                    const Pattern& p = (*it);
                    double weight = p.weight ();
                    Batch batch (it, it+1);
                    output.clear ();
		    std::tuple<Settings&, Batch&, DropContainer&> passThrough (settings, batch, dropContainer);
                    double testPatternError = (*this) (passThrough, weights, ModeOutput::FETCH, output);
                    if (output.size () == 1)
		    {
                        settings.testSample (testPatternError, output.at (0), p.output ().at (0), weight);
		    }
                    weightSum += fabs (weight);
                    testError += testPatternError*weight;
                }
                settings.endTestCycle ();
                testError /= weightSum;

		settings.computeResult (*this, weights);
            }
            ++testCycleCount;
	    ++dropOutChangeCount;


            static double x = -1.0;
            x += 1.0;
            settings.resetPlot ("errors");
            settings.addPoint ("trainErrors", cycleCount, trainError);
            settings.addPoint ("testErrors", cycleCount, testError);
//            settings.plot ("errors", "training_", "trainErrors", "points", "");
            settings.plot ("errors", "training", "trainErrors", "lines", "cspline");
//            settings.plot ("errors", "test_", "testErrors", "points", "");
            settings.plot ("errors", "test", "testErrors", "lines", "cspline");



//	(*this).print (std::cout);
            std::cout << "check convergence; minError " << minError << "  current " << testError << "  current convergence count " << convergenceCount << std::endl;
            if (testError < minError)
            {
                convergenceCount = 0;
                minError = testError;
            }
            else
                ++convergenceCount;


	    if (convergenceCount >= settings.convergenceSteps ())
	    {
		if (dropOutChangeCount > 0)
		    dropOutWeightFactor (dropContainer, weights, dropFraction);
		break;
	    }


            std::cout << "testError : " << testError << "   trainError : " << trainError << std::endl;
        }
	while (true);

        std::cout << "END TRAINING" << std::endl;
        return testError;
    }



    template <typename Iterator, typename Minimizer>
        inline double Net::trainCycle (Minimizer& minimizer, std::vector<double>& weights, 
			      Iterator itPatternBegin, Iterator itPatternEnd, Settings& settings, DropContainer& dropContainer)
    {
	double error = 0.0;
	size_t numPattern = std::distance (itPatternBegin, itPatternEnd);
	size_t numBatches = numPattern/settings.batchSize ();
	size_t numBatches_stored = numBatches;

	Iterator itPatternBatchBegin = itPatternBegin;
	Iterator itPatternBatchEnd = itPatternBatchBegin;
	std::random_shuffle (itPatternBegin, itPatternEnd);
	while (numBatches > 0)
	{
	    std::advance (itPatternBatchEnd, settings.batchSize ());
            Batch batch (itPatternBatchBegin, itPatternBatchEnd);
            std::tuple<Settings&, Batch&, DropContainer&> settingsAndBatch (settings, batch, dropContainer);
	    error += minimizer ((*this), weights, settingsAndBatch);
	    itPatternBatchBegin = itPatternBatchEnd;
	    --numBatches;
	}
	if (itPatternBatchEnd != itPatternEnd)
        {
            Batch batch (itPatternBatchEnd, itPatternEnd);
            std::tuple<Settings&, Batch&, DropContainer&> settingsAndBatch (settings, batch, dropContainer);
	    error += minimizer ((*this), weights, settingsAndBatch);
        }
	error /= numBatches_stored;
    
	return error;
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


    template <typename Weights>
        std::vector<double> Net::compute (const std::vector<double>& input, const Weights& weights) const
    {
	std::vector<LayerData> layerData;
	layerData.reserve (m_layers.size ()+1);
	auto itWeight = begin (weights);
	auto itInputBegin = begin (input);
	auto itInputEnd = end (input);
	DropContainer drop;
	layerData.push_back (LayerData (itInputBegin, itInputEnd));
	size_t numNodesPrev = input.size ();
	for (auto& layer: m_layers)
	{
	    layerData.push_back (LayerData (layer.numNodes (), itWeight, 
						   begin (layer.activationFunctions ()),
						   layer.modeOutputValues ()));
	    size_t numWeights = layer.numWeights (numNodesPrev);
	    itWeight += numWeights;
	    numNodesPrev = layer.numNodes ();
	}
	    

	// --------- forward -------------
	size_t idxLayer = 0, idxLayerEnd = m_layers.size ();
	for (; idxLayer < idxLayerEnd; ++idxLayer)
	{
	    LayerData& prevLayerData = layerData.at (idxLayer);
	    LayerData& currLayerData = layerData.at (idxLayer+1);
		
	    forward (prevLayerData, currLayerData);
	}

	// ------------- fetch output ------------------
	if (layerData.back ().outputMode () == ModeOutputValues::DIRECT)
	{
	    std::vector<double> output;
	    output.assign (layerData.back ().valuesBegin (), layerData.back ().valuesEnd ());
	    return output;
	}
	return layerData.back ().probabilities ();
    }


    template <typename Weights, typename PassThrough>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights) const
    {
	std::vector<double> nothing; // empty gradients; no backpropagation is done, just forward
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (nothing), std::end (nothing), 100, nothing, false);
        return error;
    }

    template <typename Weights, typename PassThrough, typename OutContainer>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights, ModeOutput eFetch, OutContainer& outputContainer) const
    {
	std::vector<double> nothing; // empty gradients; no backpropagation is done, just forward
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (nothing), std::end (nothing), 1000, outputContainer, true);
        return error;
    }

    
    template <typename Weights, typename Gradients, typename PassThrough>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients) const
    {
        std::vector<double> nothing;
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (gradients), std::end (gradients), 0, nothing, false);
        return error;
    }

    template <typename Weights, typename Gradients, typename PassThrough, typename OutContainer>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients, ModeOutput eFetch, OutContainer& outputContainer) const
    {
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (gradients), std::end (gradients), 0, outputContainer, true);
        return error;
    }





    template <typename LayerContainer, typename PassThrough, typename ItWeight, typename ItGradient, typename OutContainer>
        double Net::forward_backward (LayerContainer& layers, PassThrough& settingsAndBatch, 
			     ItWeight itWeightBegin, 
			     ItGradient itGradientBegin, ItGradient itGradientEnd, 
			     size_t trainFromLayer, 
			     OutContainer& outputContainer, bool fetchOutput) const
    {
        Settings& settings = std::get<0>(settingsAndBatch);
        Batch& batch = std::get<1>(settingsAndBatch);
	DropContainer& drop = std::get<2>(settingsAndBatch);
	
	bool usesDropOut = !drop.empty ();

	std::vector<std::vector<std::function<double(double)> > > activationFunctionsDropOut;
	std::vector<std::vector<std::function<double(double)> > > inverseActivationFunctionsDropOut;

	if (layers.empty ())
	    throw std::string ("no layers in this net");

	if (usesDropOut)
	{
	    auto itDrop = begin (drop);
	    for (auto& layer: layers)
	    {
		activationFunctionsDropOut.push_back (std::vector<std::function<double(double)> >());
		inverseActivationFunctionsDropOut.push_back (std::vector<std::function<double(double)> >());
		auto& actLine = activationFunctionsDropOut.back ();
		auto& invActLine = inverseActivationFunctionsDropOut.back ();
		auto& actFnc = layer.activationFunctions ();
		auto& invActFnc = layer.inverseActivationFunctions ();
		for (auto itAct = begin (actFnc), itActEnd = end (actFnc), itInv = begin (invActFnc); itAct != itActEnd; ++itAct, ++itInv)
		{
		    if (!*itDrop)
		    {
			actLine.push_back (ZeroFnc);
			invActLine.push_back (ZeroFnc);
		    }
		    else
		    {
			actLine.push_back (*itAct);
			invActLine.push_back (*itInv);
		    }
		    ++itDrop;
		}
	    }
	}

	double sumError = 0.0;
	double sumWeights = 0.0;	// -------------
	for (const Pattern& pattern : batch)
	{
	    assert (layers.back ().numNodes () == pattern.output ().size ());
	    size_t totalNumWeights = 0;
	    std::vector<LayerData> layerData;
            layerData.reserve (layers.size ()+1);
	    ItWeight itWeight = itWeightBegin;
	    ItGradient itGradient = itGradientBegin;
	    typename Pattern::const_iterator itInputBegin = pattern.beginInput ();
	    typename Pattern::const_iterator itInputEnd = pattern.endInput ();
	    layerData.push_back (LayerData (itInputBegin, itInputEnd));
	    size_t numNodesPrev = pattern.input ().size ();
	    auto itActFncLayer = begin (activationFunctionsDropOut);
	    auto itInvActFncLayer = begin (inverseActivationFunctionsDropOut);
	    for (auto& layer: layers)
	    {
		const std::vector<std::function<double(double)> >& actFnc = usesDropOut ? (*itActFncLayer) : layer.activationFunctions ();
		const std::vector<std::function<double(double)> >& invActFnc = usesDropOut ? (*itInvActFncLayer) : layer.inverseActivationFunctions ();
		if (usesDropOut)
		{
		    ++itActFncLayer;
		    ++itInvActFncLayer;
		}
		if (itGradientBegin == itGradientEnd)
		    layerData.push_back (LayerData (layer.numNodes (), itWeight, 
						    begin (actFnc),
						    layer.modeOutputValues ()));
		else
		    layerData.push_back (LayerData (layer.numNodes (), itWeight, itGradient, 
						    begin (actFnc), begin (invActFnc),
						    layer.modeOutputValues ()));
		size_t numWeights = layer.numWeights (numNodesPrev);
		totalNumWeights += numWeights;
		itWeight += numWeights;
		itGradient += numWeights;
		numNodesPrev = layer.numNodes ();
//                std::cout << layerData.back () << std::endl;
	    }
	    

	    // --------- forward -------------
//            std::cout << "forward" << std::endl;
	    bool doTraining (true);
	    size_t idxLayer = 0, idxLayerEnd = layers.size ();
	    for (; idxLayer < idxLayerEnd; ++idxLayer)
	    {
		LayerData& prevLayerData = layerData.at (idxLayer);
		LayerData& currLayerData = layerData.at (idxLayer+1);
		
		doTraining = idxLayer >= trainFromLayer;
		if (doTraining)
		    forward_training (prevLayerData, currLayerData);
		else
		    forward (prevLayerData, currLayerData);
	    }

            
            // ------------- fetch output ------------------
            if (fetchOutput)
            {
		if (layerData.back ().outputMode () == ModeOutputValues::DIRECT)
		    outputContainer.insert (outputContainer.end (), layerData.back ().valuesBegin (), layerData.back ().valuesEnd ());
		else
		    outputContainer = layerData.back ().probabilities ();
            }


	    // ------------- error computation -------------
	    // compute E and the deltas of the computed output and the true output 
	    itWeight = itWeightBegin;
	    double error = errorFunction (layerData.back (), pattern.output (), 
					  itWeight, itWeight + totalNumWeights, 
					  pattern.weight (), settings.factorWeightDecay ());
	    sumWeights += fabs (pattern.weight ());
	    sumError += error;

	    if (!doTraining) // no training
		continue;

	    // ------------- backpropagation -------------
	    idxLayer = layerData.size ();
	    for (auto itLayer = end (layers), itLayerBegin = begin (layers); itLayer != itLayerBegin; --itLayer)
	    {
		--idxLayer;
		doTraining = idxLayer >= trainFromLayer;
		if (!doTraining) // no training
		    break;

		LayerData& currLayerData = layerData.at (idxLayer);
		LayerData& prevLayerData = layerData.at (idxLayer-1);

		backward (prevLayerData, currLayerData);
		update (prevLayerData, currLayerData, settings.factorWeightDecay ()/sumWeights, settings.isL1 ());
	    }
	}
        
        double batchSize = std::distance (begin (batch), end (batch));
        for (auto it = itGradientBegin; it != itGradientEnd; ++it)
            (*it) /= batchSize;


	sumError /= sumWeights;
	return sumError;
    }



    


    template <typename Container, typename ItWeight>
        double Net::errorFunction (LayerData& layerData, Container truth, ItWeight itWeight, ItWeight itWeightEnd, double patternWeight, double factorWeightDecay) const
    {
	double error (0);
	switch (m_eErrorFunction)
	{
	case ModeErrorFunction::SUMOFSQUARES:
	{
	    error = sumOfSquares (layerData.valuesBegin (), layerData.valuesEnd (), begin (truth), end (truth), 
				  layerData.deltasBegin (), layerData.deltasEnd (), 
				  layerData.inverseFunctionBegin (), 
				  patternWeight);
	    break;
	}
	case ModeErrorFunction::CROSSENTROPY:
	{
	    assert (layerData.outputMode () != ModeOutputValues::DIRECT);
	    std::vector<double> probabilities = layerData.probabilities ();
	    error = crossEntropy (begin (probabilities), end (probabilities), 
				  begin (truth), end (truth), 
				  layerData.deltasBegin (), layerData.deltasEnd (), 
				  layerData.inverseFunctionBegin (), 
				  patternWeight);
	    break;
	}
	case ModeErrorFunction::CROSSENTROPY_MUTUALEXCLUSIVE:
	{
	    assert (layerData.outputMode () != ModeOutputValues::DIRECT);
	    std::vector<double> probabilities = layerData.probabilities ();
	    error = softMaxCrossEntropy (begin (probabilities), end (probabilities), 
					 begin (truth), end (truth), 
					 layerData.deltasBegin (), layerData.deltasEnd (), 
					 layerData.inverseFunctionBegin (), 
					 patternWeight);
	    break;
	}
	}
	if (factorWeightDecay != 0)
	    error = weightDecay (error, itWeight, itWeightEnd, factorWeightDecay);
	return error;
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





















}; // namespace NN
