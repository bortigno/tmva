#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <tuple>
#include <math.h>
#include <cassert>
#include <random>
#include <thread>
#include <future>
//#include <boost/iterator/zip_iterator.hpp>
#include "../gnuplotWrapper/gnuplot_i.hpp" //Gnuplot class handles POSIX-Pipe-communikation with Gnuplot


#include <fenv.h>

#include "../pattern/pattern.hpp"

#define NNTYPE double




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










int randomInt (int maxValue)
{
    static std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,maxValue-1);
    return distribution(generator);
}


double uniformDouble (double minValue, double maxValue)
{
    static std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(minValue, maxValue);
    return distribution(generator);
}

double studenttDouble (double distributionParameter)
{
    static std::default_random_engine generator;
    std::student_t_distribution<double> distribution (distributionParameter);
    return distribution (generator);
}

double gaussDouble (double mean, double sigma)
{
    static std::default_random_engine generator;
    std::normal_distribution<double> distribution (mean, sigma);
    return distribution (generator);
}




// // http://kbokonseriousstuff.blogspot.co.at/2011/09/using-reverse-iterators-with-c11-range.html
// template<class Cont>
// class const_reverse_wrapper {
//     const Cont& container;

// public:
//     const_reverse_wrapper(const Cont& cont) : container(cont){ }
//     decltype(container.rbegin()) begin() const { return container.rbegin(); }
//     decltype(container.rend()) end() const { return container.rend(); }
// };

// template<class Cont>
// class reverse_wrapper {
//     Cont& container;

// public:
//     reverse_wrapper(Cont& cont) : container(cont){ }
//     decltype(container.rbegin()) begin() { return container.rbegin(); }
//     decltype(container.rend()) end() { return container.rend(); }
// };

// template<class Cont>
// const_reverse_wrapper<Cont> reverse(const Cont& cont) {
//     return const_reverse_wrapper<Cont>(cont);
// }

// template<class Cont>
// reverse_wrapper<Cont> reverse(Cont& cont) {
//     return reverse_wrapper<Cont>(cont);
// }





enum class EnumFunction
{
    LINEAR = 'L',
    TANH = 'T',
    RELU = 'R',
    SYMMRELU = 'r',
    TANHSHIFT = 't',
    SIGMOID = 's',
    SOFTSIGN = 'S',
    GAUSS = 'G',
    GAUSSCOMPLEMENT = 'C',
    DOUBLEINVERTEDGAUSS = 'D'
};

std::function<double(double)> Sigmoid = [](double value){ value = std::max (-100.0, std::min (100.0,value)); return 1.0/(1.0 + std::exp (-value)); };
std::function<double(double)> InvSigmoid = [](double value){ double s = Sigmoid (value); return s*(1.0-s); };
//std::function<double(double)> InvSigmoid = [](double value){ return 1.0; };

std::function<double(double)> Tanh = [](double value){ return tanh (value); };
std::function<double(double)> InvTanh = [](double value){ return 1.0 - std::pow (value, 2.0); };

std::function<double(double)> Linear = [](double value){ return value; };
std::function<double(double)>  InvLinear = [](double value){ return 1.0; };

std::function<double(double)> SymmReLU = [](double value){ const double margin = 0.3; return value > margin ? value-margin : value < -margin ? value+margin : 0; };
std::function<double(double)> InvSymmReLU = [](double value){ const double margin = 0.3; return value > margin ? 1.0 : value < -margin ? 1.0 : 0; };

std::function<double(double)> ReLU = [](double value){ const double margin = 0.3; return value > margin ? value-margin : 0; };
std::function<double(double)> InvReLU = [](double value){ const double margin = 0.3; return value > margin ? 1.0 : 0; };

std::function<double(double)> SoftPlus = [](double value){ return std::log (1.0+ std::exp (value)); };
std::function<double(double)> InvSoftPlus = [](double value){ return 1.0 / (1.0 + std::exp (-value)); };

std::function<double(double)> TanhShift = [](double value){ return tanh (value-0.3); };
std::function<double(double)> InvTanhShift = [](double value){ return 0.3 + (1.0 - std::pow (value, 2.0)); };

std::function<double(double)> SoftSign = [](double value){ return value / (1.0 + fabs (value)); };
std::function<double(double)> InvSoftSign = [](double value){ return std::pow ((1.0 - fabs (value)),2.0); };

std::function<double(double)> Gauss = [](double value){ const double s = 6.0; return exp (-std::pow(value*s,2.0)); };
std::function<double(double)> InvGauss = [](double value){ const double s = 6.0; return -2.0 * value * s*s * Gauss (value); };

std::function<double(double)> GaussComplement = [](double value){ const double s = 6.0; return 1.0 - exp (-std::pow(value*s,2.0));; };
std::function<double(double)> InvGaussComplement = [](double value){ const double s = 6.0; return +2.0 * value * s*s * GaussComplement (value); };

std::function<double(double)> DoubleInvertedGauss = [](double value)
{ const double s = 8.0; const double shift = 0.1; return exp (-std::pow((value-shift)*s,2.0)) - exp (-std::pow((value+shift)*s,2.0)); };
std::function<double(double)> InvDoubleInvertedGauss = [](double value)
{ const double s = 8.0; const double shift = 0.1; return -2.0 * (value-shift) * s*s * DoubleInvertedGauss (value-shift) + 2.0 * (value+shift) * s*s * DoubleInvertedGauss (value+shift);  };



// class Peak
// {
// public:
//     const double s = 0.2;

//     virtual inline NNTYPE apply (NNTYPE value) const 
//     {   
//         if (value < -s)
//             return 0;
//         if (value > s)
//             return 0;
//         if (value <=0)
//             return (value/s) +1.0;
//         return -value/s + 1.0;
//     }
//     virtual inline NNTYPE applyInverse (NNTYPE value) const 
//     { 
//         if (value < -s)
//             return 0;
//         if (value > s)
//             return 0;
//         if (value <=0)
//             return 1.0/s;
//         return -1.0/s;
//     }
// };





class Net;

void writeKaggleHiggs (std::string fileName, const Net& net, const std::vector<double>& weights, 
		       std::vector<Pattern>& patternContainer, double cutValue);

static void write (std::string fileName, const Net& net, const std::vector<double>& weights);









class Batch 
{
public:
    
    Batch (typename std::vector<Pattern>::const_iterator itBegin, typename std::vector<Pattern>::const_iterator itEnd)
	: m_itBegin (itBegin)
	, m_itEnd (itEnd)
    {}

    typename std::vector<Pattern>::const_iterator begin () const { return m_itBegin; }
    typename std::vector<Pattern>::const_iterator end   () const { return m_itEnd; }

private:
    typename std::vector<Pattern>::const_iterator m_itBegin;
    typename std::vector<Pattern>::const_iterator m_itEnd;
};





auto zero = [] (double& value) { value = 0; };



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









template <typename T>
T uniformFromTo (T from, T to)
{
    return from + (rand ()* (to - from)/RAND_MAX);
}



template <typename Container, typename T>
void uniform (Container& container, T maxValue)
{
    for (auto it = begin (container), itEnd = end (container); it != itEnd; ++it)
    {
//        (*it) = uniformFromTo (-1.0*maxValue, 1.0*maxValue);
        (*it) = uniformDouble (-1.0*maxValue, 1.0*maxValue);
    }
}

template <typename Container, typename T>
void studentT (Container& container, T distrValue)
{
    for (auto it = begin (container), itEnd = end (container); it != itEnd; ++it)
    {
        (*it) = studenttDouble (distrValue);
    }
}

template <typename Container, typename T>
void gaussDistribution (Container& container, T mean, T sigma)
{
    for (auto it = begin (container), itEnd = end (container); it != itEnd; ++it)
    {
        (*it) = gaussDouble (mean, sigma);
    }
}




// class Minimizer
// {
// public:

//     template <typename Function, typename Variables, typename PassThrough>
//     double operator() (Function& fnc, Variables& vars, PassThrough& passThrough) 
//     {
//         // auto itVars = begin (vars);
//         // auto itVarsEnd = end (vars);

//         std::vector<double> myweights;
//         std::vector<double> gradients;

//         double value = fnc (passThrough, myweights);
//         value = fnc (passThrough, myweights, gradients);
//         return value;
//     } 
// };









class Steepest
{
public:

    size_t m_repetitions;

    Steepest (double learningRate = 1e-4, double momentum = 0.5, size_t repetitions = 10) 
	: m_repetitions (repetitions)
        , m_alpha (learningRate)
        , m_beta (momentum)
    {}

    template <typename Function, typename Weights, typename PassThrough>
    double operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough) 
    {
	size_t numWeights = weights.size ();
	std::vector<NNTYPE> gradients (numWeights, 0.0);
	std::vector<NNTYPE> localWeights (begin (weights), end (weights));
        if (m_prevGradients.empty ())
            m_prevGradients.assign (weights.size (), 0);


        double Ebase = fitnessFunction (passThrough, weights, gradients);
        double Emin = Ebase;

        bool success = true;
        size_t currentRepetition = 0;
        while (success)
        {
            if (currentRepetition >= m_repetitions)
                break;

            auto itLocW = begin (localWeights);
            auto itLocWEnd = end (localWeights);
            auto itG = begin (gradients);
            auto itPrevG = begin (m_prevGradients);
            for (; itLocW != itLocWEnd; ++itLocW, ++itG, ++itPrevG)
            {
                (*itG) *= m_alpha;
                (*itG) += m_beta * (*itPrevG);
                (*itLocW) += (*itG);
                (*itPrevG) = (*itG);
            }
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


    double m_alpha;
    double m_beta;
    std::vector<double> m_prevGradients;
};










class SteepestThreaded
{
public:

    size_t m_repetitions;

    SteepestThreaded (double learningRate = 1e-4, double learningRatePrev = 1e-4, size_t repetitions = 10) 
	: m_repetitions (repetitions)
        , m_alpha (learningRate)
        , m_beta (learningRatePrev)
    {}


    template <typename Function, typename Weights, typename Gradients, typename PassThrough>
    double fitWrapper (Function& function, PassThrough& passThrough, Weights weights)
    {
	return fitnessFunction (passThrough, weights);
    }



    template <typename Function, typename Weights, typename PassThrough>
    double operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough) 
    {
	size_t numWeights = weights.size ();
	std::vector<NNTYPE> gradients (numWeights, 0.0);
	std::vector<NNTYPE> localWeights (begin (weights), end (weights));
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


    double m_alpha;
    double m_beta;
    std::vector<double> m_prevGradients;
};










class MaxGradWeight
{
public:

    size_t m_repetitions;

    MaxGradWeight (double learningRate = 1e-4, size_t repetitions = 10) 
	: m_repetitions (repetitions)
        , m_learningRate (learningRate)
    {}



    template <typename Function, typename Weights, typename PassThrough>
    double operator() (Function& fitnessFunction, const Weights& weights, PassThrough& passThrough) 
    {
	NNTYPE alpha = m_learningRate;

	size_t numWeights = weights.size ();
	std::vector<NNTYPE> gradients (numWeights, 0.0);
	std::vector<NNTYPE> localWeights (begin (weights), end (weights));


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

private:
    double m_learningRate;
};








template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
NNTYPE sumOfSquares (ItOutput itOutputBegin, ItOutput itOutputEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, NNTYPE patternWeight) 
{
    NNTYPE errorSum = 0.0;

    // output - truth
    ItTruth itTruth = itTruthBegin;
    bool hasDeltas = (itDelta != itDeltaEnd);
    for (ItOutput itOutput = itOutputBegin; itOutput != itOutputEnd; ++itOutput, ++itTruth)
    {
	assert (itTruth != itTruthEnd);
	NNTYPE output = (*itOutput);
	NNTYPE error = output - (*itTruth);
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
NNTYPE crossEntropy (ItProbability itProbabilityBegin, ItProbability itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, NNTYPE patternWeight) 
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
NNTYPE softMaxCrossEntropy (ItOutput itProbabilityBegin, ItOutput itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, NNTYPE patternWeight) 
{
    NNTYPE errorSum = 0.0;

    bool hasDeltas = (itDelta != itDeltaEnd);
    // output - truth
    ItTruth itTruth = itTruthBegin;
    for (auto itProbability = itProbabilityBegin; itProbability != itProbabilityEnd; ++itProbability, ++itTruth)
    {
	assert (itTruth != itTruthEnd);
	NNTYPE probability = (*itProbability);
	NNTYPE truth = (*itTruth);
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
NNTYPE weightDecay (NNTYPE error, ItWeight itWeight, ItWeight itWeightEnd, NNTYPE factorWeightDecay)
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




enum class ModeOutputValues
{
    DIRECT = 'd',
    SIGMOID = 's',
    SOFTMAX = 'S'
};




class LayerData
{
public:
    typedef std::vector<double> container_type;
    typedef typename container_type::iterator iterator_type;
    typedef typename container_type::const_iterator const_iterator_type;
    typedef std::vector<std::function<double(double)> > function_container_type;
    typedef typename function_container_type::iterator function_iterator_type;
    typedef typename function_container_type::const_iterator const_function_iterator_type;

    LayerData (const_iterator_type itInputBegin, const_iterator_type itInputEnd, ModeOutputValues eModeOutput = ModeOutputValues::DIRECT)
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

    ~LayerData ()    {}



    LayerData (size_t size, 
	       const_iterator_type itWeightBegin, 
	       iterator_type itGradientBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       const_function_iterator_type itInverseFunctionBegin,
	       ModeOutputValues eModeOutput = ModeOutputValues::DIRECT)
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





    LayerData (size_t size, const_iterator_type itWeightBegin, const_function_iterator_type itFunctionBegin, 
	       ModeOutputValues eModeOutput = ModeOutputValues::DIRECT)
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

    LayerData (const LayerData& other)
    : m_size (other.m_size)
    , m_itInputBegin (other.m_itInputBegin)
    , m_itInputEnd (other.m_itInputEnd)
    , m_deltas (other.m_deltas)
    , m_valueGradients (other.m_valueGradients)
    , m_values (other.m_values)
    , m_itConstWeightBegin   (other.m_itConstWeightBegin)
    , m_itGradientBegin (other.m_itGradientBegin)
    , m_itFunctionBegin (other.m_itFunctionBegin)
    , m_itInverseFunctionBegin (other.m_itInverseFunctionBegin)
    , m_isInputLayer (other.m_isInputLayer)
    , m_hasWeights (other.m_hasWeights)
    , m_hasGradients (other.m_hasGradients)
    , m_eModeOutput (other.m_eModeOutput) 
    {}

    LayerData (LayerData&& other)
    : m_size (other.m_size)
    , m_itInputBegin (other.m_itInputBegin)
    , m_itInputEnd (other.m_itInputEnd)
    , m_deltas (other.m_deltas)
    , m_valueGradients (other.m_valueGradients)
    , m_values (other.m_values)
    , m_itConstWeightBegin   (other.m_itConstWeightBegin)
    , m_itGradientBegin (other.m_itGradientBegin)
    , m_itFunctionBegin (other.m_itFunctionBegin)
    , m_itInverseFunctionBegin (other.m_itInverseFunctionBegin)
    , m_isInputLayer (other.m_isInputLayer)
    , m_hasWeights (other.m_hasWeights)
    , m_hasGradients (other.m_hasGradients)
    , m_eModeOutput (other.m_eModeOutput) 
    {}


    const_iterator_type valuesBegin () const { return m_isInputLayer ? m_itInputBegin : begin (m_values); }
    const_iterator_type valuesEnd   () const { return m_isInputLayer ? m_itInputEnd   : end   (m_values); }
    
    iterator_type valuesBegin () { assert (!m_isInputLayer); return begin (m_values); }
    iterator_type valuesEnd   () { assert (!m_isInputLayer); return end   (m_values); }

    ModeOutputValues outputMode () const { return m_eModeOutput; }
    container_type probabilities () { return computeProbabilities (); }

    iterator_type deltasBegin () { return begin (m_deltas); }
    iterator_type deltasEnd   () { return end   (m_deltas); }

    const_iterator_type deltasBegin () const { return begin (m_deltas); }
    const_iterator_type deltasEnd   () const { return end   (m_deltas); }

    iterator_type valueGradientsBegin () { return begin (m_valueGradients); }
    iterator_type valueGradientsEnd   () { return end   (m_valueGradients); }

    const_iterator_type valueGradientsBegin () const { return begin (m_valueGradients); }
    const_iterator_type valueGradientsEnd   () const { return end   (m_valueGradients); }

    iterator_type gradientsBegin () { assert (m_hasGradients); return m_itGradientBegin; }
    const_iterator_type gradientsBegin () const { assert (m_hasGradients); return m_itGradientBegin; }
    const_iterator_type weightsBegin   () const { assert (m_hasWeights); return m_itConstWeightBegin; }
    const_function_iterator_type functionBegin () const { return m_itFunctionBegin; }
    const_function_iterator_type inverseFunctionBegin () const { return m_itInverseFunctionBegin; }

    size_t size () const { return m_size; }

private:

    container_type computeProbabilities ()
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

private:
    
    size_t m_size;

    const_iterator_type m_itInputBegin;
    const_iterator_type m_itInputEnd;

    std::vector<double> m_deltas;
    std::vector<double> m_valueGradients;
    std::vector<double> m_values;

    const_iterator_type m_itConstWeightBegin;
    iterator_type       m_itGradientBegin;

    const_function_iterator_type m_itFunctionBegin;

    const_function_iterator_type m_itInverseFunctionBegin;

    bool m_isInputLayer;
    bool m_hasWeights;
    bool m_hasGradients;

    ModeOutputValues m_eModeOutput;

    friend std::ostream& operator<< (std::ostream& ostr, LayerData const& data);
};



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




class Layer
{
public:

    Layer (size_t numNodes, EnumFunction activationFunction, ModeOutputValues eModeOutputValues = ModeOutputValues::DIRECT) 
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

    ModeOutputValues modeOutputValues () const { return m_eModeOutputValues; }
    void modeOutputValues (ModeOutputValues eModeOutputValues) { m_eModeOutputValues = eModeOutputValues; }

    size_t numNodes () const { return m_numNodes; }
    size_t numWeights (size_t numInputNodes) const { return numInputNodes * numNodes (); } // fully connected

    const std::vector<std::function<double(double)> >& activationFunctions  () const { return m_vecActivationFunctions; }
    const std::vector<std::function<double(double)> >& inverseActivationFunctions  () const { return m_vecInverseActivationFunctions; }



    std::string write () const
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

    
private:


    std::vector<std::function<double(double)> > m_vecActivationFunctions;
    std::vector<std::function<double(double)> > m_vecInverseActivationFunctions;

    EnumFunction m_activationFunction;

    size_t m_numNodes;

    ModeOutputValues m_eModeOutputValues;

    friend class Net;
};



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







void forward (const LayerData& prevLayerData, LayerData& currLayerData)
{
    applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		  currLayerData.weightsBegin (), 
		  currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin ());
}

void forward_training (const LayerData& prevLayerData, LayerData& currLayerData)
{
    applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		  currLayerData.weightsBegin (), 
		  currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin (), 
		    currLayerData.inverseFunctionBegin (), currLayerData.valueGradientsBegin ());
}


void backward (LayerData& prevLayerData, LayerData& currLayerData)
{
    applyWeightsBackwards (currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			   currLayerData.weightsBegin (), 
			   prevLayerData.deltasBegin (), prevLayerData.deltasEnd ());
}




void update (const LayerData& prevLayerData, LayerData& currLayerData, double weightDecay, bool isL1)
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




class Settings
{
public:
    typedef std::map<std::string,Gnuplot*> PlotMap;
    typedef std::map<std::string,std::pair<std::vector<double>,std::vector<double> > > DataXYMap;

    Settings (size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, double _factorWeightDecay = 1e-5, bool isL1Regularization = false)
        : m_convergenceSteps (_convergenceSteps)
        , m_batchSize (_batchSize)
        , m_testRepetitions (_testRepetitions)
        , m_factorWeightDecay (_factorWeightDecay)
	, count_E (0)
	, count_dE (0)
	, count_mb_E (0)
	, count_mb_dE (0)
        , m_isL1Regularization (isL1Regularization)
    {
    }
    
    virtual ~Settings () {}

    size_t convergenceSteps () const { return m_convergenceSteps; }
    size_t batchSize () const { return m_batchSize; }
    size_t testRepetitions () const { return m_testRepetitions; }
    double factorWeightDecay () const { return m_factorWeightDecay; }


    Gnuplot* plot (std::string plotName, std::string subName, std::string dataName, std::string style = "points", std::string smoothing = "");
    void resetPlot (std::string plotName);



    void addPoint (std::string dataName, double x, double y);

    virtual void testSample (double error, double output, double target, double weight) {}

    
    virtual void startTestCycle () {}
    virtual void endTestCycle () {}
    virtual void drawSample (const std::vector<NNTYPE>& input, const std::vector<NNTYPE>& output, const std::vector<NNTYPE>& target, NNTYPE patternWeight) {}

    virtual void computeResult (const Net& net, std::vector<double>& weights) {}

    void clearData (std::string dataName);

    bool isL1 () const { return m_isL1Regularization; }

public:
    size_t m_convergenceSteps;
    size_t m_batchSize;
    size_t m_testRepetitions;
    double m_factorWeightDecay;

    size_t count_E;
    size_t count_dE;
    size_t count_mb_E;
    size_t count_mb_dE;

    bool m_isL1Regularization;

private:    
    std::pair<std::vector<double>,std::vector<double> >& getData (std::string dataName);
    Gnuplot* getPlot (std::string plotName);

    PlotMap plots;
    DataXYMap dataXY;

};





inline void Settings::clearData (std::string dataName)
{
    std::pair<std::vector<double>,std::vector<double> >& data = getData (dataName);

    std::vector<double>& vecX = data.first;
    std::vector<double>& vecY = data.second;

    vecX.clear ();
    vecY.clear ();
}

inline void Settings::addPoint (std::string dataName, double x, double y)
{
    std::pair<std::vector<double>,std::vector<double> >& data = getData (dataName);

    std::vector<double>& vecX = data.first;
    std::vector<double>& vecY = data.second;

    vecX.push_back (x);
    vecY.push_back (y);
}


inline Gnuplot* Settings::plot (std::string plotName, std::string subName, std::string dataName, std::string style, std::string smoothing)
{
    Gnuplot* pPlot = getPlot (plotName);
    std::pair<std::vector<double>,std::vector<double> >& data = getData (dataName);

    std::vector<double>& vecX = data.first;
    std::vector<double>& vecY = data.second;

    pPlot->set_style(style).set_smooth(smoothing).plot_xy(vecX,vecY,subName);
    pPlot->unset_smooth ();

    return pPlot;
}


void Settings::resetPlot (std::string plotName)
{
    Gnuplot* pPlot = getPlot (plotName);
    pPlot->reset_plot ();
}



inline Gnuplot* Settings::getPlot (std::string plotName)
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


inline std::pair<std::vector<double>,std::vector<double> >& Settings::getData (std::string dataName)
{
    DataXYMap::iterator itDataXY = dataXY.find (dataName);
    if (itDataXY == dataXY.end ())
    {
        std::pair<DataXYMap::iterator, bool> result = dataXY.insert (std::make_pair (dataName, std::make_pair(std::vector<double>(),std::vector<double>())));
        itDataXY = result.first;
    }

    return itDataXY->second;
}


















// enthaelt additional zu den settings die plot-kommandos fuer die graphischen
// ausgaben. 
class ClassificationSettings : public Settings
{
public:
    ClassificationSettings (size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
			    double _factorWeightDecay = 1e-5, bool _isL1Regularization = false, size_t _scaleToNumEvents = 0)
        : Settings (_convergenceSteps, _batchSize, _testRepetitions, _factorWeightDecay, _isL1Regularization)
        , m_ams ()
        , m_sumOfSigWeights (0)
        , m_sumOfBkgWeights (0)
	, m_scaleToNumEvents (_scaleToNumEvents)
	, m_cutValue (10.0)
	, m_pResultPatternContainer (NULL)
	, m_fileNameResult ()
	, m_fileNameNetConfig ()
    {
    }

    virtual ~ClassificationSettings () {}



    void testSample (double error, double output, double target, double weight)
    {
        m_output.push_back (output);
        m_targets.push_back (target);
        m_weights.push_back (weight);
    }


    virtual void startTestCycle () 
    {
        m_output.clear ();
        m_targets.clear ();
        m_weights.clear ();
        resetPlot ("roc");
        clearData ("datRoc");
        resetPlot ("output");
        clearData ("datOutputSig");
        clearData ("datOutputBkg");
        resetPlot ("amsSig");
        clearData ("datAms");
        clearData ("datSignificance");
    }

    virtual void endTestCycle () 
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



    void setWeightSums (double sumOfSigWeights, double sumOfBkgWeights) { m_sumOfSigWeights = sumOfSigWeights; m_sumOfBkgWeights = sumOfBkgWeights; }
    void setResultComputation (std::string _fileNameNetConfig, std::string _fileNameResult, std::vector<Pattern>* _resultPatternContainer)
    {
	m_pResultPatternContainer = _resultPatternContainer;
	m_fileNameResult = _fileNameResult;
	m_fileNameNetConfig = _fileNameNetConfig;
    }


    virtual void computeResult (const Net& net, std::vector<double>& weights) 
    {
	write (m_fileNameNetConfig, net, weights);
	if (!m_fileNameResult.empty () && m_pResultPatternContainer && !m_pResultPatternContainer->empty ())
	    writeKaggleHiggs (m_fileNameResult, net, weights, *m_pResultPatternContainer, m_cutValue);
    }


    std::vector<double> m_input;
    std::vector<double> m_output;
    std::vector<double> m_targets;
    std::vector<double> m_weights;

    std::vector<double> m_ams;
    std::vector<double> m_significances;


    double m_sumOfSigWeights;
    double m_sumOfBkgWeights;
    size_t m_scaleToNumEvents;

    double m_cutValue;
    std::vector<Pattern>* m_pResultPatternContainer;
    std::string m_fileNameResult;
    std::string m_fileNameNetConfig;
};








enum class ModeOutput
{
    FETCH
};


enum class ModeErrorFunction
{
    SUMOFSQUARES = 'S',
	CROSSENTROPY = 'C',
    CROSSENTROPY_MUTUALEXCLUSIVE = 'M'
};




class Net
{
public:

    typedef std::vector<double> container_type;
    typedef container_type::iterator iterator_type;
    typedef std::pair<iterator_type,iterator_type> begin_end_type;


    Net () 
	: m_eErrorFunction (ModeErrorFunction::SUMOFSQUARES)
    {
    }

    void addLayer (Layer& layer) { m_layers.push_back (layer); }
    void addLayer (Layer&& layer) { m_layers.push_back (layer); }
    void setErrorFunction (ModeErrorFunction eErrorFunction) { m_eErrorFunction = eErrorFunction; }
    


    template <typename Minimizer>
    double train (std::vector<double>& weights, std::vector<Pattern>& trainPattern, const std::vector<Pattern>& testPattern, 
                  Minimizer& minimizer, Settings& settings)
    {
        std::cout << "START TRAINING" << std::endl;
        size_t convergenceCount = 0;
        NNTYPE minError = 1e10;

        size_t cycleCount = 0;
        size_t testCycleCount = 0;
        NNTYPE testError = 1e20;

        // until convergence
        while (convergenceCount < settings.convergenceSteps ())
        {
            std::cout << "train cycle " << cycleCount << std::endl;
            ++cycleCount;
            std::random_shuffle (begin (trainPattern), end (trainPattern));
            NNTYPE trainError = trainCycle (minimizer, weights, begin (trainPattern), end (trainPattern), settings);
//	std::cout << "test cycle" << std::endl;

            if (testCycleCount % settings.testRepetitions () == 0)
            {
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
                    std::tuple<Settings&, Batch&> passThrough (settings, batch);
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


            static double x = -1.0;
            x += 1.0;
            settings.resetPlot ("errors");
            settings.addPoint ("trainErrors", cycleCount, trainError);
            settings.addPoint ("testErrors", cycleCount, testError);
            settings.plot ("errors", "training", "trainErrors", "points", "");
            settings.plot ("errors", "training_", "trainErrors", "lines", "cspline");
            settings.plot ("errors", "test", "testErrors", "points", "");
            settings.plot ("errors", "test_", "testErrors", "lines", "cspline");



//	(*this).print (std::cout);
            std::cout << "check convergence; minError " << minError << "  current " << testError << "  current convergence count " << convergenceCount << std::endl;
            if (testError < minError)
            {
                convergenceCount = 0;
                minError = testError;
            }
            else
                ++convergenceCount;

            std::cout << "testError : " << testError << "   trainError : " << trainError << std::endl;
        }
        std::cout << "END TRAINING" << std::endl;
        return testError;
    }



    template <typename Iterator, typename Minimizer>
    inline NNTYPE trainCycle (Minimizer& minimizer, std::vector<double>& weights, 
			      Iterator itPatternBegin, Iterator itPatternEnd, Settings& settings)
    {
	NNTYPE error = 0.0;
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
            std::tuple<Settings&, Batch&> settingsAndBatch (settings, batch);
	    error += minimizer ((*this), weights, settingsAndBatch);
	    itPatternBatchBegin = itPatternBatchEnd;
	    --numBatches;
	}
	if (itPatternBatchEnd != itPatternEnd)
        {
            Batch batch (itPatternBatchEnd, itPatternEnd);
            std::tuple<Settings&, Batch&> settingsAndBatch (settings, batch);
	    error += minimizer ((*this), weights, settingsAndBatch);
        }
	error /= numBatches_stored;
    
	return error;
    }






    size_t numWeights (size_t numInputNodes, size_t trainingStartLayer = 0) const 
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
    std::vector<double> operator() (const std::vector<double>& input, const Weights& weights) const
    {
	std::vector<LayerData> layerData;
	layerData.reserve (m_layers.size ()+1);
	auto itWeight = begin (weights);
	auto itInputBegin = begin (input);
	auto itInputEnd = end (input);
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
    double operator() (PassThrough& settingsAndBatch, const Weights& weights)
    {
	std::vector<double> nothing; // empty gradients; no backpropagation is done, just forward
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (nothing), std::end (nothing), 100, nothing, false);
        return error;
    }

    template <typename Weights, typename PassThrough, typename OutContainer>
    double operator() (PassThrough& settingsAndBatch, const Weights& weights, ModeOutput eFetch, OutContainer& outputContainer)
    {
	std::vector<double> nothing; // empty gradients; no backpropagation is done, just forward
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (nothing), std::end (nothing), 100, outputContainer, true);
        return error;
    }

    
    template <typename Weights, typename Gradients, typename PassThrough>
    double operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients)
    {
        std::vector<double> nothing;
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (gradients), std::end (gradients), 0, nothing, false);
        return error;
    }

    template <typename Weights, typename Gradients, typename PassThrough, typename OutContainer>
    double operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients, ModeOutput eFetch, OutContainer& outputContainer)
    {
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (gradients), std::end (gradients), 0, outputContainer, true);
        return error;
    }





    template <typename LayerContainer, typename PassThrough, typename ItWeight, typename ItGradient, typename OutContainer>
    double forward_backward (LayerContainer& layers, PassThrough& settingsAndBatch, ItWeight itWeightBegin, ItGradient itGradientBegin, ItGradient itGradientEnd, size_t trainFromLayer, OutContainer& outputContainer, bool fetchOutput)
    {
        Batch& batch = std::get<1>(settingsAndBatch);
        Settings& settings = std::get<0>(settingsAndBatch);

	if (layers.empty ())
	    throw std::string ("no layers in this net");

	double sumError = 0.0;
	double sumWeights = 0.0;

	// -------------
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
//            std::cout << layerData.back () << std::endl;
	    size_t numNodesPrev = pattern.input ().size ();
	    for (auto& layer: layers)
	    {
                if (itGradientBegin == itGradientEnd)
                    layerData.push_back (LayerData (layer.numNodes (), itWeight, 
                                                    begin (layer.activationFunctions ()),
						    layer.modeOutputValues ()));
                else
                    layerData.push_back (LayerData (layer.numNodes (), itWeight, itGradient, 
                                                    begin (layer.activationFunctions ()), begin (layer.inverseActivationFunctions ()),
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


	sumError /= sumWeights;
	return sumError;
    }



    
    double E ();
    void dE ();


    template <typename Container, typename ItWeight>
    NNTYPE errorFunction (LayerData& layerData, Container truth, ItWeight itWeight, ItWeight itWeightEnd, double patternWeight, double factorWeightDecay)
    {
	NNTYPE error (0);
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

    template <typename ItOutput, typename ItTruth, typename ItWeight>
    NNTYPE errorFunction (ItOutput itOutputBegin, ItOutput itOutputEnd, 
			  ItTruth itTruthBegin, ItTruth itTruthEnd, 
			  ItWeight itWeight, ItWeight itWeightEnd, 
			  NNTYPE patternWeight, double factorWeightDecay) 
    {
	NNTYPE error (0);
	switch (m_eErrorFunction)
	{
	case ModeErrorFunction::SUMOFSQUARES:
	    error = sumOfSquares (itOutputBegin, itOutputEnd, itTruthBegin, itTruthEnd, (int*)NULL, (int*)NULL, (std::function<double(double)>*)NULL, patternWeight);
	    break;
	case ModeErrorFunction::CROSSENTROPY:
	    error = sumOfSquares (itOutputBegin, itOutputEnd, itTruthBegin, itTruthEnd, (int*)NULL, (int*)NULL, (std::function<double(double)>*)NULL, patternWeight);
	    break;
	}
	if (factorWeightDecay != 0)
	    error = weightDecay (error, itWeight, itWeightEnd, factorWeightDecay);
	return error;
    } 

    const std::vector<Layer>& layers () const { return m_layers; }
    std::vector<Layer>& layers ()  { return m_layers; }


    std::ostream& write (std::ostream& ostr) const
    {
	ostr << "===NET===" << std::endl;
	ostr << "ERRORFUNCTION=" << (char)m_eErrorFunction << std::endl;
	for (const Layer& layer: m_layers)
	{
	    ostr << layer.write () << std::endl;
	}
	return ostr;
    }

private:

    std::vector<Layer> m_layers;
    ModeErrorFunction m_eErrorFunction;

    friend std::ostream& operator<< (std::ostream& ostr, Net const& net);
};




void writeKaggleHiggs (std::string fileName, const Net& net, const std::vector<double>& weights, 
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
	
	value = net (pattern.input (), weights).at (0); 
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







static void write (std::string fileName, const Net& net, const std::vector<double>& weights) 
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









void checkGradients ()
{
    Net net;

    size_t inputSize = 1;
    size_t outputSize = 1;


    net.addLayer (Layer (30, EnumFunction::SOFTSIGN)); 
    net.addLayer (Layer (30, EnumFunction::SOFTSIGN)); 
//    net.addLayer (Layer (outputSize, EnumFunction::LINEAR)); 
    net.addLayer (Layer (outputSize, EnumFunction::LINEAR, ModeOutputValues::SIGMOID)); 
    net.setErrorFunction (ModeErrorFunction::CROSSENTROPY);
//    net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);
    //weights.at (0) = 1000213.2;

    std::vector<Pattern> pattern;
    for (size_t iPat = 0, iPatEnd = 10; iPat < iPatEnd; ++iPat)
    {
        std::vector<double> input;
        std::vector<double> output;
        for (size_t i = 0; i < inputSize; ++i)
        {
            input.push_back (uniformDouble (-1.5, 1.5));
        }
        for (size_t i = 0; i < outputSize; ++i)
        {
            output.push_back (uniformDouble (-1.5, 1.5));
        }
        pattern.push_back (Pattern (input,output));
    }


    Settings settings (/*_convergenceSteps*/ 15, /*_batchSize*/ 1, /*_testRepetitions*/ 7, /*_factorWeightDecay*/ 0, /*isL1*/ false);

    size_t improvements = 0;
    size_t worsenings = 0;
    size_t smallDifferences = 0;
    size_t largeDifferences = 0;
    for (size_t iTest = 0; iTest < 1000; ++iTest)
    {
        uniform (weights, 0.7);
        std::vector<double> gradients (numWeights, 0);
        Batch batch (begin (pattern), end (pattern));
        std::tuple<Settings&, Batch&> settingsAndBatch (settings, batch);
        double E = net (settingsAndBatch, weights, gradients);
        std::vector<double> changedWeights;
        changedWeights.assign (weights.begin (), weights.end ());

        int changeWeightPosition = randomInt (numWeights);
        double dEdw = gradients.at (changeWeightPosition);
        while (dEdw == 0.0)
        {
            changeWeightPosition = randomInt (numWeights);
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


    Net net;

    size_t inputSize = 2;
    size_t outputSize = 1;

    net.addLayer (Layer (4, EnumFunction::TANH)); 
    net.addLayer (Layer (outputSize, EnumFunction::LINEAR)); 

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);

    std::vector<Pattern> patterns;
    patterns.push_back (Pattern ({0, 0}, {0}));
    patterns.push_back (Pattern ({1, 1}, {0}));
    patterns.push_back (Pattern ({1, 0}, {1}));
    patterns.push_back (Pattern ({0, 1}, {1}));

    uniform (weights, 0.7);
    
//    StochasticCG minimizer;
    Steepest minimizer (/*learningRate*/ 1e-5);
    Settings settings (/*_convergenceSteps*/ 50, /*_batchSize*/ 4, /*_testRepetitions*/ 7);
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


    Net net;

    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.addLayer (Layer (10, EnumFunction::TANH)); 
    net.addLayer (Layer (outputSize, EnumFunction::LINEAR, ModeOutputValues::SIGMOID)); 
//    net.addLayer (Layer (outputSize, EnumFunction::LINEAR, ModeOutputValues::DIRECT)); 
    net.setErrorFunction (ModeErrorFunction::CROSSENTROPY);
//    net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);

    uniform (weights, 0.2);
    
//    Steepest minimizer (1e-6, 0.0, 3);
    SteepestThreaded minimizer (1e-4, 1e-4, 6);
//    MaxGradWeight minimizer;
    ClassificationSettings settings (/*_convergenceSteps*/ 150, /*_batchSize*/ 30, /*_testRepetitions*/ 7, /*factorWeightDecay*/ 1.0e-5);
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


    Net net;

    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.addLayer (Layer (3, EnumFunction::TANH)); 
    net.addLayer (Layer (outputSize, EnumFunction::LINEAR)); 

    size_t numWeights = net.numWeights (inputSize);
    std::vector<double> weights (numWeights);

    uniform (weights, 0.2);
    
    Steepest minimizer;
    ClassificationSettings settings (/*_convergenceSteps*/ 2, /*_batchSize*/ 30, /*_testRepetitions*/ 7, /*factorWeightDecay*/ 1.0e-5);
    //double E = net.train (weights, trainPattern, testPattern, minimizer, settings);


    std::cout << "BEFORE" << std::endl;
    std::cout << net << std::endl;
    std::cout << "WEIGHTS BEFORE" << std::endl;
    for_each (begin (weights), end (weights), [](double w){ std::cout << w << " "; });
    std::cout << std::endl << std::endl;

    // writing
    write ("testfile.nn", net, weights);

    // reading
    Net readNet;
    std::vector<double> readWeights;
    std::tie (readNet, readWeights) = read ("testfile.nn");


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


    std::string filenameTrain ("/home/peter/code/kaggle_Higgs/training.csv");
    std::string filenameTest ("/home/peter/code/kaggle_Higgs/training.csv");
    std::string filenameSubmission ("/home/peter/code/kaggle_Higgs/test.csv");

    // std::string filenameTrain ("/home/developer/test/kaggle_Higgs/training.csv");
    // std::string filenameTest ("/home/developer/test/kaggle_Higgs/training.csv");
    // std::string filenameSubmission ("/home/developer/test/kaggle_Higgs/test.csv");

    std::vector<std::string> fieldNamesTrain; 
    std::vector<std::string> fieldNamesTest; 
    size_t skipTrain = 0;
    size_t numberTrain = 20000;
    size_t skipTest  =  20000;
    size_t numberTest  =  20000;
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
    Net net;
    std::vector<double> weights;

#if false // read from saved file
    std::tie (net, weights) = read ("higgs.net");

    // net.layers ().back ().modeOutputValues (ModeOutputValues::DIRECT); 
    // net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);
    
#else
    size_t inputSize = trainPattern.front ().input ().size ();
    size_t outputSize = trainPattern.front ().output ().size ();

    net.addLayer (Layer (40, EnumFunction::SYMMRELU)); 
    net.addLayer (Layer (30, EnumFunction::SYMMRELU)); 
    net.addLayer (Layer (20, EnumFunction::SYMMRELU)); 
    net.addLayer (Layer (outputSize, EnumFunction::LINEAR, ModeOutputValues::SIGMOID)); 
    net.setErrorFunction (ModeErrorFunction::CROSSENTROPY);

//    size_t numWeightsFirstLayer = net.layers ().front ().numWeights (inputSize);

    size_t numWeights = net.numWeights (inputSize);
    weights.resize (numWeights, 0.0);
//    uniform (weights, 0.2);
//    studentT (weights, 1.0/sqrt(inputSize));
//    studentT (weights, 10.0);
    gaussDistribution (weights, 0.1, 1.0/sqrt(inputSize));

#endif
    
//    StochasticCG minimizer;
    Steepest minimizer (1e-4, 0.3, 3);
//    SteepestThreaded minimizer (1e-3, 1e-3, 6);
//    MaxGradWeight minimizer (1e-5, 3);
    ClassificationSettings settings (/*_convergenceSteps*/ 150, /*_batchSize*/ 70, /*_testRepetitions*/ 7, 
				     /*factorWeightDecay*/ 0.0, /*isL1*/true, /*scaleToNumEvents*/ 550000);
    settings.setWeightSums (sumOfSigWeights_test, sumOfBkgWeights_test);
    settings.setResultComputation ("higgs.net", "submission.csv", &submissionPattern);
    /*double E = */net.train (weights, trainPattern, testPattern, minimizer, settings);


}





int main ()
{ 
//    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    feenableexcept (FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); // exceptions bei underflow, overflow und divide by zero (damit man den fehler gleich findet)

//    checkGradients ();
//    testXOR ();
    Higgs ();
//    testClassification ();
//    testWriteRead ();

    wait_for_key();
    return 0;
} 


