#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <tuple>
#include <math.h>
#include <cassert>
#include <random>
#include <thread>
#include <future>

#include "Pattern.h"
#include "Monitoring.h"

#include "TApplication.h"

#include <fenv.h> // turn on or off exceptions for NaN and other numeric exceptions



namespace TMVA
{

namespace NN
{

    double gaussDouble (double mean, double sigma);








enum class EnumFunction
{
    ZERO = '0',
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

std::function<double(double)> ZeroFnc = [](double /*value*/){ return 0; };


std::function<double(double)> Sigmoid = [](double value){ value = std::max (-100.0, std::min (100.0,value)); return 1.0/(1.0 + std::exp (-value)); };
std::function<double(double)> InvSigmoid = [](double value){ double s = Sigmoid (value); return s*(1.0-s); };

std::function<double(double)> Tanh = [](double value){ return tanh (value); };
std::function<double(double)> InvTanh = [](double value){ return 1.0 - std::pow (value, 2.0); };

std::function<double(double)> Linear = [](double value){ return value; };
std::function<double(double)>  InvLinear = [](double /*value*/){ return 1.0; };

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







class Net;


static void write (std::string fileName, const Net& net, const std::vector<double>& weights);





typedef std::vector<char> DropContainer;


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






template <typename ItSource, typename ItWeight, typename ItTarget>
    void applyWeights (ItSource itSourceBegin, ItSource itSourceEnd, ItWeight itWeight, ItTarget itTargetBegin, ItTarget itTargetEnd);



template <typename ItSource, typename ItWeight, typename ItPrev>
    void applyWeightsBackwards (ItSource itCurrBegin, ItSource itCurrEnd, ItWeight itWeight, ItPrev itPrevBegin, ItPrev itPrevEnd);





template <typename ItValue, typename ItFunction>
    void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction);


template <typename ItValue, typename ItFunction, typename ItInverseFunction, typename ItGradient>
    void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction, ItInverseFunction itInverseFunction, ItGradient itGradient);



template <typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient);



template <bool isL1, typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient, typename ItWeight>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient, 
	     ItWeight itWeight, double weightDecay);



// ----- signature of a minimizer -------------
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
        double operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough);


    double m_alpha;
    double m_beta;
    std::vector<double> m_prevGradients;
};









// test multithreaded training
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
        double fitWrapper (Function& function, PassThrough& passThrough, Weights weights);


    template <typename Function, typename Weights, typename PassThrough>
        double operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough);


    double m_alpha;
    double m_beta;
    std::vector<double> m_prevGradients;
};









// walk along the maximum gradient
class MaxGradWeight
{
public:

    size_t m_repetitions;

    MaxGradWeight (double learningRate = 1e-4, size_t repetitions = 10) 
	: m_repetitions (repetitions)
        , m_learningRate (learningRate)
    {}



    template <typename Function, typename Weights, typename PassThrough>
        double operator() (Function& fitnessFunction, const Weights& weights, PassThrough& passThrough);

private:
    double m_learningRate;
};








template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
    double sumOfSquares (ItOutput itOutputBegin, ItOutput itOutputEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight);



template <typename ItProbability, typename ItTruth, typename ItDelta, typename ItInvActFnc>
    double crossEntropy (ItProbability itProbabilityBegin, ItProbability itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight);




template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
    double softMaxCrossEntropy (ItOutput itProbabilityBegin, ItOutput itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight);





template <typename ItWeight>
    double weightDecay (double error, ItWeight itWeight, ItWeight itWeightEnd, double factorWeightDecay);



enum class ModeOutputValues
{
    DIRECT = 'd',
    SIGMOID = 's',
    SOFTMAX = 'S'
};











// the actual data for the layer (not the layout)
class LayerData
{
public:
    typedef std::vector<double> container_type;

    typedef typename container_type::iterator iterator_type;
    typedef typename container_type::const_iterator const_iterator_type;

    typedef std::vector<std::function<double(double)> > function_container_type;
    typedef typename function_container_type::iterator function_iterator_type;
    typedef typename function_container_type::const_iterator const_function_iterator_type;

    LayerData (const_iterator_type itInputBegin, const_iterator_type itInputEnd, ModeOutputValues eModeOutput = ModeOutputValues::DIRECT);


    ~LayerData ()    {}


    LayerData (size_t size, 
	       const_iterator_type itWeightBegin, 
	       iterator_type itGradientBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       const_function_iterator_type itInverseFunctionBegin,
	       ModeOutputValues eModeOutput = ModeOutputValues::DIRECT);

    LayerData (size_t size, const_iterator_type itWeightBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       ModeOutputValues eModeOutput = ModeOutputValues::DIRECT);

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
    const_iterator_type valuesEnd   () const { return m_isInputLayer ? m_itInputEnd   : end (m_values); }
    
    iterator_type valuesBegin () { assert (!m_isInputLayer); return begin (m_values); }
    iterator_type valuesEnd   () { assert (!m_isInputLayer); return end (m_values); }

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

    container_type computeProbabilities ();

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



std::ostream& operator<< (std::ostream& ostr, LayerData const& data);


// defines the layout of a layer
class Layer
{
public:

    Layer (size_t numNodes, EnumFunction activationFunction, ModeOutputValues eModeOutputValues = ModeOutputValues::DIRECT);

    ModeOutputValues modeOutputValues () const { return m_eModeOutputValues; }
    void modeOutputValues (ModeOutputValues eModeOutputValues) { m_eModeOutputValues = eModeOutputValues; }

    size_t numNodes () const { return m_numNodes; }
    size_t numWeights (size_t numInputNodes) const { return numInputNodes * numNodes (); } // fully connected

    const std::vector<std::function<double(double)> >& activationFunctions  () const { return m_vecActivationFunctions; }
    const std::vector<std::function<double(double)> >& inverseActivationFunctions  () const { return m_vecInverseActivationFunctions; }



    std::string write () const;
    
private:


    std::vector<std::function<double(double)> > m_vecActivationFunctions;
    std::vector<std::function<double(double)> > m_vecInverseActivationFunctions;

    EnumFunction m_activationFunction;

    size_t m_numNodes;

    ModeOutputValues m_eModeOutputValues;

    friend class Net;
};



static Layer readLayer (std::istream& ss);


template <typename LAYERDATA>
    void forward (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData);

template <typename LAYERDATA>
    void forward_training (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData);


template <typename LAYERDATA>
    void backward (LAYERDATA& prevLayerData, LAYERDATA& currLayerData);


template <typename LAYERDATA>
    void update (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData, double weightDecay, bool isL1);



class Settings
{
public:
//    typedef std::map<std::string,Gnuplot*> PlotMap;
    typedef std::map<std::string,std::pair<std::vector<double>,std::vector<double> > > DataXYMap;

    Settings (size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
	      double _factorWeightDecay = 1e-5, bool isL1Regularization = false, double dropFraction = 0.0,
	      size_t dropRepetitions = 7);
    
    virtual ~Settings ();

    void SetMonitoring (std::shared_ptr<Monitoring> ptrMonitoring) { fMonitoring = ptrMonitoring; }

    size_t convergenceSteps () const { return m_convergenceSteps; }
    size_t batchSize () const { return m_batchSize; }
    size_t testRepetitions () const { return m_testRepetitions; }
    double factorWeightDecay () const { return m_factorWeightDecay; }

    size_t dropRepetitions () const { return m_dropRepetitions; }
    double dropFraction () const { return m_dropFraction; }

//    Gnuplot* plot (std::string plotName, std::string subName, std::string dataName, std::string style = "points", std::string smoothing = "");
    virtual void resetPlot (std::string plotName);
    virtual void plot ();



    void addPoint (std::string dataName, double x, double y);

    virtual void testSample (double /*error*/, double /*output*/, double /*target*/, double /*weight*/) {}

    
    virtual void startTestCycle () {}
    virtual void endTestCycle () {}
    virtual void drawSample (const std::vector<double>& /*input*/, const std::vector<double>& /* output */, const std::vector<double>& /* target */, double /* patternWeight */) {}

    virtual void computeResult (const Net& /* net */, std::vector<double>& /* weights */) {}

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

    double m_dropFraction;
    double m_dropRepetitions;

private:    
    std::pair<std::vector<double>,std::vector<double> >& getData (std::string dataName);
//    Gnuplot* getPlot (std::string plotName);

//    PlotMap plots;
    DataXYMap dataXY;
    std::shared_ptr<Monitoring> fMonitoring;
};























// enthaelt additional zu den settings die plot-kommandos fuer die graphischen
// ausgaben. 
class ClassificationSettings : public Settings
{
public:
    ClassificationSettings (size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
			    double _factorWeightDecay = 1e-5, bool _isL1Regularization = false, 
			    double _dropFraction = 0.0, size_t _dropRepetitions = 7,
			    size_t _scaleToNumEvents = 0)
        : Settings (_convergenceSteps, _batchSize, _testRepetitions, _factorWeightDecay, _isL1Regularization, _dropFraction, _dropRepetitions)
        , m_ams ()
        , m_sumOfSigWeights (0)
        , m_sumOfBkgWeights (0)
	, m_scaleToNumEvents (_scaleToNumEvents)
	, m_cutValue (10.0)
	, m_pResultPatternContainer (NULL)
	, m_fileNameResult ()
	, m_fileNameNetConfig ()
    {
        int argc = 0;
        char* txt = "";
        char **argv = &txt;
        m_application = new TApplication ("my app", &argc, argv);
        m_application->SetReturnFromRun (true);
    }

    virtual ~ClassificationSettings () 
    {
        delete m_application;
    }



    void testSample (double error, double output, double target, double weight);

    virtual void startTestCycle ();

    virtual void endTestCycle ();


    void setWeightSums (double sumOfSigWeights, double sumOfBkgWeights);
    void setResultComputation (std::string _fileNameNetConfig, std::string _fileNameResult, std::vector<Pattern>* _resultPatternContainer);

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

    TApplication* m_application;
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
    

    template <typename WeightsType>
        void dropOutWeightFactor (const DropContainer& dropContainer, WeightsType& weights, double factor);

    template <typename Minimizer>
    double train (std::vector<double>& weights, 
		  std::vector<Pattern>& trainPattern, 
		  const std::vector<Pattern>& testPattern, 
                  Minimizer& minimizer, Settings& settings);


    template <typename Iterator, typename Minimizer>
    inline double trainCycle (Minimizer& minimizer, std::vector<double>& weights, 
			      Iterator itPatternBegin, Iterator itPatternEnd, Settings& settings, DropContainer& dropContainer);

    size_t numWeights (size_t numInputNodes, size_t trainingStartLayer = 0) const;

    template <typename Weights>
        std::vector<double> compute (const std::vector<double>& input, const Weights& weights) const;

    template <typename Weights, typename PassThrough>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights) const;

    template <typename Weights, typename PassThrough, typename OutContainer>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights, ModeOutput eFetch, OutContainer& outputContainer) const;
    
    template <typename Weights, typename Gradients, typename PassThrough>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients) const;

    template <typename Weights, typename Gradients, typename PassThrough, typename OutContainer>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients, ModeOutput eFetch, OutContainer& outputContainer) const;




    template <typename LayerContainer, typename PassThrough, typename ItWeight, typename ItGradient, typename OutContainer>
    double forward_backward (LayerContainer& layers, PassThrough& settingsAndBatch, 
			     ItWeight itWeightBegin, 
			     ItGradient itGradientBegin, ItGradient itGradientEnd, 
			     size_t trainFromLayer, 
			     OutContainer& outputContainer, bool fetchOutput) const;


    
    double E ();
    void dE ();


    template <typename Container, typename ItWeight>
        double errorFunction (LayerData& layerData, Container truth, ItWeight itWeight, ItWeight itWeightEnd, double patternWeight, double factorWeightDecay) const;


    const std::vector<Layer>& layers () const { return m_layers; }
    std::vector<Layer>& layers ()  { return m_layers; }


    std::ostream& write (std::ostream& ostr) const;

    void clear () 
    {
        m_layers.clear ();
	m_eErrorFunction = ModeErrorFunction::SUMOFSQUARES;
    }

private:

    std::vector<Layer> m_layers;
    ModeErrorFunction m_eErrorFunction;

    friend std::ostream& operator<< (std::ostream& ostr, Net const& net);
};








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




}; // namespace NN
}; // namespace TMVA


// include the implementations (in header file, because they are templated)
#include "NeuralNet_i.h"
