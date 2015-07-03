#ifndef __NEURAL_NET__H
#define __NEURAL_NET__H
#pragma once


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
#include <map>


class Gnuplot;



#include <fenv.h>

#include "../pattern/pattern.hpp"



namespace NN
{

    double gaussDouble (double mean, double sigma);
    double studenttDouble (double distributionParameter);
    int randomInt (int maxValue);
    double uniformDouble (double minValue, double maxValue);
    
template <typename Container, typename T>
    void uniform (Container& container, T maxValue);

template <typename Container, typename T>
    void gaussDistribution (Container& container, T mean, T sigma);





/* template <typename Iterator> */
/*     std::pair<double, double> meanVariance (Iterator begin, Iterator end) */
/* { */
/*     double sum = std::accumulate (begin, end, 0.0); */
/*     double mean = sum/std::distance (begin, end); */
    
/*     std::vector<double> diff (v.size ()); */
/*     std::transform (begin, end, diff.begin (), std::bind2nd (std::minux<double>(), mean)); */
/*     double sq_sum = std::inner_product (diff.begin (), diff.end (), diff.begin (), 0.0); */
/*     double variance = sq_sum/v.size (); */

/*     return std::make_pair (mean, variance); */
/* } */


template <typename Iterator>
    std::pair<double, double> computeMeanVariance (Iterator begin, Iterator end)
{
    double mean = 0;
    double M2 = 0;
    double variance = 0.0;

    size_t n = std::distance (begin, end);
    if (n == 0)
        return std::make_pair (0.0,0.0);
    if (n == 1)
        return std::make_pair (*begin,0.0);
    for (Iterator it = begin; it != end; ++it)
    {
        double value = (*it);
        double delta = value - mean;
        mean += delta/n;
        M2 += delta*(value - mean);
        variance = M2/(n - 1);
    }

    return std::make_pair(mean,variance);
}











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



enum class EnumRegularization
{
    NONE, L1, L2, L1MAX
};


enum class ModeOutputValues
{
    DIRECT = 'd',
    SIGMOID = 's',
    SOFTMAX = 'S'
};




class Monitoring
{
public:
    struct PlotData
    {
        PlotData (bool _useErr) : useErr (_useErr) {}
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> err;
        bool useErr;
        
        void clear () { x.clear (); y.clear (); err.clear (); }
    };


    typedef std::map<std::string,Gnuplot*> PlotMap;
    typedef std::map<std::string,PlotData> DataMap;

    virtual ~Monitoring ();

    Gnuplot* plot (std::string plotName, std::string subName, std::string dataName, std::string style = "points", std::string smoothing = "");
    void resetPlot (std::string plotName);

    void addPoint (std::string dataName, double x, double y);
    void addPoint (std::string dataName, double x, double y, double err);
    void clearData (std::string dataName);

private:    
    DataMap::mapped_type& getData (std::string dataName);
    Gnuplot* getPlot (std::string plotName);

    PlotMap plots;
    DataMap data;
};



class Net;







typedef std::vector<char> DropContainer;


class Batch 
{
public:
    typedef typename std::vector<Pattern>::const_iterator const_iterator;

    Batch (typename std::vector<Pattern>::const_iterator itBegin, typename std::vector<Pattern>::const_iterator itEnd)
	: m_itBegin (itBegin)
	, m_itEnd (itEnd)
    {}

    const_iterator begin () const { return m_itBegin; }
    const_iterator end   () const { return m_itEnd; }

private:
    const_iterator m_itBegin;
    const_iterator m_itEnd;
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



template <EnumRegularization Regularization, typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient, typename ItWeight>
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





class MinimizerMonitoring
{
public:
    MinimizerMonitoring (Monitoring* pMonitoring = NULL, std::vector<size_t> layerSizes = std::vector<size_t> ());


    // plotting
    Gnuplot* plot (std::string plotName, std::string subName, std::string dataName, std::string style = "points", std::string smoothing = "");
    void resetPlot (std::string plotName);
    void addPoint (std::string dataName, double x, double y);
    void addPoint (std::string dataName, double x, double y, double err);
    void clearData (std::string dataName);

    // plot gradient means and variances
    template <typename Gradients>
    void plotGradients (const Gradients& gradients);


    // plot weight means and variances
    template <typename Weights>
    void plotWeights (const Weights& weights);


private:
    Monitoring* m_pMonitoring;
    std::vector<size_t> m_layerSizes;

    double m_xGrad;
    double m_xWeights;

    size_t m_countGrad;
    size_t m_countWeights;
};



class Steepest : public MinimizerMonitoring
{
public:

    size_t m_repetitions;

    Steepest (double learningRate = 1e-4, 
              double momentum = 0.5, 
              size_t repetitions = 10, 
              Monitoring* pMonitoring = NULL, 
              std::vector<size_t> layerSizes = std::vector<size_t> ());

    template <typename Function, typename Weights, typename PassThrough>
        double operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough);


    double m_alpha;
    double m_beta;
    std::vector<double> m_prevGradients;

    

private:
    Monitoring* m_pMonitoring;
    std::vector<size_t> m_layerSizes;
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
    double weightDecay (double error, ItWeight itWeight, ItWeight itWeightEnd, double factorWeightDecay, EnumRegularization eRegularization);














// the actual data for the layer (not the layout)
class LayerData
{
public:
    typedef std::vector<double> container_type;

    typedef container_type::iterator iterator_type;
    typedef container_type::const_iterator const_iterator_type;

    typedef std::vector<std::function<double(double)> > function_container_type;
    typedef function_container_type::iterator function_iterator_type;
    typedef function_container_type::const_iterator const_function_iterator_type;

    typedef DropContainer::const_iterator const_dropout_iterator;
    
    LayerData (const_iterator_type itInputBegin, const_iterator_type itInputEnd, ModeOutputValues eModeOutput = ModeOutputValues::DIRECT);


    LayerData  (size_t inputSize);
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
    , m_hasDropOut (false)
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
    , m_hasDropOut (false)
    , m_itConstWeightBegin   (other.m_itConstWeightBegin)
    , m_itGradientBegin (other.m_itGradientBegin)
    , m_itFunctionBegin (other.m_itFunctionBegin)
    , m_itInverseFunctionBegin (other.m_itInverseFunctionBegin)
    , m_isInputLayer (other.m_isInputLayer)
    , m_hasWeights (other.m_hasWeights)
    , m_hasGradients (other.m_hasGradients)
    , m_eModeOutput (other.m_eModeOutput) 
    {}


    void setInput (const_iterator_type itInputBegin, const_iterator_type itInputEnd)
    {
        m_isInputLayer = true;
        m_itInputBegin = itInputBegin;
        m_itInputEnd = itInputEnd;
    }

    void clear ()
    {
        m_values.assign (m_values.size (), 0.0);
        m_deltas.assign (m_deltas.size (), 0.0);
    }

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

    template <typename Iterator>
        void setDropOut (Iterator itDrop) { m_itDropOut = itDrop; m_hasDropOut = true; }
    void clearDropOut () { m_hasDropOut = false; }
    
    bool hasDropOut () const { return m_hasDropOut; }
    const_dropout_iterator dropOut () const { return m_itDropOut; }
    
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
    const_dropout_iterator m_itDropOut; // correlates with m_values
    bool m_hasDropOut;

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



//static Layer readLayer (std::istream& ss);


template <typename LAYERDATA>
    void forward (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData);

template <typename LAYERDATA>
    void forward_training (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData);


template <typename LAYERDATA>
    void backward (LAYERDATA& prevLayerData, LAYERDATA& currLayerData);


template <typename LAYERDATA>
    void update (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData, double weightDecay, EnumRegularization regularization);



class Settings
{
public:
    Settings (size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
	      double _factorWeightDecay = 1e-5, NN::EnumRegularization _regularization = NN::EnumRegularization::NONE,
              bool _multithreading = true, Monitoring* pMonitoring = NULL);

    template <typename Iterator>
        void setDropOut (Iterator begin, Iterator end, size_t _dropRepetitions) { m_dropOut.assign (begin, end); m_dropRepetitions = _dropRepetitions; }

    size_t dropRepetitions () const { return m_dropRepetitions; }
    const std::vector<double>& dropFractions () const { return m_dropOut; }

    size_t convergenceSteps () const { return m_convergenceSteps; }
    size_t batchSize () const { return m_batchSize; }
    size_t testRepetitions () const { return m_testRepetitions; }
    double factorWeightDecay () const { return m_factorWeightDecay; }


    Gnuplot* plot (std::string plotName, std::string subName, std::string dataName, std::string style = "points", std::string smoothing = "");
    void resetPlot (std::string plotName);



    void addPoint (std::string dataName, double x, double y);

    virtual void testSample (double error, double output, double target, double weight) {}
    virtual void startTrainCycle () {}
    virtual void endTrainCycle (double /*error*/) {}

    
    virtual void startTestCycle () {}
    virtual void endTestCycle () {}
    virtual void drawSample (const std::vector<double>& input, const std::vector<double>& output, const std::vector<double>& target, double patternWeight) {}

    virtual void computeResult (const Net& net, std::vector<double>& weights) {}

    void clearData (std::string dataName);

    EnumRegularization regularization () const { return m_regularization; }

    bool useMultithreading () const { return m_useMultithreading; }
    
public:
    size_t m_convergenceSteps;
    size_t m_batchSize;
    size_t m_testRepetitions;
    double m_factorWeightDecay;

    size_t count_E;
    size_t count_dE;
    size_t count_mb_E;
    size_t count_mb_dE;

    EnumRegularization m_regularization;

    double m_dropRepetitions;
    std::vector<double> m_dropOut;

private:

    bool m_useMultithreading;
    Monitoring*   m_pMonitoring;

};























// enthaelt additional zu den settings die plot-kommandos fuer die graphischen
// ausgaben. 
class ClassificationSettings : public Settings
{
public:
    ClassificationSettings (size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
			    double _factorWeightDecay = 1e-5, EnumRegularization _regularization = EnumRegularization::NONE,
                            size_t _scaleToNumEvents = 0, bool _useMultithreading = true, Monitoring* pMonitoring = NULL)
        : Settings (_convergenceSteps, _batchSize, _testRepetitions, _factorWeightDecay, _regularization, _useMultithreading, pMonitoring)
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

enum class WeightInitializationStrategy
{
    XAVIER, TEST, LAYERSIZE, XAVIERUNIFORM
};



class Net
{
public:

    typedef std::vector<double> container_type;
    typedef container_type::iterator iterator_type;
    typedef std::pair<iterator_type,iterator_type> begin_end_type;


    Net () 
	: m_eErrorFunction (ModeErrorFunction::SUMOFSQUARES)
	, m_sizeInput (0)
        , m_layers ()
    {
    }

    Net (const Net& other)
        : m_eErrorFunction (other.m_eErrorFunction)
        , m_sizeInput (other.m_sizeInput)
        , m_layers (other.m_layers)
    {
    }

    void setInputSize (size_t sizeInput) { m_sizeInput = sizeInput; }
    void setOutputSize (size_t sizeOutput) { m_sizeOutput = sizeOutput; }
    void addLayer (Layer& layer) { m_layers.push_back (layer); }
    void addLayer (Layer&& layer) { m_layers.push_back (layer); }
    void setErrorFunction (ModeErrorFunction eErrorFunction) { m_eErrorFunction = eErrorFunction; }
    
    size_t inputSize () const { return m_sizeInput; }
    size_t outputSize () const { return m_sizeOutput; }

    template <typename WeightsType, typename DropProbabilities>
        void dropOutWeightFactor (WeightsType& weights,
                                  const DropProbabilities& drops, 
                                  bool inverse = false);

    template <typename Minimizer>
    double train (std::vector<double>& weights, 
		  std::vector<Pattern>& trainPattern, 
		  const std::vector<Pattern>& testPattern, 
                  Minimizer& minimizer, Settings& settings);


    template <typename Iterator, typename Minimizer>
    inline double trainCycle (Minimizer& minimizer, std::vector<double>& weights, 
			      Iterator itPatternBegin, Iterator itPatternEnd, Settings& settings, DropContainer& dropContainer);

    size_t numWeights (size_t trainingStartLayer = 0) const;

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
        double errorFunction (LayerData& layerData,
                              Container truth,
                              ItWeight itWeight,
                              ItWeight itWeightEnd,
                              double patternWeight,
                              double factorWeightDecay,
                              EnumRegularization eRegularization) const;


    const std::vector<Layer>& layers () const { return m_layers; }
    std::vector<Layer>& layers ()  { return m_layers; }



    void clear () 
    {
        m_layers.clear ();
	m_eErrorFunction = ModeErrorFunction::SUMOFSQUARES;
    }


    template <typename ItPat, typename OutIterator>
    void initializeWeights (WeightInitializationStrategy eInitStrategy, 
			    ItPat itPatternBegin, 
			    ItPat itPatternEnd, 
			    OutIterator itWeight);


    std::ostream& write (std::ostream& ostr) const;

private:

    ModeErrorFunction m_eErrorFunction;
    size_t m_sizeInput;
    size_t m_sizeOutput;
    std::vector<Layer> m_layers;

    friend std::ostream& operator<< (std::ostream& ostr, Net const& net);
};








std::ostream& operator<< (std::ostream& ostr, Net const& net);

std::istream& read (std::istream& istr, Net& net);
void write (std::string fileName, const Net& net, const std::vector<double>& weights);
std::tuple<Net, std::vector<double>> read (std::string fileName);




}; // namespace NN



#include "neuralNet_i.h"


#endif


