#ifndef _NEURALNET_HPP_
#define _NEURALNET_HPP_

#include <vector>
#include <map>
#include <cmath>
#include <iostream> 
#include <iterator> 
#include <cstddef> 

#include "../readpng/readpng.h"
#include "../pattern/pattern.hpp"

#define PRINT(TARGET) 
//#define PRINT(TARGET) std::cout<<TARGET;
//#define ENABLE_PRINT 1

#define NNTYPE double

NNTYPE randomNumber (NNTYPE from, NNTYPE to);
int randomInt (int maxVal);






enum EnumFunction
{
    eSigmoid = 0x1,
    eLinear = eSigmoid << 1,
    eSoftSign = eSigmoid << 2,
    eTanh = eSigmoid << 3,
    eGauss = eSigmoid << 4,
    eGaussComplement = eSigmoid << 5,
    eStep = eSigmoid << 6,
    eDoubleInvertedGauss = eSigmoid << 7,
    eTanhShift = eSigmoid << 8,
    ePeak = eSigmoid << 9,

    eVarying = eTanh
//    eVarying = eSoftSign | eTanh | eGauss | eGaussComplement
//    eVarying = eSoftSign | eTanh | eDoubleInvertedGauss
//    eVarying = eGauss | eGaussComplement
};


enum EnumMode
{
    e_E = 0x1,
    e_dE = e_E << 1,
    e_shift = e_E << 2
};



class TanhShift 
{
public:
    const NNTYPE shift = 0.5;

    virtual inline NNTYPE apply (NNTYPE value) const { return tanh (value-shift); }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return 1.0 - std::pow (value-shift, 2.0); }
};

class Tanh 
{
public:
    virtual inline NNTYPE apply (NNTYPE value) const { return tanh (value); }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return 1.0 - std::pow (value, 2.0); }
};


class SoftSign
{
public:
    virtual inline NNTYPE apply (NNTYPE value) const { return value / (1.0 + fabs (value)); }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return std::pow ((1.0 - fabs (value)),2.0); }
};

class Linear
{
public:
    virtual inline NNTYPE apply (NNTYPE value) const { return value; }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return 1.0; }
};

class Gauss
{
public:
    const double s = 6.0;

    virtual inline NNTYPE apply (NNTYPE value) const { return exp (-std::pow(value*s,2.0)); }
//    virtual inline NNTYPE applyInverse (NNTYPE value) const { return 1.0/(1.0+std::pow(s*value,2.0)); }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return -2.0 * value * s*s * apply (value); }
};

class Peak
{
public:
    const double s = 0.2;

    virtual inline NNTYPE apply (NNTYPE value) const 
    {   
        if (value < -s)
            return 0;
        if (value > s)
            return 0;
        if (value <=0)
            return (value/s) +1.0;
        return -value/s + 1.0;
    }
    virtual inline NNTYPE applyInverse (NNTYPE value) const 
    { 
        if (value < -s)
            return 0;
        if (value > s)
            return 0;
        if (value <=0)
            return 1.0/s;
        return -1.0/s;
    }
};


class GaussComplement
{
public:
    const double s = 6.0;

    virtual inline NNTYPE apply (NNTYPE value) const { return 1.0 - exp (-std::pow(value*s,2.0)); }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return +2.0 * value * s*s * apply (value); }
//    virtual inline NNTYPE applyInverse (NNTYPE value) const { return -1.0/(1.0+std::pow(s*value,2.0)); }
};


class DoubleInvertedGauss
{
public:
    const double s = 8.0;
    const double shift = 0.1;

    virtual inline NNTYPE apply (NNTYPE value) const 
    { 
        return exp (-std::pow((value-shift)*s,2.0)) - exp (-std::pow((value+shift)*s,2.0)); 
    }

    virtual inline NNTYPE applyInverse (NNTYPE value) const 
    { 
        return -2.0 * (value-shift) * s*s * apply (value-shift) + 2.0 * (value+shift) * s*s * apply (value+shift); 
    }
};



class Step
{
public:
    const double position = 0.3;

    virtual inline NNTYPE apply (NNTYPE value) const { return value > position ? 0.0 : value < -position ? 1.0 : 0; }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return 0.0; }
};

class Bias
{
public:
    virtual inline NNTYPE apply (NNTYPE value) const { return 1.0; }
    virtual inline NNTYPE applyInverse (NNTYPE value) const { return 0.0; }
};


class Gnuplot;




struct Settings
{
    typedef std::map<std::string,Gnuplot*> PlotMap;
    typedef std::map<std::string,std::pair<std::vector<double>,std::vector<double> > > DataXYMap;


    Settings ()
        : weightDecay (0)
        , repetitions (50)
        , maxRepetitions (100)
        , maxInitWeight (0.3)
        , testRepetitions (0)
	, count_E (0)
	, count_dE (0)
	, count_mb_E (0)
	, count_mb_dE (0)
    {}

    double weightDecay;
    size_t repetitions;
    size_t maxRepetitions;
    double maxInitWeight;
    size_t testRepetitions;

    size_t count_E;
    size_t count_dE;
    size_t count_mb_E;
    size_t count_mb_dE;


    Gnuplot* plot (std::string plotName, std::string subName, std::string dataName, std::string style = "points", std::string smoothing = "");
    void resetPlot (std::string plotName);

    void addPoint (std::string dataName, double x, double y);


    
    template <typename NodeIterator, typename OutIterator>
    void testSampleComputed (NNTYPE error, NodeIterator beginInput, NodeIterator endInput, 
                             NodeIterator beginOutput, NodeIterator endOutput,
                             OutIterator itOutputBegin, OutIterator itOutputEnd, NNTYPE weight)
    {
        std::vector<NNTYPE> input;
        for (NodeIterator it = beginInput; it != endInput; ++it)
        {
            input.push_back ((*it)->value);
        }
        std::vector<NNTYPE> output;
        for (NodeIterator it = beginOutput; it != endOutput; ++it)
        {
            output.push_back ((*it)->value);
        }
        std::vector<NNTYPE> target (itOutputBegin, itOutputEnd);

        drawSample (input, output, target, weight);
    }
    
    virtual void startTestCycle () {}
    virtual void endTestCycle () {}
    virtual void drawSample (const std::vector<NNTYPE>& input, const std::vector<NNTYPE>& output, const std::vector<NNTYPE>& target, NNTYPE patternWeight) {}

    void clearData (std::string dataName);

private:    
    std::pair<std::vector<double>,std::vector<double> >& getData (std::string dataName);
    Gnuplot* getPlot (std::string plotName);

    PlotMap plots;
    DataXYMap dataXY;
};




class INode
{
public:
    INode () : value (0), dValue (0), delta (0) {}

    virtual NNTYPE applyActivationFunction (NNTYPE value) const = 0;
    virtual NNTYPE applyInverseActivationFunction (NNTYPE value) const = 0;

    NNTYPE value;
    NNTYPE dValue;
    NNTYPE delta;
};


template <typename ActivationFunction>
class Node : public INode
{
public:
    virtual inline NNTYPE applyActivationFunction (NNTYPE value) const { return m_activationFunction.apply (value); }
    virtual inline NNTYPE applyInverseActivationFunction (NNTYPE value) const { return m_activationFunction.applyInverse (value); }

private:
    ActivationFunction m_activationFunction;
};


class Synapsis
{
public:
    Synapsis (INode* _start, NNTYPE _weight, INode* _end) 
        : start (_start)
        , end (_end)
        , weight (_weight)
    {}

    Synapsis (const Synapsis& other) 
    {
        start = other.start;
        end = other.end;
        weight = other.weight;
    }

    Synapsis& operator= (const Synapsis& other) 
    {
    start = other.start;
	end = other.end;
	weight = other.weight;
	return *this;
    }



    INode* start;
    INode* end;

    NNTYPE weight;
    NNTYPE dE;
    NNTYPE dE_shifted;

    NNTYPE p;
    NNTYPE r;
    NNTYPE r1;
};




enum EnumDensity
{
    eFull, 
    eSparse
};




template <typename T>
struct TypeSpecifier
{
    typedef T type;
};


class SynapsisLayer;
std::ostream& operator<< (std::ostream& ostr, const SynapsisLayer& syn);



class SynapsisLayer
{
public:
    typedef std::vector<INode*> node_container;
    typedef node_container::iterator node_iterator;
    typedef std::vector<Synapsis> synapsis_container;
    typedef typename synapsis_container::iterator synapsis_iterator;
    typedef typename synapsis_container::const_iterator const_synapsis_iterator;

    SynapsisLayer (Settings* settings);

    // setters
    void addStartNode (INode* node);
    void addEndNode (INode* node);
    void addSynapsis (Synapsis synapsis);

    // 
    void connect (size_t startSynapsisLayer, size_t endSynapsisLayer);

    // getters
//    std::vector<Node*>& startNodes ();
//    std::vector<Node*>& endNodes ();
    synapsis_container& synapses ();
    
    synapsis_iterator beginSynapses () { return  m_synapses.begin (); }
    synapsis_iterator endSynapses () { return  m_synapses.end (); }
    const_synapsis_iterator beginSynapses () const { return  m_synapses.begin (); }
    const_synapsis_iterator endSynapses () const { return  m_synapses.end (); }

    node_iterator beginStartNodes ();
    node_iterator endStartNodes   ();
    node_iterator beginEndNodes ();
    node_iterator endEndNodes   ();

    void clear (EnumMode eClear);

    void compute_E (NNTYPE alpha = 0.0);
    void compute_dE (NNTYPE sigma = 0.0);
    void update_dE (bool shifted);

    EnumFunction function ();

    std::ostream& print (std::ostream& ostr)
    {
	#ifdef ENABLE_PRINT
	for (auto it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
	{
	    Synapsis& syn = (*it);
	    ostr << " |W: " << syn.weight << " dE: " << syn.dE;
	}
	ostr << std::endl;
	#endif
	return ostr;
    }


private:

    node_container     m_startNodes;
    node_container     m_endNodes;
    synapsis_container m_synapses;

    EnumFunction m_eFunction;

    Settings* m_pSettings;

    friend std::ostream& operator<< (std::ostream&, const SynapsisLayer&);
};



class Net;


class SynapsisIterator
{
public:  
    typedef Synapsis value_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef Synapsis* pointer;
    typedef Synapsis& reference;

    typedef std::map<size_t, SynapsisLayer> synapsis_layer_map;
    typedef typename synapsis_layer_map::iterator synapsis_layer_iterator;
    typedef typename synapsis_layer_map::reverse_iterator reverse_synapsis_layer_iterator;

    typedef SynapsisLayer::synapsis_iterator synapsis_iterator;

    SynapsisIterator (Net& net, bool isEnd = false);
    SynapsisIterator (const SynapsisIterator& other);
    SynapsisIterator& operator= (const SynapsisIterator& other);

    bool operator== (const SynapsisIterator& other) const;
    bool operator!= (const SynapsisIterator& other) const;
    Synapsis& operator* () { return (*m_itSynapsis); }
    Synapsis* operator->() { return &(*m_itSynapsis); }

    SynapsisIterator& operator++ ();
    SynapsisIterator operator++ (int);


private:
    Net& m_net;
    synapsis_layer_iterator m_itLayer;
    synapsis_layer_iterator m_itLayerEnd;
    synapsis_iterator m_itSynapsis;
    synapsis_iterator m_itSynapsisEnd;
    bool m_isEndIterator;
};


class Net
{
public:
    typedef std::vector<INode*> node_layer;
    typedef std::map<size_t, node_layer> node_layer_map;
    typedef typename node_layer::iterator node_layer_iterator;
    typedef typename node_layer::iterator node_iterator;

    typedef std::map<size_t, SynapsisLayer> synapsis_layer_map;
    typedef typename synapsis_layer_map::iterator synapsis_layer_iterator;
    typedef typename synapsis_layer_map::reverse_iterator reverse_synapsis_layer_iterator;

    typedef SynapsisIterator synapsis_iterator;

    Net (Settings* settings); 
    void addNode (size_t layer, INode* node);
    node_iterator nodesBegin (size_t layer);
    node_iterator nodesEnd   (size_t layer);

    void connect (size_t startSynapsisLayer, size_t endSynapsisLayer);
    SynapsisLayer* getSynapsisLayer (size_t layer); 
    void removeSynapsisLayer (size_t layer);

    template <typename ItPattern>
    NNTYPE computeBatch (ItPattern beginPattern, ItPattern endPattern, EnumMode eMode, NNTYPE shift_E = 0.0, NNTYPE shift_dE = 0.0);


    template <typename ItOut>
    NNTYPE errorFunction (ItOut beginOutput, ItOut endOutput, NNTYPE patternWeight);



    void compute_E (NNTYPE shift = 0.0);
    void compute_dE (NNTYPE shift = 0.0);
    void update_dE (bool isShifted = false);
    void clear (EnumMode eClear);

    template <typename Iterator, typename OutIterator>
    void compute (Iterator begin, Iterator end, OutIterator itOut);


    template <typename ItPattern>
    void trainBatch_BP    (ItPattern beginBatch, ItPattern endBatch); 

    template <typename ItPattern>
    void trainBatch_LBFGS (ItPattern beginBatch, ItPattern endBatch);

    template <typename ItPattern>
    NNTYPE trainBatch_SCG   (ItPattern beginBatch, ItPattern endBatch);

    size_t layerCount () const { return m_synapsisLayers.size (); }
    node_iterator beginNodes (size_t layer);
    node_iterator endNodes   (size_t layer);

    synapsis_layer_iterator beginSynapsisLayers ();
    synapsis_layer_iterator endSynapsisLayers ();

    synapsis_iterator begin () { return synapsis_iterator (*this); }
    synapsis_iterator end   () { return synapsis_iterator (*this, true); }
    

    std::ostream& print (std::ostream& ostr)
    {
	#ifdef ENABLE_PRINT
	ostr << "NN" << std::endl;
	int layer = 0;
	for (auto it = beginSynapsisLayers (), itEnd = endSynapsisLayers (); it != itEnd; ++it)
	{
	    ostr << "Nodes : ";
	    for (auto itN = beginNodes (layer), itNEnd = endNodes (layer); itN != itNEnd; ++itN)
	    {
		INode* node = (*itN);
//		ostr << " |" << node << " V: " << node->value << " D: " << node->delta;
		ostr << " |V: " << node->value << " D: " << node->delta;
	    }
	    ostr << std::endl;
	    ostr << "SynapsisLayer " << (*it).first << " : ";
	    (*it).second.print (ostr);
	    ++layer;
	}
	ostr << "Nodes : ";
	for (auto itN = beginNodes (layer), itNEnd = endNodes (layer); itN != itNEnd; ++itN)
	{
	    INode* node = (*itN);
//	    ostr << " |" << node << " V: " << node->value << " D: " << node->delta;
	    ostr << " |V: " << node->value << " D: " << node->delta;
	}
	#endif
	return ostr;
    }


private:

    node_layer_map        m_nodes;
    synapsis_layer_map    m_synapsisLayers;

    size_t m_count_E;
    size_t m_count_dE;
    size_t m_count_E_mb;
    size_t m_count_dE_mb;

    Settings* m_pSettings;
};

















class NeuralNet
{
public:

    typedef typename Net::node_layer::iterator node_iterator;

    NeuralNet (Settings* settings) 
    : m_net (settings) 
    , m_pSettings (settings)
    {}


    template <typename Iterator>
    void mlp (Iterator itNodeNumbersBegin, Iterator itNodeNumbersEnd, EnumFunction eActivationFunction, EnumFunction eOutputLayerActivationFunction);

    template <typename Iterator, typename IteratorOut>
    NNTYPE testSample (Iterator itInput, Iterator itInputEnd, IteratorOut itOutputBegin, IteratorOut itOutputEnd, NNTYPE weight = 1.0);

    template <typename Iterator>
    void calculateSample (Iterator itInput, Iterator itInputEnd);


    template <typename Iterator>
    void training_BP (size_t startSynapsisLayer, size_t endSynapsisLayer, 
		   Iterator itTrainingsPatternBegin, Iterator itTrainingsPatternEnd, 
		   Iterator itTestPatternBegin, Iterator itTestPatternEnd,
		   int convergence, NNTYPE learningRate, NNTYPE learningDecrease, NNTYPE momentum, size_t batchSize = 1);


    template <typename Iterator>
    void training_LBFGS (size_t startSynapsisLayer, size_t endSynapsisLayer, 
			 Iterator itTrainingsPatternBegin, Iterator itTrainingsPatternEnd, Iterator itTestPatternBegin, Iterator itTestPatternEnd, int convergence, int memory, NNTYPE maxStep, NNTYPE damping, NNTYPE alpha, size_t batchSize);

    template <typename Iterator>
    void training_SCG (Iterator itTrainingsPatternBegin, Iterator itTrainingsPatternEnd, 
                       Iterator itTestPatternBegin, Iterator itTestPatternEnd, size_t batchSize, size_t convergenceSteps);


    template <typename Iterator>
    NNTYPE trainCycles_LBFGS (size_t startSynapsisLayer, size_t endSynapsisLayer, 
			      Iterator itPatternBegin, Iterator itPatternEnd, 
			      size_t numberCycles, int memory, NNTYPE maxStep, NNTYPE damping, NNTYPE alpha,
			      size_t batchSize = 1);


    node_iterator beginNodes (size_t layer);
    node_iterator endNodes (size_t layer);
    size_t outputLayerIndex () const { return m_net.layerCount (); }
    

    std::ostream& print (std::ostream& ostr)
    {
	#ifdef ENABLE_PRINT
	ostr << "NN" << std::endl;
	int layer = 0;
	for (auto it = m_net.beginSynapsisLayers (), itEnd = m_net.endSynapsisLayers (); it != itEnd; ++it)
	{
	    ostr << "Nodes : ";
	    for (auto itN = beginNodes (layer), itNEnd = endNodes (layer); itN != itNEnd; ++itN)
	    {
		INode* node = (*itN);
//		ostr << " |" << node << " V: " << node->value << " D: " << node->delta;
		ostr << " |V: " << node->value << " D: " << node->delta;
	    }
	    ostr << std::endl;
	    ostr << "SynapsisLayer " << (*it).first << " : ";
	    (*it).second.print (ostr);
	    ++layer;
	}
	ostr << "Nodes : ";
	for (auto itN = beginNodes (layer), itNEnd = endNodes (layer); itN != itNEnd; ++itN)
	{
	    INode* node = (*itN);
//	    ostr << " |" << node << " V: " << node->value << " D: " << node->delta;
	    ostr << " |V: " << node->value << " D: " << node->delta;
	}
	#endif
	return ostr;
    }


    bool checkGradient (NNTYPE& E, NNTYPE& E_delta, NNTYPE& E_delta_gradient, const Pattern& pattern, NNTYPE delta, NNTYPE limit = 0.1);


private:

    template <typename Iterator>
    NNTYPE trainCycles_BP (size_t startSynapsisLayer, size_t endSynapsisLayer, 
			Iterator itPatternBegin, Iterator itPatternEnd, 
			size_t numberCycles,
			NNTYPE learningRate, NNTYPE momentum, size_t batchSize = 1);

    template <typename Iterator>
    NNTYPE trainCycles_SCG (Iterator itPatternBegin, Iterator itPatternEnd, 
			    size_t batchSize = 10);

    template <typename Iterator>
    NNTYPE trainCycles_LBFGS (size_t startSynapsisLayer, size_t endSynapsisLayer, 
			      Iterator itPatternBegin, Iterator itPatternEnd, 
			      size_t numberCycles,
			      int convergence, int memory, NNTYPE maxStep, NNTYPE damping, NNTYPE alpha, size_t batchSize = 1);


    template <typename Iterator>
    NNTYPE testCycle (Iterator itPatternBegin, Iterator itPatternEnd);



    

    


private:

    Net m_net;
    Settings* m_pSettings;
};



#include "neuralNet_i.hpp"

#endif

