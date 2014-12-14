
#include "neuralNet.hpp"
#include "../gnuplotWrapper/gnuplot_i.hpp" //Gnuplot class handles POSIX-Pipe-communikation with Gnuplot

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif
#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

NNTYPE randomNumber (NNTYPE from, NNTYPE to)
{
    return from + (rand ()* (to - from)/RAND_MAX);
}

int randomInt (int maxVal)
{
    return rand () % maxVal;
}




Gnuplot* Settings::getPlot (std::string plotName)
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


std::pair<std::vector<double>,std::vector<double> >& Settings::getData (std::string dataName)
{
    DataXYMap::iterator itDataXY = dataXY.find (dataName);
    if (itDataXY == dataXY.end ())
    {
        std::pair<DataXYMap::iterator, bool> result = dataXY.insert (std::make_pair (dataName, std::make_pair(std::vector<double>(),std::vector<double>())));
        itDataXY = result.first;
    }

    return itDataXY->second;
}


void Settings::clearData (std::string dataName)
{
    std::pair<std::vector<double>,std::vector<double> >& data = getData (dataName);

    std::vector<double>& vecX = data.first;
    std::vector<double>& vecY = data.second;

    vecX.clear ();
    vecY.clear ();
}

void Settings::addPoint (std::string dataName, double x, double y)
{
    std::pair<std::vector<double>,std::vector<double> >& data = getData (dataName);

    std::vector<double>& vecX = data.first;
    std::vector<double>& vecY = data.second;

    vecX.push_back (x);
    vecY.push_back (y);
}


Gnuplot* Settings::plot (std::string plotName, std::string subName, std::string dataName, std::string style, std::string smoothing)
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








inline std::ostream& operator<< (std::ostream& ostr, const SynapsisLayer& syn) 
{
    for (SynapsisLayer::const_synapsis_iterator it = syn.beginSynapses (), itEnd = syn.endSynapses (); it != itEnd; ++it)
    {
        ostr << " | " << (*it).weight << " dE: " << (*it).dE;
    }
    ostr << std::endl;
    return ostr;
}



SynapsisIterator::SynapsisIterator (Net& net, bool isEnd)
    : m_net (net)
    , m_itLayer (isEnd ? net.endSynapsisLayers () : net.beginSynapsisLayers ())
    , m_itLayerEnd (net.endSynapsisLayers ())
    , m_isEndIterator (isEnd)
{
    if (m_itLayer == m_itLayerEnd)
        return;
    
    m_itSynapsis = (*m_itLayer).second.beginSynapses ();
    m_itSynapsisEnd = (*m_itLayer).second.endSynapses ();
}

SynapsisIterator::SynapsisIterator (const SynapsisIterator& other)
    : m_net (other.m_net)
    , m_itLayer (other.m_itLayer)
    , m_itLayerEnd (other.m_itLayerEnd)
    , m_itSynapsis (other.m_itSynapsis)
    , m_itSynapsisEnd (other.m_itSynapsisEnd)
    , m_isEndIterator (other.m_isEndIterator)
{}

SynapsisIterator& SynapsisIterator::operator= (const SynapsisIterator& other)
{
    m_net = other.m_net;
    m_itLayer = other.m_itLayer;
    m_itLayerEnd = other.m_itLayerEnd;
    m_itSynapsis = other.m_itSynapsis;
    m_itSynapsisEnd = other.m_itSynapsisEnd;
    m_isEndIterator = other.m_isEndIterator;
    return *this;
}

SynapsisIterator& SynapsisIterator::operator++ ()
{
    ++m_itSynapsis;
    if (m_itSynapsis != m_itSynapsisEnd)
    return *this;
    
    ++m_itLayer;
    if (m_itLayer == m_itLayerEnd)
    {
        m_isEndIterator= true;
	m_itSynapsis = m_itSynapsisEnd;
	return *this;
    }

    m_itSynapsis = (*m_itLayer).second.beginSynapses ();
    m_itSynapsisEnd = (*m_itLayer).second.endSynapses ();
    return *this;
}
    
SynapsisIterator SynapsisIterator::operator++ (int)
{
    SynapsisIterator temp = *this;
    ++(*this);
    return temp;
}

bool SynapsisIterator::operator== (const SynapsisIterator& other) const 
{ 
    return m_itLayer == other.m_itLayer && (m_isEndIterator || (m_itSynapsis == other.m_itSynapsis)); 
}

bool SynapsisIterator::operator!= (const SynapsisIterator& other) const 
{ 
    return !((*this)==other); 
}



SynapsisLayer::SynapsisLayer (Settings* settings)
    : m_pSettings (settings)
{
}

// setters
void SynapsisLayer::addStartNode (INode* node) { m_startNodes.push_back (node); }
void SynapsisLayer::addEndNode (INode* node) { m_endNodes.push_back (node); }
void SynapsisLayer::addSynapsis (Synapsis synapsis) { m_synapses.push_back (synapsis); }

// 

void SynapsisLayer::connect (size_t startSynapsisLayer, size_t endSynapsisLayer)
{
    for (node_iterator itStart = beginStartNodes (), itStart_end = endStartNodes (); itStart != itStart_end; ++itStart)
    {
        INode* startNode = (*itStart);
        for (node_iterator itEnd = beginEndNodes (), itEnd_end = endEndNodes (); itEnd != itEnd_end; ++itEnd)
        {
            INode* endNode = (*itEnd);
            
            NNTYPE maxInitWeight = m_pSettings->maxInitWeight;
            NNTYPE weight = randomNumber (-1.0*maxInitWeight, 1.0*maxInitWeight);

            addSynapsis (Synapsis (startNode, weight, endNode));
        }
    }
}

// getters
//    std::vector<Node*>& SynapsisLayer::startNodes () { return m_startNodes; }
//    std::vector<Node*>& SynapsisLayer::endNodes () { return m_endNodes; }
    std::vector<Synapsis >& SynapsisLayer::synapses () { return m_synapses; }

    typename SynapsisLayer::node_iterator SynapsisLayer::beginStartNodes () { return m_startNodes.begin (); }
    typename SynapsisLayer::node_iterator SynapsisLayer::endStartNodes   () { return m_startNodes.end   (); }
    typename SynapsisLayer::node_iterator SynapsisLayer::beginEndNodes () { return m_endNodes.begin (); }
    typename SynapsisLayer::node_iterator SynapsisLayer::endEndNodes   () { return m_endNodes.end   (); }


void SynapsisLayer::compute_E (NNTYPE alpha)
{
    // delete end node values
    for (node_iterator it = beginEndNodes (), itEnd = endEndNodes (); it != itEnd; ++it)
    {
        INode* node = (*it);
        node->value = 0.0;
    }
    
    if (alpha == 0.0)
    {
        for (synapsis_iterator it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
        {
            Synapsis& syn = *it;
            syn.end->value += syn.start->value * syn.weight;
        }
    }
    else
    {
        for (synapsis_iterator it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
        {
            Synapsis& syn = *it;
            syn.end->value += syn.start->value * (syn.weight + alpha * syn.p);
        }
    }

    for (node_iterator it = beginEndNodes (), itEnd = endEndNodes (); it != itEnd; ++it)
    {
        INode* node = (*it);
        NNTYPE& value = node->value;
        value = node->applyActivationFunction (value);
        NNTYPE& dValue = node->dValue;
        dValue = node->applyInverseActivationFunction (value);
    }
}



void SynapsisLayer::compute_dE (NNTYPE sigma)
{
    // delete start node deltas
    for (node_iterator it = beginStartNodes (), itEnd = endStartNodes (); it != itEnd; ++it)
    {
        INode* node = (*it);
        node->delta = 0.0;
    }

    if (sigma == 0.0)
    {
        for (synapsis_iterator it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
        {
            Synapsis& syn = *it;
            syn.start->delta += syn.end->delta * syn.weight;
        }
    }
    else
    {
        for (synapsis_iterator it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
        {
            Synapsis& syn = *it;
            syn.start->delta += syn.end->delta * (syn.weight + sigma * syn.p);
        }
    }
}

void SynapsisLayer::update_dE (bool shifted)
{
    for (synapsis_iterator it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
    {
	Synapsis& syn = *it;
        NNTYPE& dE = (shifted ? syn.dE_shifted : syn.dE);
	dE += -syn.end->delta * syn.start->value * syn.end->dValue + m_pSettings->weightDecay * syn.weight;
    }
}






void SynapsisLayer::clear (EnumMode eClear)
{
    bool clear_dE = (eClear&e_dE)!=0 && (eClear&e_shift)==0;
    bool clear_dE_shifted = (eClear&e_dE)!=0 && (eClear&e_shift)!=0;

    if (clear_dE)
    {
        for (auto it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
        {
            Synapsis& syn = *it;
            syn.dE = 0;
        }
    }
    if (clear_dE_shifted)
    {
        for (auto it = beginSynapses (), itEnd = endSynapses (); it != itEnd; ++it)
        {
            Synapsis& syn = *it;
            syn.dE_shifted = 0;
        }
    }
}





Net::Net (Settings* settings) 
    : m_pSettings (settings)
{
}
    
    
void Net::addNode (size_t layer, INode* node) 
{ 
    node_layer_map::iterator itSynapsisLayer = m_nodes.find (layer);
    if (itSynapsisLayer == m_nodes.end ())
	itSynapsisLayer = m_nodes.insert (typename node_layer_map::value_type (layer, node_layer ())).first;

    itSynapsisLayer->second.push_back (node);
}

    
typename Net::node_iterator Net::beginNodes (size_t layer)   { return m_nodes.find (layer)->second.begin (); }
    
typename Net::node_iterator Net::endNodes   (size_t layer)   { return m_nodes.find (layer)->second.end (); }

    
void Net::connect (size_t startSynapsisLayer, size_t endSynapsisLayer)
{
    std::pair<synapsis_layer_iterator, bool> insertResult = m_synapsisLayers.insert (std::make_pair(endSynapsisLayer, SynapsisLayer (m_pSettings)));
    SynapsisLayer& layer = insertResult.first->second;

    for (node_iterator it = beginNodes (startSynapsisLayer), itEnd = endNodes (startSynapsisLayer); it != itEnd; ++it)
    {
	INode* node = (*it);
	layer.addStartNode (node);
    }
    for (node_iterator it = beginNodes (endSynapsisLayer), itEnd = endNodes (endSynapsisLayer); it != itEnd; ++it)
    {
	INode* node = (*it);
	layer.addEndNode (node);
    }
    layer.connect (startSynapsisLayer, endSynapsisLayer);
}


SynapsisLayer* Net::getSynapsisLayer (size_t layer) 
{ 
    synapsis_layer_iterator it = m_synapsisLayers.find (layer); 
    if (it == m_synapsisLayers.end ())
	return NULL;

    return &it->second;
}

    
void Net::removeSynapsisLayer (size_t layer)
{
    synapsis_layer_iterator it = m_synapsisLayers.find (layer); 
    if (it != m_synapsisLayers.end ())
	m_synapsisLayers.erase (it);
}

    
void Net::compute_E (NNTYPE shift)
{
    if (m_nodes.empty ())
	return;

    if (m_synapsisLayers.empty ())
	return;

    for (synapsis_layer_iterator it = beginSynapsisLayers (), itEnd = endSynapsisLayers (); it != itEnd; ++it)
    {
	SynapsisLayer& layer = it->second;
	layer.compute_E (shift); // propagate
    }
    ++m_pSettings->count_E;
}


    
void Net::compute_dE (NNTYPE shift)
{
    if (m_nodes.empty ())
	return;

    if (m_synapsisLayers.empty ())
	return;


    for (reverse_synapsis_layer_iterator it = m_synapsisLayers.rbegin (), itEnd = m_synapsisLayers.rend (); it != itEnd; ++it)
    {
	SynapsisLayer& layer = it->second;
	layer.compute_dE (shift); // backprop
    }
    ++m_pSettings->count_dE;
}

void Net::update_dE (bool isShifted)
{
    if (m_nodes.empty ())
	return;

    if (m_synapsisLayers.empty ())
	return;


    for (reverse_synapsis_layer_iterator it = m_synapsisLayers.rbegin (), itEnd = m_synapsisLayers.rend (); it != itEnd; ++it)
    {
	SynapsisLayer& layer = it->second;
	layer.update_dE (isShifted); 
    }
}


void Net::clear (EnumMode eClear)
{
    for (auto it = beginSynapsisLayers (), itEnd = endSynapsisLayers (); it != itEnd; ++it)
    {
	SynapsisLayer& layer = it->second;
	layer.clear (eClear);
    }
}








    

    

typename Net::synapsis_layer_iterator Net::beginSynapsisLayers () { return m_synapsisLayers.begin (); }
    
typename Net::synapsis_layer_iterator Net::endSynapsisLayers ()   { return m_synapsisLayers.end (); }














typename NeuralNet::node_iterator NeuralNet::beginNodes (size_t layer) { return m_net.beginNodes (layer); }

typename NeuralNet::node_iterator NeuralNet::endNodes (size_t layer)   { return m_net.endNodes (layer); }


// template void NeuralNet::mlp (double*, double*, EnumFunction, EnumFunction);
// template Pattern::Pattern<double*>(double*, double*, double*, double*, double);
// // template void NeuralNet::training_BP (size_t, size_t, 
// // 				  std::vector<Pattern>::iterator, std::vector<Pattern>::iterator,
// // 				  std::vector<Pattern>::iterator, std::vector<Pattern>::iterator,
// // 				   int, NNTYPE, NNTYPE, NNTYPE, size_t);
// // template void NeuralNet::training_LBFGS (size_t, size_t, 
// // 				  std::vector<Pattern>::iterator, std::vector<Pattern>::iterator,
// // 				  std::vector<Pattern>::iterator, std::vector<Pattern>::iterator,
// // 					 int, int, NNTYPE, NNTYPE, NNTYPE, size_t);
// template void NeuralNet::training_SCG (std::vector<Pattern>::iterator, std::vector<Pattern>::iterator,
//                                        std::vector<Pattern>::iterator, std::vector<Pattern>::iterator,
//                                        size_t, size_t);
// template void NeuralNet::calculateEvent (std::vector<NNTYPE>::iterator, std::vector<NNTYPE>::iterator);


bool NeuralNet::checkGradient (NNTYPE& E, NNTYPE& E_delta, NNTYPE& E_delta_gradient, const Pattern& pattern, NNTYPE delta, NNTYPE limit)
{

//    m_net.clear (e_dE);

    calculateSample (pattern.beginInput (), pattern.endInput ());

    // compute the error
    E = m_net.errorFunction (pattern.beginOutput (), pattern.endOutput (), pattern.weight ());
    

    int numWeights = std::distance (m_net.begin (), m_net.end ());


    m_net.compute_dE ();
    m_net.update_dE (false);

    int changeWeightPosition = randomInt (numWeights);
    auto it = m_net.begin ();
    std::advance (it, changeWeightPosition);
    Synapsis& syn = *it;

    NNTYPE dE = syn.dE;

    NNTYPE myDelta = -E/dE;

    syn.weight += myDelta;
    
    calculateSample (pattern.beginInput (), pattern.endInput ());
    E_delta = m_net.errorFunction (pattern.beginOutput (), pattern.endOutput (), pattern.weight ());

    E_delta_gradient = E + myDelta*dE;

    NNTYPE relDiff = fabs (E_delta-E_delta_gradient)/E;

    syn.weight -= myDelta;


    return relDiff <= limit;
}




