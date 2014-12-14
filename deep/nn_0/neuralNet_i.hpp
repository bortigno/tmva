#include <algorithm> 



#include <vector>
#include <map>
#include <math.h> 
#include <iostream> 
#include <ostream> 
#include <cassert> 

#include "../lbfgs/lbfgs.hpp"



template <typename Iterator, typename OutIterator>
inline void Net::compute (Iterator begin, Iterator end, OutIterator itOut)
{
    const int inputLayer = 0;
    node_iterator itNode = beginNodes (inputLayer);
    node_iterator itNodeEnd = endNodes (inputLayer);
    for (Iterator it = begin; it != end && itNode != itNodeEnd; ++it, ++itNode)
    {
    (*itNode)->value = (*it);
    }

    compute_E ();

    itNode  = beginNodes (layerCount () -1);
    itNodeEnd = endNodes (layerCount () -1);
    for (Iterator it = begin; it != end && itNode != itNodeEnd; ++it, ++itNode)
    {
    (*itOut) = (*it);
    }
}



template <typename ItOut>
inline NNTYPE Net::errorFunction (ItOut beginOutput, ItOut endOutput, NNTYPE patternWeight)
{
    NNTYPE errorSum = 0.0;

    size_t idxOutLyr = m_nodes.size () - 1;
//    size_t idx = 0;
    
    // output - value
    ItOut itOut = beginOutput;
//    std::cout << "end layer nodes : " << std::distance (beginNodes (idxOutLyr), endNodes (idxOutLyr)) << std::endl;
    for (node_iterator itNode = beginNodes (idxOutLyr), itNodeEnd = endNodes (idxOutLyr); itNode != itNodeEnd; ++itNode, ++itOut)
    {
        assert (itOut != endOutput);
//        std::cout << ".";
        INode* node = (*itNode);
        NNTYPE error = (*itOut) - node->value;
	node->delta = node->applyInverseActivationFunction (node->value) * error * patternWeight;
	errorSum += error*error  * patternWeight;
    }

    // weight decay (regularization)
    NNTYPE w2 = 0;
    for (synapsis_layer_iterator itLyr = beginSynapsisLayers (), itLyrEnd = endSynapsisLayers (); itLyr != itLyrEnd; ++itLyr)
    {
	SynapsisLayer& synLyr = itLyr->second;
	for (SynapsisLayer::synapsis_iterator itSyn = synLyr.beginSynapses (), itSynEnd = synLyr.endSynapses (); itSyn != itSynEnd; ++itSyn)
	{
	    Synapsis& syn = (*itSyn);
	    w2 += syn.weight*syn.weight;
	}
    }


    // error sum + weight term
    errorSum += 0.5 * m_pSettings->weightDecay * w2;

    return errorSum;
}




template <typename ItPattern>
inline NNTYPE Net::computeBatch (ItPattern beginPattern, ItPattern endPattern, EnumMode eMode, NNTYPE shift_E, NNTYPE shift_dE)
{
    clear ((EnumMode)(eMode | (shift_E!=0?e_shift:0)));

    NNTYPE batchError = 0.0;

    size_t idx = 0;
    size_t inputLayer = 0;

    double sumOfWeights = 0.0;
//    size_t outputLayer = layerCount () -1;
    for (ItPattern itP = beginPattern; itP != endPattern; ++itP, ++idx)
    {
        const Pattern& pattern = (*itP);
        Pattern::const_iterator itInp = pattern.beginInput ();
        Pattern::const_iterator itInpEnd = pattern.endInput ();
        Pattern::const_iterator itOut = pattern.beginOutput ();
        Pattern::const_iterator itOutEnd = pattern.endOutput ();

	node_iterator it = beginNodes (inputLayer);
	node_iterator itEnd = endNodes (inputLayer);
        for (; it != itEnd && itInp != itInpEnd; ++it, ++itInp)
        {
            INode* node = (*it);
            node->value = (*itInp);
        }
	while (it != itEnd)
	{
            INode* node = (*it);
            node->value = 1.0;
	    ++it;
	}

        NNTYPE patternWeight = pattern.weight ();
        sumOfWeights += fabs (patternWeight);

	NNTYPE error = 0.0;

	if ((eMode & e_E) != 0)
	{
	    compute_E (shift_E);
	    error = errorFunction (itOut, itOutEnd, patternWeight);
	    batchError += error;
	    ++m_count_E;
	}
	

	if ((eMode & e_dE) != 0)
	{
	    PRINT(std::endl << "before compute dE" << std::endl);
	    print (std::cout);
	    compute_dE (shift_dE);
	    PRINT(std::endl << "after compute dE" << std::endl);
	    print (std::cout);
	    update_dE (shift_dE != 0.0);
	    PRINT(std::endl << "after update dE" << std::endl);
	    print (std::cout);
	    PRINT(std::endl);
	    ++m_count_dE;
	}
    }
    if (sumOfWeights > 0.0)
        batchError /= sumOfWeights;

    if ((eMode & e_E) != 0)
	++m_pSettings->count_mb_E;

    if ((eMode & e_dE) != 0)
	++m_pSettings->count_mb_dE;

    return batchError;
}







    
template <typename ItPattern>
inline NNTYPE Net::trainBatch_SCG (ItPattern beginBatch, ItPattern endBatch)
{
//    std::cout << "SCG" << std::endl;
    NNTYPE sigma0 = 1.0e-2;
    NNTYPE lmbd = 1.0e-4;
    NNTYPE lmbd_x = 0.0;
    size_t k = 1;
    NNTYPE len_r = 0.0;
    bool success = true;

    size_t numSynapses (0);
    Net::synapsis_iterator itSynEnd = end ();
    for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
    {
	Synapsis& syn = (*itSyn);
//        std::cout << "syn : from : " << (syn.start) << "   to : " << (syn.end) << "       isend ? " << (itSyn==itSynEnd) << std::endl;
	syn.dE = 0.0;
	syn.dE_shifted = 0.0;
	++numSynapses;
    }

    std::vector<NNTYPE> syn_r  (numSynapses, 0.0);
    std::vector<NNTYPE> syn_r1 (numSynapses, 0.0);
    std::vector<NNTYPE>::iterator it_r  = syn_r.begin ();
    std::vector<NNTYPE>::iterator it_r1 = syn_r1.begin ();
    std::vector<NNTYPE>::iterator it_rEnd  = syn_r.end ();
//    std::vector<NNTYPE>::iterator it_r1End = syn_r1.end ();

    

    NNTYPE E = computeBatch (beginBatch, endBatch, (EnumMode)(e_E | e_dE));
    NNTYPE E_start = E;
    for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
    {
	Synapsis& syn = (*itSyn);
	(*it_r) = syn.p = -syn.dE;
//	syn.r1 = 0.0;
        ++it_r;
    }

    


    while (success)
    {
        len_r = 0.0;
//        std::cout << "while" << std::endl;
	NNTYPE len_p2 = 0.0;
	
	for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
	{
	    Synapsis& syn = (*itSyn);
	    len_p2 += std::pow (syn.p,2);
	}
	NNTYPE len_p = sqrt (len_p2);

	NNTYPE sigma = sigma0/len_p;
//	computeBatch (beginBatch, endBatch, (EnumMode)(e_E | e_dE | e_shift), 0.0, sigma);

	NNTYPE delta (0.0);

	size_t idx (0);
	for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn, ++idx)
	{
	    Synapsis& syn = (*itSyn);
//	    NNTYPE s = (syn.dE_shifted - syn.dE)/sigma;
	    NNTYPE s = (- syn.dE)/sigma;
//	    syn_s.at (idx) = s;
	    delta += syn.p*s;
	}

        delta += (lmbd-lmbd_x)*len_p2;

	if (delta <= 0)
	{
	    lmbd_x = 2.0*(lmbd - delta/(len_p2));
	    delta = -delta + lmbd* len_p2;
	    lmbd = lmbd_x;
	}


        NNTYPE mu = 0.0;
        it_r  = syn_r.begin ();
	for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
	{
	    Synapsis& syn = (*itSyn);
	    mu += syn.p * (*it_r);
            ++it_r;
	}

	NNTYPE alpha = mu/delta;

	NNTYPE E_shifted = computeBatch (beginBatch, endBatch, (EnumMode)(e_E), alpha);
	NNTYPE DELTA = 2*delta * (E-E_shifted)/(std::pow(mu,2.0));
	assert (DELTA == DELTA); // check for nan 

//	std::cout << "    DELTA " << DELTA << std::endl;
	if (DELTA >= 1.e-6)
	{
//            std::cout << "alpha = " << alpha << "    len_p = " << len_p << "    E = " << E << "     E_shifted = " << E_shifted << "    DELTA = " << DELTA << std::endl;
	    for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
	    {
		Synapsis& syn = (*itSyn);
		syn.weight += alpha * syn.p;
	    }	    
	    E = computeBatch (beginBatch, endBatch, (EnumMode)(e_E | e_dE));
            it_r1 = syn_r1.begin ();
	    for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
	    {
		Synapsis& syn = (*itSyn);
		//syn.r1 = -syn.dE;
		(*it_r1) = -syn.dE;
                ++it_r1;
	    }	    
	    lmbd_x = 0;
	    success = true;
	    
	    if (k % m_pSettings->repetitions == 0) // restart algorithm
	    {
//		std::cout << "      re k " << k << std::endl;
                it_r = syn_r.begin ();
		for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
		{
		    Synapsis& syn = (*itSyn);
//		    syn.p = syn.r;
		    syn.p = (*it_r);
                    ++it_r;
		}
	    }
	    else
	    {
//		std::cout << "      k " << k << std::endl;
		len_r = 0.0;
		NNTYPE r1_r = 0.0;
		for (it_r = syn_r.begin (), it_r1 = syn_r1.begin (); it_r != it_rEnd; ++it_r, ++it_r1)
		{
		    len_r += std::pow ((*it_r),2.0);
		    r1_r += (*it_r1) * (*it_r);
		}
		NNTYPE beta = (len_r - r1_r)/mu;
                it_r1 = syn_r1.begin ();
		for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
		{
		    Synapsis& syn = (*itSyn);
//		    syn.p = syn.r1 + beta * syn.p;
		    syn.p = (*it_r1) + beta * syn.p;
                    ++it_r1;
		}
	    }

	    if (k % m_pSettings->maxRepetitions == 0)
            {
//                std::cout << "  br maxrep " << k << std::endl;
		break;
            }

	    if (DELTA >= 0.75)
	    {
//		std::cout << "      DELTA >= 0.75 " << DELTA << std::endl;
		lmbd *= 0.5;
	    }
	    
	}
	else
	{
	    lmbd_x = lmbd;
	    success = false;
//            std::cout << "  br DELTA < 0 " << DELTA << std::endl;
	    break;
	}

	if (DELTA < 0.25 && DELTA > 1.0e-6)
	{
//	    std::cout << "      DELTA < 0.25 " << DELTA << std::endl;
	    lmbd *= 4.0;
	}

        if (lmbd > 1.0)
        {
//            std::cout << "  br lmbd > 1.0 " << lmbd << std::endl;
            break;
        }

	if (len_r > 1.0e-8)
	{
	    ++k;
            std::copy (syn_r1.begin (), syn_r1.end (), syn_r.begin ());
//	    std::cout << "      k+=1 " << k << std::endl;
	    
	}
	else
	{
//            std::cout << "  br len_r <= 1.0e-8 " << len_r << std::endl;
//	    clear ((EnumMode)(e_dE | e_shift));
	    return (E + E_start)/2.0;
	}
    }
//    std::cout << "ret E " << E << std::endl;
//    clear ((EnumMode)(e_dE | e_shift));
    return (E+E_start)/2.0;
}


    

template <typename Iterator>
inline void NeuralNet::mlp (Iterator itNodeNumbersBegin, Iterator itNodeNumbersEnd, EnumFunction eActivationFunction, EnumFunction eOutputLayerActivationFunction)
{

    EnumFunction eFunction;
    eFunction = eActivationFunction;
    size_t fncMax = std::max ((size_t)eActivationFunction, (size_t)eOutputLayerActivationFunction);
    size_t fncPos = 0x1;

//    std::cout << "layers : " << std::distance (itNodeNumbersBegin, itNodeNumbersEnd) << std::endl;

    size_t layer = 0;
    for (Iterator it = itNodeNumbersBegin; it != itNodeNumbersEnd; ++it)
    {
//        std::cout << "Create layer " << layer << std::endl;
	if (it+1 == itNodeNumbersEnd) // last layer
	    eFunction = eOutputLayerActivationFunction;

	for (size_t i = 0, iEnd = *it; i < iEnd; ++i)
	{
            while ((fncPos & eFunction) == 0)
            {
                if (fncPos > fncMax)
                    fncPos = 0x1;
                else
                    fncPos <<= 0x1;
            }
	    
	    INode* node (NULL);
	    switch (fncPos)
	    {
	    case eTanh:
		node = new Node<Tanh>;
		break;
	    case eTanhShift:
		node = new Node<TanhShift>;
		break;
	    case eLinear:
		node = new Node<Linear>;
		break;
	    case eGauss:
		node = new Node<Gauss>;
		break;
	    case eGaussComplement:
		node = new Node<GaussComplement>;
		break;
	    case eStep:
		node = new Node<Step>;
		break;
	    case eDoubleInvertedGauss:
		node = new Node<DoubleInvertedGauss>;
		break;
	    case ePeak:
		node = new Node<Peak>;
		break;
	    case eSoftSign:
	    default:
		node = new Node<SoftSign>;
		break;
	    };
	    m_net.addNode (layer, node);
//            std::cout << "    add node with " << fncPos << std::endl;
            fncPos <<= 0x1;
	}

	if (it+1 != itNodeNumbersEnd) // last layer
	{
	    INode* biasNode = new Node<Bias> ();
	    biasNode->value = 1.0;
	    m_net.addNode (layer, biasNode); // bias nodes
	}

        if (layer>0)
        {
//            std::cout << "connect layer " << (layer-1) << " with " << layer  << std::endl;
            m_net.connect (layer-1, layer);
        }

	++layer;
    }


    (*this).print (std::cout) ;


    // for (size_t i = 0, iEnd = layer-1; i < iEnd; ++i)
    // {
    // 	m_net.connect (i, i+1, eSigmoid, eFull);
    // }

//    m_net.connect (layer-2, layer-1, eLinear, eFull);
}



template <typename Iterator, typename IteratorOut>
inline NNTYPE NeuralNet::testSample (Iterator itInput, Iterator itInputEnd, IteratorOut itOutputBegin, IteratorOut itOutputEnd, NNTYPE weight)
{
    calculateSample (itInput, itInputEnd);

    // compute the error
    NNTYPE error = m_net.errorFunction (itOutputBegin, itOutputEnd, weight);


    size_t endLayer = m_net.layerCount (); // count of synapsis layers is count of node layers -1 
    m_pSettings->testSampleComputed (error, m_net.beginNodes (0), m_net.endNodes (0), m_net.beginNodes (endLayer), m_net.endNodes (endLayer), itOutputBegin, itOutputEnd, weight);

    return error;
}






template <typename Iterator>
inline void NeuralNet::calculateSample (Iterator itInput, Iterator itInputEnd)
{
    m_net.clear (e_dE);

    // set the values of the input nodes
    const int startLayer = 0;
    node_iterator itNode = m_net.beginNodes (startLayer);
    node_iterator itNodeEnd = m_net.endNodes (startLayer);
    while (itInput != itInputEnd)
    {
	INode* inputNode = *itNode;
	inputNode->value = *itInput;
	++itNode;
	++itInput;
    }
    while (itNode != itNodeEnd)
    {
	INode* inputNode = *itNode;
	inputNode->value = 1.0;
	++itNode;
    }

    m_net.compute_E ();
}






template <typename Iterator>
inline void NeuralNet::training_SCG (Iterator itTrainingsPatternBegin, Iterator itTrainingsPatternEnd, 
                              Iterator itTestPatternBegin, Iterator itTestPatternEnd, 
                              size_t batchSize, 
                              size_t convergenceSteps)
{
    std::cout << "START TRAINING : SCG" << std::endl;
    size_t convergenceCount = 0;
    NNTYPE minError = 1e10;

    size_t cycleCount = 0;
    size_t testCycleCount = 0;
    NNTYPE testError = 1e20;

    // until convergence
    while (convergenceCount < convergenceSteps)
    {
	std::cout << "train cycle " << cycleCount << std::endl;
        ++cycleCount;
	NNTYPE trainError = trainCycles_SCG (itTrainingsPatternBegin, itTrainingsPatternEnd, batchSize);
//	std::cout << "test cycle" << std::endl;

        if (testCycleCount % m_pSettings->testRepetitions == 0)
            testError = testCycle (itTestPatternBegin, itTestPatternEnd);

        ++testCycleCount;


	static double x = -1.0;
	x += 1.0;
	m_pSettings->resetPlot ("errors");
	m_pSettings->addPoint ("trainErrors", m_pSettings->count_E, trainError);
	m_pSettings->addPoint ("testErrors", m_pSettings->count_E, testError);
	m_pSettings->plot ("errors", "training", "trainErrors", "points", "");
	m_pSettings->plot ("errors", "training_", "trainErrors", "lines", "cspline");
	m_pSettings->plot ("errors", "test", "testErrors", "points", "");
	m_pSettings->plot ("errors", "test_", "testErrors", "lines", "cspline");



	(*this).print (std::cout);
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
}





template <typename Iterator>
inline NNTYPE NeuralNet::trainCycles_SCG (Iterator itPatternBegin, Iterator itPatternEnd, size_t batchSize)
{
    NNTYPE error = 0.0;
    size_t numPattern = std::distance (itPatternBegin, itPatternEnd);
    size_t numBatches = numPattern/batchSize;
    size_t numBatches_stored = numBatches;

    Iterator itPatternBatchBegin = itPatternBegin;
    Iterator itPatternBatchEnd = itPatternBatchBegin;
    std::random_shuffle (itPatternBegin, itPatternEnd);
    while (numBatches > 0)
    {
//        std::cout << "batch = " << numBatches << std::endl;
        std::advance (itPatternBatchEnd, batchSize);
        error += m_net.trainBatch_SCG (itPatternBatchBegin, itPatternBatchEnd);
        itPatternBatchBegin = itPatternBatchEnd;
        --numBatches;
    }
    if (itPatternBatchEnd != itPatternEnd)
	error += m_net.trainBatch_SCG (itPatternBatchEnd, itPatternEnd);
    error /= numBatches_stored;
    
    return error;
}




template <typename Iterator>
inline NNTYPE NeuralNet::trainCycles_LBFGS (size_t startSynapsisLayer, size_t endSynapsisLayer, 
				     Iterator itPatternBegin, Iterator itPatternEnd, 
				     size_t numberCycles, int memory, NNTYPE maxStep, NNTYPE damping, NNTYPE alpha,
				     size_t batchSize)
{
    std::cout << "train cycles LBFGS" << std::endl;
    std::vector<LBFGS> vecLbfgs;
    
//    size_t layerIdx = 0;
    for (Net::synapsis_layer_iterator it = m_net.beginSynapsisLayers (), itEnd = m_net.endSynapsisLayers (); it != itEnd; ++it)
	vecLbfgs.push_back (LBFGS (maxStep, memory, damping, alpha));

    // create position and force vector for all layers
    std::vector<LBFGS::container_type> vecPositions;
    std::vector<LBFGS::container_type> vecForces;
    for (Net::synapsis_layer_iterator it = m_net.beginSynapsisLayers (), itEnd = m_net.endSynapsisLayers (); it != itEnd; ++it)
    {
	SynapsisLayer& layer = (*it).second;
	std::vector<Synapsis>& synapses = layer.synapses ();

	vecPositions.push_back (LBFGS::container_type()); // positions
	vecPositions.back ().assign (synapses.size (), 0.0);
	vecForces.push_back (LBFGS::container_type()); // forces
	vecForces.back ().assign (synapses.size (), 0.0);
    }	



    NNTYPE sumError = 0.0;
//    if (true)
    {
	for (size_t i = 0; i < numberCycles; ++i)
	{
	    std::cout << "    cycle " << i << std::endl;
	    NNTYPE cycleError = 0.0;
	    size_t numPattern = std::distance (itPatternBegin, itPatternEnd);
	    int numBatches = numPattern/batchSize;
	    int numBatchesCount = numBatches;

	    Iterator itPatternBatchBegin = itPatternBegin;
	    Iterator itPatternBatchEnd = itPatternBatchBegin;
	    while (numBatchesCount >= 0)
	    {
		std::cout << "        remaining batches " << numBatchesCount << std::endl;
		NNTYPE batchError = 0.0;
		if (itPatternBatchEnd == itPatternEnd)
		    break;

		// next batch
		if (numBatchesCount == 0)
		{
		    itPatternBatchEnd = itPatternEnd;
		}

		std::advance (itPatternBatchEnd, batchSize);

		int maxOptCount = 10;
		bool isOptimizing = true;
		while (isOptimizing && maxOptCount > 0)
		{
		    --maxOptCount;
		    isOptimizing = false;
		    NNTYPE error = propagateBatch (startSynapsisLayer, endSynapsisLayer, itPatternBatchBegin, itPatternBatchEnd);
		    std::cout << "        propagate Batch, error ::: " << error << "    maxCount= " << maxOptCount << std::endl;

		    if (error < 0.01)
		    {
			isOptimizing = false;
			break;
		    }

		    // train all layers
		    size_t layerIdx = 0;
		    for (Net::synapsis_layer_iterator it = m_net.beginSynapsisLayers (), itEnd = m_net.endSynapsisLayers (); it != itEnd; ++it)
		    {
			SynapsisLayer& layer = (*it).second;

			LBFGS::container_type& positions = vecPositions.at (layerIdx);
			LBFGS::container_type& forces = vecForces.at (layerIdx);

			// get positions and forces from the layer
			size_t synIdx = 0;
			for (typename SynapsisLayer::synapsis_iterator it = layer.beginSynapses (), itEnd = layer.endSynapses (); it != itEnd; ++it)
			{
			    Synapsis& syn = *it;
			    positions.at (synIdx) = syn.weight;
			    forces.at (synIdx) = syn.end->delta;
			    ++synIdx;
			}

			// optimization
			LBFGS& lbfgs = vecLbfgs.at (layerIdx);
			isOptimizing |= lbfgs.step (positions, forces);
//			std::cout << "forces = " << forces << std::endl;
			// set new positions
			synIdx = 0;
			for (typename SynapsisLayer::synapsis_iterator it = layer.beginSynapses (), itEnd = layer.endSynapses (); it != itEnd; ++it)
			{
			    Synapsis& syn = *it;
			    syn.weight = positions.at (synIdx);
			    ++synIdx;
			}
//			(*this).print (std::cout );
//			std::cout << std::endl;
			++layerIdx;

		    }
		    batchError += error;
		}
		--numBatchesCount;
		itPatternBatchBegin = itPatternBatchEnd;
		cycleError = batchError/numBatches;
	    }
	    std::cout << "cycle error= " << cycleError << "   batchsize = " << batchSize << std::endl;
	    cycleError /= batchSize;
	    sumError += cycleError;
//	    (*this).print (std::cout );

	}
	sumError /= numberCycles;
    }
    // else
    // {
    // 	for (size_t i = 0; i < numberCycles; ++i)
    // 	{
    // 	    NNTYPE sumWeights = 0.0;
    // 	    NNTYPE error = 0.0;
    // 	    for (Iterator itPattern = itPatternBegin; itPattern != itPatternEnd; ++itPattern)
    // 	    {
    // 		Pattern& pattern = *itPattern;
    // 		error += pattern.weight () * propagateEvent (startSynapsisLayer, endSynapsisLayer, pattern.beginInput (), pattern.endInput (), pattern.beginOutput ());
    // 		m_net.updateWeights_LBFGS (memory, maxStep, damping, alpha);
    // 		sumWeights += pattern.weight ();
    // 	    }
    // 	    error /= sumWeights;
    // 	    sumError += error;
    // 	}
    // 	sumError /= numberCycles;
    // }
    return sumError;
}




template <typename Iterator>
inline NNTYPE NeuralNet::testCycle (Iterator itPatternBegin, Iterator itPatternEnd)
{
    m_pSettings->startTestCycle ();

    NNTYPE error = 0.0;
    NNTYPE sumOfWeights = 0.0;
    for (Iterator itPattern = itPatternBegin; itPattern != itPatternEnd; ++itPattern)
    {
	Pattern& pattern = *itPattern;
	error += testSample (pattern.beginInput (), pattern.endInput (), pattern.beginOutput (), pattern.endOutput (), pattern.weight ());
	sumOfWeights += pattern.weight ();
    }
    error /= sumOfWeights;

    m_pSettings->endTestCycle ();
    return error;
}












// template <typename Iterator>
// void NeuralNet::training_LBFGS (size_t startSynapsisLayer, size_t endSynapsisLayer, 
// 				Iterator itTrainingsPatternBegin, Iterator itTrainingsPatternEnd, Iterator itTestPatternBegin, Iterator itTestPatternEnd, int convergence, int memory, NNTYPE maxStep, NNTYPE damping, NNTYPE alpha, size_t batchSize)
// {
//     std::cout << "START TRAINING : LBFGS" << std::endl;
//     int convergenceCount = 0;
//     NNTYPE minError = 1e10;


//     // until convergence
//     while (convergenceCount < convergence)
//     {
// //	std::cout << "train cycles" << std::endl;
// 	NNTYPE trainError = trainCycles_LBFGS (startSynapsisLayer, endSynapsisLayer, 
// 					       itTrainingsPatternBegin, itTrainingsPatternEnd, 5,
// 					 memory, maxStep, damping, alpha, batchSize);
// //	std::cout << "test cycle" << std::endl;
// 	NNTYPE testError = testCycle (startSynapsisLayer, endSynapsisLayer, 
// 				      itTestPatternBegin, itTestPatternEnd);

// 	(*this).print (std::cout);
// 	std::cout << "check convergence; minError " << minError << "  current " << testError << std::endl;
// 	if (testError < minError)
// 	{
// 	    convergenceCount = 0;
// 	    minError = testError;
// 	}
// 	else
// 	    ++convergenceCount;

// 	std::cout << "testError : " << testError << "   trainError : " << trainError << std::endl;
//     }
//     std::cout << "END TRAINING" << std::endl;
// }




// template <typename Iterator>
// void NeuralNet::training_BP (size_t startSynapsisLayer, size_t endSynapsisLayer, 
// 			  Iterator itTrainingsPatternBegin, Iterator itTrainingsPatternEnd, 
// 			  Iterator itTestPatternBegin, Iterator itTestPatternEnd,
// 			  int convergence, NNTYPE learningRate, NNTYPE learningDecrease, NNTYPE momentum, size_t batchSize)
// {
//     std::cout << "START TRAINING : BP" << std::endl;
//     int convergenceCount = 0;
//     NNTYPE minError = 1e10;
//     while (convergenceCount < convergence)
//     {
// //	std::cout << "train cycles" << std::endl;
// 	NNTYPE trainError = trainCycles_BP (startSynapsisLayer, endSynapsisLayer, 
// 					 itTrainingsPatternBegin, itTrainingsPatternEnd,
// 					 2, learningRate, momentum, batchSize);
// //	std::cout << "test cycle" << std::endl;
// 	NNTYPE testError = testCycle (startSynapsisLayer, endSynapsisLayer, 
// 				      itTestPatternBegin, itTestPatternEnd);

// 	learningRate *= learningDecrease;
// 	std::cout << "check convergence; minError " << minError << "  current " << testError << std::endl;
// 	if (testError < minError)
// 	{
// 	    convergenceCount = 0;
// 	    minError = testError;
// 	}
// 	else
// 	    ++convergenceCount;

// 	std::cout << "testError : " << testError << "   trainError : " << trainError << "     learningRate=" << learningRate << std::endl;
//     }
//     std::cout << "END TRAINING" << std::endl;
// }


