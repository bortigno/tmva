#ifndef _PATTERN_HPP_
#define _PATTERN_HPP_

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <sstream>

#define NNTYPE double




class Pattern
{
public:
    
    typedef typename std::vector<NNTYPE>::iterator iterator;
    typedef typename std::vector<NNTYPE>::const_iterator const_iterator;


    Pattern ()
    : m_ID (0)
    , m_input ()
    , m_output ()
    , m_weight (0)
    {
//	std::cout << "Pattern()" << "  this=" << this << std::endl;)
    }

    ~Pattern () 
    { 
//	std::cout << "~Pattern()" << "  this=" << this << std::endl;
    }

    Pattern (Pattern&& other) 
    : m_ID (0)
    , m_input (std::move (other.m_input))
    , m_output (std::move (other.m_output))
    , m_weight (other.m_weight)
    { 
//    	std::cout << "Pattern&&" << "  this=" << this << "   other=" << (&other) << std::endl;
    }

    Pattern (const Pattern& other) 
    : m_ID (other.m_ID)
    , m_input (other.m_input)
    , m_output (other.m_output)
    , m_weight (other.m_weight)
    { 
//	std::cout << "Pattern&" << "   this=" << this << "   other=" << (&other) << std::endl;
    }


    Pattern& operator= (Pattern&& other) 
    {
//	std::cout << "Pattern=&&" << "  this=" << this << "   other=" << (&other) << std::endl;
        m_ID = other.m_ID;
    	m_input = std::move (other.m_input);
    	m_output = std::move (other.m_output);
    	m_weight = other.m_weight;
    	return *this;
    }

    Pattern& operator= (const Pattern& other) 
    { 
//	std::cout << "Pattern=&" << "  this=" << this << "   other=" << (&other) << std::endl;
        m_ID = other.m_ID;
	m_input = other.m_input;
	m_output = other.m_output;
	m_weight = other.m_weight; 
	return *this;
    }


    template <typename ItValue>
    Pattern (ItValue inputBegin, ItValue inputEnd, ItValue outputBegin, ItValue outputEnd, NNTYPE weight = 1.0)
    : m_ID (0)
    , m_input (inputBegin, inputEnd)
    , m_output (outputBegin, outputEnd)
    , m_weight (weight)
    {
//	std::cout << "Pattern(it)" << "  this=" << this << std::endl;
    }



    Pattern (std::initializer_list<double> input, std::initializer_list<double> output, NNTYPE weight = 1.0)
    : m_ID (0)
    , m_input (std::begin (input), std::end (input))
    , m_output (std::begin (output), std::end (output))
    , m_weight (weight)
    {
//	std::cout << "Pattern(container)" << "  this=" << this << std::endl;
    }


    template <typename InputContainer, typename OutputContainer>
    Pattern (InputContainer& input, OutputContainer& output, NNTYPE weight = 1.0)
    : m_ID (0)
    , m_input (std::begin (input), std::end (input))
    , m_output (std::begin (output), std::end (output))
    , m_weight (weight)
    {
//	std::cout << "Pattern(container)" << "  this=" << this << std::endl;
    }


    const_iterator beginInput () const { return m_input.begin (); }
    const_iterator endInput   () const  { return m_input.end (); }
    const_iterator beginOutput () const  { return m_output.begin (); }
    const_iterator endOutput   () const  { return m_output.end (); }

    NNTYPE weight () const { return m_weight; }
    void weight (NNTYPE w) { m_weight = w; }

    size_t inputSize () const { return m_input.size (); }
    size_t outputSize () const { return m_output.size (); }

    void addInput (NNTYPE value) { m_input.push_back (value); }
    void addOutput (NNTYPE value) { m_output.push_back (value); }

    std::vector<NNTYPE>& input  () { return m_input; }
    std::vector<NNTYPE>& output () { return m_output; }
    const std::vector<NNTYPE>& input  () const { return m_input; }
    const std::vector<NNTYPE>& output () const { return m_output; }

    void setID (size_t id) { m_ID = id; }
    size_t getID () const { return m_ID; }

private:
    size_t m_ID;
    std::vector<NNTYPE> m_input;
    std::vector<NNTYPE> m_output;
    NNTYPE m_weight;

    friend std::ostream& operator<< (std::ostream& os, const Pattern& pattern);
};


inline std::ostream& operator<< (std::ostream& os, const Pattern& pattern)
{
    os << "I: ";
    for_each (pattern.beginInput (), pattern.endInput (), [&os](double v){ os << v << ", "; } );
    os << std::endl << "O: ";
    for_each (pattern.beginOutput (), pattern.endOutput (), [&os](double v){ os << v << ", "; } );
    os << std::endl;
    return os;
}




template <typename T>
auto parseLine (std::ifstream& infile, size_t idxLabel) -> std::vector<T>
{
    std::string line;
    if (!getline (infile, line))
        return std::vector<T> ();

    std::stringstream ssline (line);
    std::vector<T> record;

    size_t idx = 0;
    while (ssline)
    {
        T value;
        std::string token;
        if (!getline (ssline, token, ',')) 
            break;

        if (idx == idxLabel)
        {
            if (token == "s")
                value = 0.9;
            else 
                value = 0.1;
        }
        else
        {
            std::stringstream tr;
            tr << token;
            tr >> value;
        }
        record.push_back (value);
        ++idx;
    }
    return record;
}




inline std::vector<Pattern> readCSV (std::string filename, std::vector<std::string>& fieldNames, 
                                     std::string IDField, std::string labelField, std::string weightField, 
                                     double& sumOfSigWeights, double& sumOfBkgWeights,
                                     size_t maxLines = ~(size_t(0)), size_t skipLines = 0)
{
    std::vector<Pattern> pattern;
    std::ifstream infile (filename);
    
    size_t idxLine = 0;

    // get field names (first line)
    if (infile.is_open () && infile.good ())
    {
        fieldNames = parseLine<std::string> (infile, 5000);
        for_each (begin (fieldNames), end (fieldNames), [](std::string name){ std::cout << name << ", "; } );
        std::cout << std::endl;
    }

    std::vector<std::string>::iterator itID = std::find_if (begin (fieldNames), end (fieldNames), [IDField](const std::string& name) { return name == IDField; } );
    std::vector<std::string>::iterator itLabel = std::find_if (begin (fieldNames), end (fieldNames), [labelField](const std::string& name) { return name == labelField; } );
    std::vector<std::string>::iterator itWeight = std::find_if (begin (fieldNames), end (fieldNames), [weightField](const std::string& name) { return name == weightField; } );

    size_t idxID = std::distance (begin (fieldNames), itID);
    size_t idxLabel = std::distance (begin (fieldNames), itLabel);
    size_t idxWeight = std::distance (begin (fieldNames), itWeight);

    sumOfSigWeights = 0.0;
    sumOfBkgWeights = 0.0;

    while (infile.is_open () && infile.good ())
    {        
        if (idxLine >= maxLines+skipLines)
            break;
        if (idxLine < skipLines)
	{
            std::string line;
            if (!getline (infile, line))
                return pattern;
	    ++idxLine;
            continue;
	}
        ++idxLine;
        std::vector<double> values;
        values = parseLine<double> (infile, idxLabel);
        size_t ID = values.at (idxID);
        values.at (0) = 1.0; // replace the ID by the bias node (value = 1)
//	for_each (begin (values), end (values), [](double& v){ v = v < -990 ? 0 : v; } );

//	auto beginValues = begin (values)+1; // +1 because of ID
	auto beginValues = begin (values); // we replaced the ID by one (== bias node)
	auto endValues = begin (values) + idxWeight;
	auto beginOutput = begin (values) + idxLabel; 
	auto endOutput = beginOutput + 1;

	auto weight = values.size () > idxWeight ? values.at (idxWeight) : 1.0;
        auto truth = values.size () > idxLabel ? values.at (idxLabel) : 0.0;

        pattern.emplace_back (Pattern (beginValues, endValues, beginOutput, endOutput, weight));
        pattern.back ().setID (ID);
        

        if (truth > 0.5)
            sumOfSigWeights += fabs (weight);
        else 
            sumOfBkgWeights += fabs (weight);

        if (infile.eof())
        {
            break;
        }
    }


    size_t length = pattern.back ().inputSize ();
// normalization
    std::vector<double> minima (length, 1e10);
    std::vector<double> maxima (length, -1e10);
    for (Pattern& p : pattern)
    {
	std::vector<double>::iterator itMin = begin (minima);
	std::vector<double>::iterator itMax = begin (maxima);
	for (auto it = begin (p.input ()), itEnd = end (p.input ()); it != itEnd; ++it, ++itMin, ++itMax)
	{
	    double& val = *it;
	    double& mi = *itMin;
	    double& ma = *itMax;
	    if (val < -900)
		continue;
	    if (val > ma)
		ma = val;
	    if (val < mi)
		mi = val;
	}
    } 
    for (Pattern& p : pattern)
    {
	std::vector<double>::iterator itMin = begin (minima);
	std::vector<double>::iterator itMax = begin (maxima);
	for (auto it = begin (p.input ()), itEnd = end (p.input ()); it != itEnd; ++it, ++itMin, ++itMax)
	{
	    double& val = *it;
	    double& mi = *itMin;
	    double& ma = *itMax;
	    if (val < -900)
		val = 0.0;
	    else if (fabs (ma-mi) > 1e-5)
		val = (val-mi)/(ma - mi);
	}
    } 


#if true

    if (sumOfSigWeights > 0 && sumOfBkgWeights > 0)
    {
	sumOfSigWeights /= maxLines;
	sumOfBkgWeights /= maxLines;
	for (auto& p : pattern)
	{
	    if (p.output ().at (0) > 0.5)
		p.weight (p.weight ()  / sumOfSigWeights);
	    else
		p.weight (p.weight ()  / sumOfBkgWeights);
	}
    }
    else
    {
	sumOfSigWeights = 1.0;
	sumOfBkgWeights = 1.0;
    }
#else
    sumOfSigWeights = 1.0;
    sumOfBkgWeights = 1.0;
#endif

    return pattern;
}





#endif









// class StochasticCG
// {
// public:

//     size_t m_repetitions;
//     size_t m_maxRepetitions;

//     StochasticCG (size_t repetitions = 20, size_t maxRepetitions = 50) 
// 	: m_repetitions (repetitions)
// 	, m_maxRepetitions (maxRepetitions)
//     {}

//     template <typename Function, typename Weights, typename PassThrough>
//     double operator() (Function& errorFunction, Weights& weights, PassThrough& passThrough) 
//     {

// 	NNTYPE sigma0 = 1.0e-2;
// 	NNTYPE lmbd = 1.0e-4;
// 	NNTYPE lmbd_x = 0.0;
// 	size_t k = 1;
// 	NNTYPE len_r = 0.0;
// 	bool success = true;

// 	size_t numWeights = weights.size ();

// 	std::vector<NNTYPE> gradients (numWeights, 0.0);

// 	std::vector<NNTYPE> syn_r  (numWeights, 0.0);
// 	std::vector<NNTYPE> syn_r1 (numWeights, 0.0);
// 	std::vector<NNTYPE>::iterator it_r  = syn_r.begin ();
// 	std::vector<NNTYPE>::iterator it_r1 = syn_r1.begin ();
// 	std::vector<NNTYPE>::iterator it_rEnd  = syn_r.end ();
    

// 	NNTYPE E = errorFunction (passThrough, weights, gradients);
// 	NNTYPE E_start = E;

// 	std::vector<NNTYPE> vecP;
// 	vecP.reserve (weights.size ());
// 	for (NNTYPE g: gradients)
// 	{
// 	    vecP.push_back (-g);
// 	    (*it_r) = -g;
// 	    ++it_r;
// 	}
    


// 	while (success)
// 	{
// 	    len_r = 0.0;
// //        std::cout << "while" << std::endl;
// 	    NNTYPE len_p2 = 0.0;

// 	    for_each (begin (vecP), end (vecP), [&len_p2](double p)
// 		      {
// 			  len_p2 += std::pow (p, 2);
// 		      } );

//             if (len_p2 < sigma0)
//                 return 0;

// 	    NNTYPE len_p = sqrt (len_p2);

// 	    NNTYPE sigma = sigma0/len_p;
// //	computeBatch (beginBatch, endBatch, (EnumMode)(e_E | e_dE | e_shift), 0.0, sigma);

// 	    NNTYPE delta (0.0);

// 	    std::vector<double>::iterator itP = begin (vecP);
// 	    for_each (begin (gradients), end (gradients), [sigma, &delta, &itP](double g)
// 		      {
// 			  NNTYPE s = -g/sigma;
// 			  delta += (*itP) * s;
// 			  ++itP;
// 		      } );


// 	    delta += (lmbd-lmbd_x)*len_p2;

// 	    if (delta <= 0)
// 	    {
// 		lmbd_x = 2.0*(lmbd - delta/(len_p2));
// 		delta = -delta + lmbd* len_p2;
// 		lmbd = lmbd_x;
// 	    }


// 	    NNTYPE mu = 0.0;
// 	    it_r  = syn_r.begin ();

// 	    it_r = begin (syn_r);
// 	    for_each (begin (vecP), end (vecP), [&mu, &it_r](double p)
// 		      {
// 			  mu += p * (*it_r);
// 			  ++it_r;
// 		      } );


// 	    NNTYPE alpha = mu/delta;

// 	    std::vector<NNTYPE> weights_plus_alpha_p (numWeights, 0.0);
// //            syn.end->value += syn.start->value * (syn.weight + alpha * syn.p);
// 	    auto it_p = begin (vecP);
// 	    auto it_wpap = begin (weights_plus_alpha_p);
// 	    for_each (begin (weights), end (weights), [&it_p, alpha, &it_wpap](double w)
// 		      {
// 			  (*it_wpap) = w + alpha*(*it_p);
// 			  ++it_wpap; ++it_p;
// 		      });
// 	    NNTYPE E_shifted = errorFunction (passThrough, weights_plus_alpha_p);
// //	NNTYPE E_shifted = computeBatch (beginBatch, endBatch, (EnumMode)(e_E), alpha);

// 	    NNTYPE DELTA = 2*delta * (E-E_shifted)/(std::pow(mu,2.0));
// 	    assert (DELTA == DELTA); // check for nan 

// //	std::cout << "    DELTA " << DELTA << std::endl;
// 	    if (DELTA >= 1.e-6)
// 	    {
// //            std::cout << "alpha = " << alpha << "    len_p = " << len_p << "    E = " << E << "     E_shifted = " << E_shifted << "    DELTA = " << DELTA << std::endl;
// 		// for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
// 		// {
// 		// 	Synapsis& syn = (*itSyn);
// 		// 	syn.weight += alpha * syn.p;
// 		// }	    
// 		gradients.assign (numWeights, 0.0);
// 		E = errorFunction (passThrough, weights_plus_alpha_p, gradients);
// //	    E = computeBatch (beginBatch, endBatch, (EnumMode)(e_E | e_dE));
// 		it_r1 = begin (syn_r1);
// 		for_each (begin (gradients), end (gradients), [&it_r1](double g)
// 			  {
// 			      (*it_r1) = -g;
// 			      ++it_r1;
// 			  });
// 		// for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
// 		// {
// 		// 	Synapsis& syn = (*itSyn);
// 		// 	//syn.r1 = -syn.dE;
// 		// 	(*it_r1) = -syn.dE;
// 		//     ++it_r1;
// 		// }	    
// 		lmbd_x = 0;
// 		success = true;
	    
// 		if (k % m_repetitions == 0) // restart algorithm
// 		{
// //		std::cout << "      re k " << k << std::endl;
// 		    it_r = begin (syn_r);
// 		    for_each (begin (vecP), end (vecP), [&it_r](double& p)
// 			      {
// 				  p = (*it_r);
// 				  ++it_r;
// 			      });

// // 		for (Net::synapsis_iterator itSyn = begin (); itSyn != itSynEnd; ++itSyn)
// // 		{
// // 		    Synapsis& syn = (*itSyn);
// // //		    syn.p = syn.r;
// // 		    syn.p = (*it_r);
// //                     ++it_r;
// // 		}
// 		}
// 		else
// 		{
// //		std::cout << "      k " << k << std::endl;
// 		    len_r = 0.0;
// 		    NNTYPE r1_r = 0.0;
// 		    for (it_r = syn_r.begin (), it_r1 = syn_r1.begin (); it_r != it_rEnd; ++it_r, ++it_r1)
// 		    {
// 			len_r += std::pow ((*it_r),2.0);
// 			r1_r += (*it_r1) * (*it_r);
// 		    }
// 		    NNTYPE beta = (len_r - r1_r)/mu;
// 		    it_r1 = syn_r1.begin ();
		
// 		    for (auto& p: vecP)
// 		    {
// 			p = (*it_r1) + beta * p;
// 			++it_r1;
// 		    }
// 		}

// 		if (k % m_maxRepetitions == 0)
// 		{
// //                std::cout << "  br maxrep " << k << std::endl;
// 		    break;
// 		}

// 		if (DELTA >= 0.75)
// 		{
// //		std::cout << "      DELTA >= 0.75 " << DELTA << std::endl;
// 		    lmbd *= 0.5;
// 		}
	    
// 	    }
// 	    else
// 	    {
// 		lmbd_x = lmbd;
// 		success = false;
// //            std::cout << "  br DELTA < 0 " << DELTA << std::endl;
// 		break;
// 	    }

// 	    if (DELTA < 0.25 && DELTA > 1.0e-6)
// 	    {
// //	    std::cout << "      DELTA < 0.25 " << DELTA << std::endl;
// 		lmbd *= 4.0;
// 	    }

// 	    if (lmbd > 1.0)
// 	    {
// //            std::cout << "  br lmbd > 1.0 " << lmbd << std::endl;
// 		break;
// 	    }

// 	    if (len_r > 1.0e-8)
// 	    {
// 		++k;
// 		std::copy (syn_r1.begin (), syn_r1.end (), syn_r.begin ());
// //	    std::cout << "      k+=1 " << k << std::endl;
	    
// 	    }
// 	    else
// 	    {
// //            std::cout << "  br len_r <= 1.0e-8 " << len_r << std::endl;
// //	    clear ((EnumMode)(e_dE | e_shift));
// 		return (E + E_start)/2.0;
// 	    }
// 	}
// //    std::cout << "ret E " << E << std::endl;
// //    clear ((EnumMode)(e_dE | e_shift));
// 	return (E+E_start)/2.0;
//     }
// };
