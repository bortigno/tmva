
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>

#include "../pattern/pattern.hpp"

#if 0
class Pattern
{
public:

    
    typedef typename std::vector<double>::iterator iterator;
    typedef typename std::vector<double>::const_iterator const_iterator;

    Pattern () 
    : m_input () 
    , m_output ()
    , m_weight (0)
    {
	std::cout << "Pattern()   this=" << this << std::endl;
    }

    ~Pattern () 
    {
	std::cout << "~Pattern()   this=" << this << std::endl;
    }


    Pattern (Pattern&& other) 
    : m_input (std::move (other.m_input))
    , m_output (std::move (other.m_output))
    , m_weight (other.m_weight)
    {
	std::cout << "Pattern(Pattern&&)   this=" << this << "    other=" << (&other) << std::endl;
    }

    Pattern (const Pattern& other) 
    : m_input (other.m_input)
    , m_output (other.m_output)
    , m_weight (other.m_weight)
    {
	std::cout << "Pattern(Pattern&&)   this=" << this << "    other=" << (&other) << std::endl;
    }

    Pattern& operator= (Pattern&& other) 
    {
	std::cout << "Pattern operator=(Pattern&&)   this=" << this << "    other=" << (&other) << std::endl;
	m_input = std::move (other.m_input);
	m_output = std::move (other.m_output);
	m_weight = other.m_weight;
	return *this;
    }

    Pattern& operator= (const Pattern& other) 
    {
	std::cout << "Pattern operator=(Pattern&)   this=" << this << "    other=" << (&other) << std::endl;
	m_input = other.m_output;
	m_output = other.m_output;
	m_weight = other.m_weight;
	return *this;
    }

    template <typename It>
    Pattern (It itInputBegin, It itInputEnd, It itOutputBegin, It itOutputEnd, double weight)
    : m_input (itInputBegin, itInputEnd)
    , m_output (itOutputBegin, itOutputEnd)
    , m_weight (weight)
    {
	std::cout << "Pattern(it)   this=" << this << std::endl;
    }


    const_iterator beginInput () const { return m_input.begin (); }
    const_iterator endInput   () const  { return m_input.end (); }
    const_iterator beginOutput () const  { return m_output.begin (); }
    const_iterator endOutput   () const  { return m_output.end (); }

    double weight () const { return m_weight; }
    void weight (double w) { m_weight = w; }

    size_t inputSize () const { return m_input.size (); }
    size_t outputSize () const { return m_output.size (); }

    void addInput (double value) { m_input.push_back (value); }
    void addOutput (double value) { m_output.push_back (value); }

    std::vector<double>& input  () { return m_input; }
    std::vector<double>& output () { return m_output; }
    const std::vector<double>& input  () const { return m_input; }
    const std::vector<double>& output () const { return m_output; }


private:
    std::vector<double> m_input;
    std::vector<double> m_output;
    double m_weight;

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
#endif


int main ()
{
    std::vector<Pattern> vecPattern;
    vecPattern.reserve (10);

    for (double i = 0; i < 10; ++i)
    {
	std::cout << "=== iteration " << i << std::endl;
	auto v = { 1+i, 2+i, 3+i, 4+i};
	auto v2 = { 1-i, 2-i, 3-i, 4-i};
	vecPattern.emplace_back (Pattern (begin (v), end (v), begin (v2), end (v2), i));
    }
    return 0;
}


