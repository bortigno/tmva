
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <tuple>
#include <cmath>
#include <cassert>
#include <random>
#include <sstream>

#define NNTYPE double

class Pattern
{
public:
    
    typedef typename std::vector<NNTYPE>::iterator iterator;
    typedef typename std::vector<NNTYPE>::const_iterator const_iterator;


    Pattern ()
	: m_weight (0)
    {
    }

    ~Pattern () 
    { 
    }

    Pattern (const Pattern& other) 
    { 
	m_input.assign (std::begin (other.m_input), std::end (other.m_input)); 
	m_output.assign (std::begin (other.m_output), std::end (other.m_output)); 
	m_weight = other.m_weight; 
    }

    Pattern (Pattern&& other) 
    { 
        m_input = std::move (other.m_input);
	m_output = std::move (other.m_output);
	m_weight = other.m_weight; 
    }


    Pattern& operator= (const Pattern& other) 
    { 
	m_input.assign (std::begin (other.input ()), std::end (other.input ())); 
	m_output.assign (std::begin (other.output ()), std::end (other.output ())); 
	m_weight = other.m_weight; 
	return *this;
    }


    template <typename ItValue>
    Pattern (ItValue inputBegin, ItValue inputEnd, ItValue outputBegin, ItValue outputEnd, NNTYPE weight = 1.0)
        : m_input (inputBegin, inputEnd)
        , m_output (outputBegin, outputEnd)
        , m_weight (weight)
    {
    }



    template <typename InputContainer, typename OutputContainer>
    Pattern (InputContainer& input, OutputContainer& output, NNTYPE weight = 1.0)
        : m_input (std::begin (input), std::end (input))
        , m_output (std::begin (output), std::end (output))
        , m_weight (weight)
    {
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

private:
    std::vector<NNTYPE> m_input;
    std::vector<NNTYPE> m_output;
    NNTYPE m_weight;
};





std::vector<Pattern> readCVS (std::string filename)
{
    std::vector<Pattern> pattern;

    std::ifstream file (filename); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/

    // first line
    size_t line = 0;
    std::string value;
    while (file.good())
    {
	getline ( file, value, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
//	std::cout << std::string (value, 1, value.length()-2); // display value removing the first and the last character from it
	std::cout << value << " ";
	if (line > 100)
	    break;
	++line;
    }
    return pattern;
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
                value = 1.0;
            else 
                value = 0.0;
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




std::vector<Pattern> readCSV_2 (std::string filename, std::vector<std::string> fieldNames, std::string labelField, std::string weightField, size_t maxLines = ~(size_t(0)))
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

    std::vector<std::string>::iterator itLabel = std::find_if (begin (fieldNames), end (fieldNames), [labelField](const std::string& name) { return name == labelField; } );
    std::vector<std::string>::iterator itWeight = std::find_if (begin (fieldNames), end (fieldNames), [weightField](const std::string& name) { return name == weightField; } );

    size_t idxLabel = std::distance (begin (fieldNames), itLabel);
    size_t idxWeight = std::distance (begin (fieldNames), itWeight);

    while (infile.is_open () && infile.good ())
    {        
        if (idxLine >= maxLines)
            break;
        std::vector<double> values;
        values = parseLine<double> (infile, idxLabel);
        pattern.push_back (Pattern (begin (values), end (values)-2), begin (values)+idxLabel, begin (values) + idxLabel +1, values.at (idxWeight));
        for_each (begin (values), end (values), [](double v){ std::cout << v << ", "; } );
        std::cout << std::endl;
        if (infile.eof())
        {
            break;
        }
        ++idxLine;
    }
    return pattern;
}

