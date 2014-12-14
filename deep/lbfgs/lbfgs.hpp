#include <iostream>

#include <vector>
#include <stdlib.h>
#include <math.h>

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif
#ifndef MAX
#define MAX(a,b) ((a)>=(b)?(a):(b))
#endif


inline std::ostream& operator<< (std::ostream& ostr, const std::vector<double>& data)
{
    for (std::vector<double>::const_iterator it = data.begin (), itEnd = data.end (); it != itEnd; ++it)
    {
	ostr << "  " << (*it);
    }
    return ostr;
}


template <typename Iterator>
double dot (Iterator itA, Iterator itAEnd, Iterator itB, Iterator itBEnd)
{
    double result = 0.0;
    for (; itA != itAEnd && itB != itBEnd; ++itA, ++itB)
    {
	result += (*itA) * (*itB);
    }
    return result;
}


class LBFGS
{
public:
    typedef std::vector<double> container_type;

    // maxStep = 0.04
    LBFGS (double maxStep = 0.04, size_t memory = 100, double damping = 1.0, double alpha = 70.0, double minForce = 0.001)
	: m_maxStep (maxStep), m_memory (memory), m_damping (damping), /*m_alpha (alpha), */m_minForce (minForce)
    {
	m_H0 = 1.0/alpha;
	m_iteration = 0;
    }

    bool step (container_type& positions, container_type& force)
    {
	if (dot (force.begin (), force.end (), force.begin (), force.end ()) < m_minForce)
	    return false;

	// container_type r;
	// r.assign (positions.begin (), positions.end ());
	update (positions, force, m_r0, m_f0);

//	container_type s; s.assign (m_s.begin (), m_s.end ());
//	container_type y; y.assign (m_y.begin (), m_y.end ());
	container_type rho; rho.assign (m_rho.begin (), m_rho.end ());
//	container_type H0; H0.assign (m_H0.begin (), m_H0.end ());
	double H0 = m_H0;

//	std::cout << "0: rho= " << rho << std::endl;


	double loopMax = MIN (m_memory, m_iteration);
	container_type a (loopMax, 0);
	
	container_type q;
	q.reserve (force.size ());
	for (container_type::const_iterator it = force.begin (), itEnd = force.end (); it != itEnd; ++it)
	    q.push_back (-(*it));
	

	for (int i = loopMax-1, iEnd = -1; i > iEnd; --i)
	{
	    
	    a.at (i) = rho.at (i) * dot (m_s.at (i).begin (), m_s.at (i).end (), q.begin (), q.end ());
	    for (container_type::iterator itQ = q.begin (), itQEnd = q.end (),
		     itY = m_y.at(i).begin()/*, itYEnd=m_y.at(i).end()*/; itQ != itQEnd; ++itQ, ++itY)
	    {
		*itQ -= a.at (i) * (*itY);
	    }
	}
//	std::cout << "0: a= " << a << std::endl;
//	std::cout << "1: q= " << q << std::endl;

	container_type z;
	z.reserve (q.size ());
	for (container_type::iterator itQ = q.begin (), itQEnd = q.end ()
		 ; itQ != itQEnd; ++itQ)
	{
	    z.push_back ((*itQ) * H0);
	}

//	std::cout << "1: z= " << z << std::endl;


	for (int i = 0, iEnd = loopMax; i < iEnd; ++i)
	{
	    double b = rho.at (i) * dot (m_y.at (i).begin (), m_y.at (i).end (), z.begin (), z.end ());
	    for (container_type::iterator 
		     itZ = z.begin (), itZEnd = z.end (),
		     itS = m_s.at(i).begin ()/*, itSEnd=m_s.at(i).end()*/; 
		 itZ != itZEnd; ++itZ, ++itS)
	    {
		(*itZ) += (*itS) * (a.at(i) - b);
	    }
	}

//	std::cout << "2: z= " << z << std::endl;

	container_type dr;
	dr.reserve (z.size ());
	for (container_type::iterator itZ = z.begin (), itZEnd = z.end (); itZ != itZEnd; ++itZ)
	{
	    dr.push_back (-(*itZ));
	}

	determineStep (dr, m_damping);

	m_r0.assign (positions.begin (), positions.end ());
	m_f0.assign (force.begin (), force.end ());

	    
	for (container_type::iterator 
		 itP = positions.begin (), itPEnd = positions.end (),
		 itDR = dr.begin ()/*, itDREnd = dr.end ()*/; itP != itPEnd; ++itP, ++itDR)
	{
	    (*itP) += (*itDR);
	}

	++m_iteration;
	return true;
    }


    void determineStep (container_type& dr, double damping)
    {
	double steplength = 0.0;
	int count = 0;
	for (container_type::iterator itDR = dr.begin (), itDREnd = dr.end (); itDR != itDREnd; ++itDR)
	{
	    steplength += (*itDR)*(*itDR);
	    ++count;
	}
//	steplength /= count;
	steplength = sqrt (steplength);
	if (steplength >= m_maxStep)
	{
	    for (container_type::iterator itDR = dr.begin (), itDREnd = dr.end (); itDR != itDREnd; ++itDR)
	    {
		(*itDR) *= m_maxStep / steplength;
	    }
	}

	for (container_type::iterator itDR = dr.begin (), itDREnd = dr.end (); itDR != itDREnd; ++itDR)
	{
	    (*itDR) *= damping;
	}

    }


    void update (container_type& positions, container_type& forces, container_type& r0, container_type& f0)
    {
	if (m_iteration > 0)
	{
	    m_s.push_back (container_type ());
	    container_type& s0 = m_s.back ();
	    s0.reserve (positions.size ());
	    for (container_type::iterator 
		     itP = positions.begin (), itPEnd = positions.end (),
		     itR0 = r0.begin ()/*, itR0End = r0.end ()*/; itP != itPEnd; ++itP, ++itR0)
	    {
		s0.push_back ((*itP) - (*itR0));
	    }

	    m_y.push_back (container_type ());
	    container_type& y0 = m_y.back ();
	    y0.reserve (positions.size ());
	    for (container_type::iterator 
		     itF = forces.begin (), itFEnd = forces.end (),
		     itF0 = f0.begin ()/*, itF0End = f0.end ()*/; itF != itFEnd; ++itF, ++itF0)
	    {
		y0.push_back ((*itF0) - (*itF));
	    }
	    
	    double rho0 = 1.0 / dot (y0.begin (), y0.end (), s0.begin (), s0.end ());
	    m_rho.push_back (rho0);
	}
	
	if (m_iteration > m_memory)
	{
	    m_s.erase (m_s.begin ());
	    m_y.erase (m_y.begin ());
	    m_rho.erase (m_rho.begin ());
	}
    }




private:
    double m_maxStep;
    size_t m_memory;
    double m_damping;
    //double m_alpha;
    double m_minForce;

    double m_H0;

    size_t m_iteration;

    std::vector<container_type> m_s;
    std::vector<container_type> m_y;
    container_type m_rho;
    container_type m_r0;
    container_type m_f0;
};

