
#include <algorithm>
#include <math.h>

#include "lbfgs.hpp"

typedef std::vector<double> container_type;

// void deltaFml (const container_type& positions, container_type& forces)
// {
//     forces.clear ();
//     forces.push_back (cos(positions.at (0)+0.0));
//     forces.push_back (sin(positions.at (1)-0.0));
// }

void deltaFml (const container_type& positions, container_type& forces)
{
    forces.clear ();
    forces.push_back (2*(positions.at (0)+20));
    forces.push_back (2*(positions.at (1)-43));
}


// def fml (pos):
//     return (pos[0]-2.0)**2+(pos[1]-4.0)**2;

// def deltaFml (pos):
//     return np.array([2*(pos[0]+20.0), 2*(pos[1]-43.0)])




int main ()
{
    LBFGS lb;

    container_type positions;
    positions.push_back (5.0);
    positions.push_back (8.0);
    container_type forces;

    for (size_t i = 0, iEnd = 5000; i < iEnd; ++i)
    {
	deltaFml (positions, forces);
	lb.step (positions, forces);
	std::cout << i << "   pos= " << positions << "   forces= " << forces << std::endl;

	double maxForce = fabs ((*std::max_element (forces.begin (), forces.end ())));
	if (maxForce < 0.0001)
	    break;
    }
    return 0;
}




