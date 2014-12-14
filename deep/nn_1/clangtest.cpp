
#include <vector>
#include <cassert>
#include <iostream>

class A
{
public:

    typedef typename std::vector<double>::iterator iterator;


    A (iterator it)
    : m_it (it)
    {
    }

    iterator get () { return m_it; }

private:
    typename std::vector<double>::iterator m_it;
};


int main ()
{
    std::vector<double> m_values (3, 1.1);
    A a (begin (m_values));
    assert (m_values.at (0) == *a.get ());
    std::cout << "success" << std::endl;
}

