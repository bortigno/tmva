#include <iostream>
#include <vector>
#include <future>
#include <algorithm>


template <typename A, typename B, typename C>
int fncT (A& constVal, const B& v, const C& vec)
{
    std::cout << "[" << constVal << "," << v << "]";
    std::for_each (begin (vec), end (vec), [](int v){ std::cout << v; } );
    std::cout << std::flush;
    return v;
}


int fnc (int& constVal, const int& v, const std::vector<int>& vec)
{
    std::cout << "[" << constVal << "," << v << "]";
    std::for_each (begin (vec), end (vec), [](int v){ std::cout << v; } );
    std::cout << std::flush;
    return v;
}


int main ()
{
    int constVal = 333;
    std::vector<int> vec = {1,2,3};
    std::cout << "START ----" << std::endl;
    std::vector<std::future<int> > futures;
    for (size_t i = 0; i < 10; ++i)
    {
	futures.push_back (
	    std::async([&constVal, i, vec]() mutable { return fncT (constVal, i, vec); })
	    );
//        futures.push_back (std::async (std::launch::async, fncT, std::ref(constVal), i, vec));
//        futures.push_back (std::async (std::launch::deferred, fnc, i));
    }
    
//    std::cout << "futures --- " << std::endl;
    for (auto& f : futures)
    {
        std::cout << "{" << f.get () << "}" << std::flush;
    }
    std::cout << std::endl;
    return 1;
}

