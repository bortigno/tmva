
CC=g++
#CC=clang++
CFLAGS=-Wl,--no-as-needed -Wall -std=c++11 -O4 -ggdb -pthread
LDFLAGS=-ggdb -lpthread
SOURCES=nn_main.cpp neuralNet.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=nn_mainExe
LIBS = -lstdc++ -lm

all: $(EXECUTABLE) #clangtest

clangtest: clangtest.cpp
	$(CC) $(CFLAGS) clangtest.cpp -o clangtest 

threadtest: test_thread.cpp
	$(CC) $(CFLAGS) test_thread.cpp -lpthread -o threadtest

gnuplot_i.o:	../gnuplotWrapper/gnuplot_i.hpp

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@ 

clean:
	rm nn_1 nn_mainExe *.o clangtest *~ $(EXECUTABLE) threadtest

