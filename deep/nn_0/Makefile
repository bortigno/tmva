
CC=gcc
#CC=clang++
CFLAGS=-Wall -std=c++11 -O0 -ggdb
LDFLAGS=-std=c++11 -O0 -ggdb
SOURCES=NNdemo.cpp neuralNet.cpp ../mnist/readnist.cpp  
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=NNdemoExe
LIBS = -lstdc++ -lm

all: $(EXECUTABLE) test_move nndemo_together

gnuplot_i.o:	../gnuplotWrapper/gnuplot_i.hpp

test_move: test_move.cpp
	$(CC) $(CFLAGS) test_move.cpp -o test_move $(LIBS)

test_move_linked: test_move.o
	$(CC) $(LDFLAGS) test_move.o -o test_move_linked $(LIBS)

nndemo_together: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o nndemo_together $(LIBS)

nndemo_test: NNdemo.o ../mnist/readnist.o neuralNet.o
	$(CC) $(LDFLAGS) NNdemo.o ../mnist/readnist.o neuralNet.o -o $@ $(LIBS)


$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

.cpp.o: 
	$(CC) $(CFLAGS) -c $< -o $@ 

clean:
	rm NNdemo NNdemoExe *.o test_move nndemo_together test_move_linked nndemo_test
