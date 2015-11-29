CXX=g++
CPPFLAGS= -std=c++11 -I/Users/anthonygagliardi/Desktop/The_BroccolLaboration/eigen-eigen-b30b87236a1b/ -Wall -pedantic -g3
all: ann clean

ann: ann.o
	$(CXX) $(CPPFLAGS) -o $@ $^

ann.o: ann.cpp
clean:
	rm -f ann.o
