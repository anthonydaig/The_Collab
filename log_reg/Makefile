CXX=g++
CPPFLAGS= -std=c++11 -I/Users/anthonygagliardi/Desktop/The_BroccolLaboration/eigen-eigen-b30b87236a1b/ -Wall -pedantic -g3
all: log_reg clean

log_reg: log_reg.o
	$(CXX) $(CPPFLAGS) -o $@ $^

log_reg.o: log_reg.cpp
clean:
	rm -f log_reg.o
