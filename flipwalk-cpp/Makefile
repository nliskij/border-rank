CXXFLAGS = -std=c++23 -O3
SRC = flipwalk.cpp

M ?= 4
N ?= 4
P ?= 4

TARGET = bin/flipwalk$(M)$(N)$(P)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -DM=$(M) -DN=$(N) -DP=$(P) -o $@ $<
