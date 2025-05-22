CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++17
DEBUG_FLAGS = -Wall -Wextra -O3 -std=c++17 -DDEBUG
LDFLAGS = -fopenmp

SRC_DIR = src
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(SRC_DIR)/*.h)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = main

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS := $(DEBUG_FLAGS)
debug: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

run: all
	./$(TARGET) 0 < low.txt > output.txt

