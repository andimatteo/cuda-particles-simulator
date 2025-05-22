CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 -fopenmp
DEBUG_FLAGS = -Wall -Wextra -O3 -std=c++17 -DDEBUG -fopenmp

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

test: all
	./$(TARGET) 0 < low.txt > output.txt

sequential: all
	./$(TARGET) 0 < high.txt > output.txt

parallel: all
	./$(TARGET) 1 < high.txt > output.txt

chunk: all
	./$(TARGET) 2 < high.txt > output.txt