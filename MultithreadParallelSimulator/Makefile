CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 -fopenmp
DEBUG_FLAGS = -Wall -Wextra -O3 -std=c++17 -DDEBUG -fopenmp -g
LDFLAGS += -Iinclude

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LOAD_DIR = ..
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(SRC_DIR)/include/*.h)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
TARGET = $(BIN_DIR)/main
OUTPUT_FILE = out/output.txt

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS := $(DEBUG_FLAGS)
debug: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET) $(OUTPUT_FILE)

test: all
	./$(TARGET) 0 < $(LOAD_DIR)/low.txt > $(OUTPUT_FILE)

sequential: all
	./$(TARGET) 0 < $(LOAD_DIR)/high.txt > $(OUTPUT_FILE)

parallel: all
	./$(TARGET) 1 < $(LOAD_DIR)/high.txt > $(OUTPUT_FILE)

chunk: all
	./$(TARGET) 2 < $(LOAD_DIR)/high.txt > $(OUTPUT_FILE)