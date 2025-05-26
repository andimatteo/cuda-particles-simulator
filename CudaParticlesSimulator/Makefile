PATH := $(PATH):/usr/local/cuda-12.9/bin

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
OUT_DIR := out
LOAD_DIR := ..
GEN_DIR := ../utils

LOADLEVELS := low medium high
THREADSBLOCK := 32 64 128 256 512 1024
VERSIONS := 0 1 3
#VERSIONS := 0 1 2 3
TILE_WIDTH := 200 300 400

EPOCHS_high := 10
EPOCHS_medium := 10
EPOCHS_low := 10
PARTICLENUM_high := 20000
PARTICLENUM_medium := 5000
PARTICLENUM_low := 100

NVCC := nvcc
NVCCFLAGS := -gencode arch=compute_61,code=sm_61 -O3 -Wno-deprecated-gpu-targets
# DEBUG_FLAGS = -Wall -Wextra -O3 -std=c++17 -DDEBUG -fopenmp -g
LDFLAGS += -I$(SRC_DIR)/include

SOURCES = $(wildcard $(SRC_DIR)/*.cu)
HEADERS = $(wildcard $(SRC_DIR)/include/*.cuh)
OBJECTS = $(wildcard $(OBJ_DIR)/*.o)
TARGETS := $(foreach level,$(LOADLEVELS), \
			$(foreach threads,$(THREADSBLOCK), \
				$(foreach version,$(VERSIONS), \
					$(foreach tile,$(TILE_WIDTH), \
						$(BIN_DIR)/main_$(level)_$(threads)_$(version)_$(tile)))))
TESTS 	= $(foreach level,$(LOADLEVELS), \
			$(foreach threads,$(THREADSBLOCK), \
				$(foreach version,$(VERSIONS), \
					$(foreach tile,$(TILE_WIDTH), \
						test_$(level)_$(threads)_$(version)_$(tile)))))
OUTPUT_FILES = $(wildcard $(OUT_DIR)/output*.txt)
ERROR_FILES = $(wildcard $(OUT_DIR)/error*.txt)
TIME_FILES = $(wildcard $(OUT_DIR)/time*.txt)


.PHONY: all clean test

all: $(TARGETS)

$(BIN_DIR)/main_%: $(SOURCES) $(HEADERS)
# Retrieve parameters from target name: main_<level>_<threads>_<version>_<tile>
	$(eval PARAMS_ := $(patsubst $(BIN_DIR)/main_%,%,$@))
	$(eval PARAMS = $(subst _, ,$(PARAMS_)))
	$(eval level := $(word 1,$(PARAMS)))
	$(eval threads := $(word 2,$(PARAMS)))
	$(eval version := $(word 3,$(PARAMS)))
	$(eval tile := $(word 4,$(PARAMS)))

	$(NVCC) $(NVCCFLAGS) -o $@ $(SOURCES) $(LDFLAGS) -DPARTICLE_NUM=$(PARTICLENUM_$(level)) -DDURATION=$(EPOCHS_$(level)) -DTHREADS_PER_BLOCK=$(threads) -DVERSION=$(version) -DTILE_WIDTH=$(tile)

clean:
	rm -f $(OBJECTS) $(TARGETS) $(OUTPUT_FILES) $(ERROR_FILES) $(TIME_FILES)

test_%: $(BIN_DIR)/main_%
# Retrieve parameters from target name: main_<level>_<threads>_<version>_<tile>
	$(eval PARAMS_ := $(patsubst test_%,%,$@))
	$(eval PARAMS = $(subst _, ,$(PARAMS_)))
	$(eval level := $(word 1,$(PARAMS)))
	$(eval threads := $(word 2,$(PARAMS)))
	$(eval version := $(word 3,$(PARAMS)))
	$(eval tile := $(word 4,$(PARAMS)))

	./$< $(OUT_DIR)/time.txt < $(LOAD_DIR)/$(level).txt > $(OUT_DIR)/output_$(level)_$(threads)_$(version)_$(tile).txt 2> $(OUT_DIR)/error_$(level)_$(threads)_$(version)_$(tile).txt

test: $(TESTS)
	@echo "All tests completed."

# low medium high
generate_%: $(GEN_DIR)/generate_config.py
	python3 $< -e $(EPOCHS_$*) -p $(PARTICLENUM_$*) -o $(LOAD_DIR)/$*.txt