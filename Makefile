CXX := g++
NVCC := nvcc
CXXSTD := c++17
CXXFLAGS := -std=$(CXXSTD) -Wall -Wextra -flto
NVCCFLAGS := -std=$(CXXSTD) -O3 --use_fast_math -Xcompiler "-fPIC"
# Override for specific GPU: make NVCC_ARCH=sm_120 (Blackwell), sm_89 (Ada), etc.
ifdef NVCC_ARCH
	#NVCCFLAGS += -arch=$(NVCC_ARCH)
endif

ifdef NEW_ENV
	CXXFLAGS += -DNEW_ENV
endif
LDFLAGS := -lpthread

# ONNX Runtime configuration
#ONNXRUNTIME_ROOT ?= /opt/onnxruntime-gpu
ONNXRUNTIME_ROOT ?= /opt/onnxruntime-1.23.2-gpu
ONNXRUNTIME_INCLUDE := $(ONNXRUNTIME_ROOT)/include
ONNXRUNTIME_LIB := $(ONNXRUNTIME_ROOT)/lib


PLATFORM := linux
LDFLAGS += -lpthread -L$(ONNXRUNTIME_LIB) -lonnxruntime -L/usr/local/cuda/lib64 -lcudart -flto 


# optimized flags based on benchmark results
RELEASE_FLAGS := -O3 -DNDEBUG -march=native -mtune=native -ffast-math -mavx2 -flto=auto
DEBUG_FLAGS := -g3 -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
PROFILE_FLAGS := -O2 -g -pg -DPROFILE

# directories
SRC_DIR := src
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
DEP_DIR := $(BUILD_DIR)/deps
BIN_DIR := $(BUILD_DIR)/bin
INCLUDE_DIR := src

# target executable
TARGET := generator
TARGET_PATH := $(BIN_DIR)/$(TARGET)
REGRESSION_TARGET := sampling_filtering_regression
REGRESSION_TARGET_PATH := $(BIN_DIR)/$(REGRESSION_TARGET)

# Source files
COMMON_SOURCES := $(shell find $(SRC_DIR) -name "*.cpp" -not -path "*/test*" -not -path "*/reference/*" -not -name "main.cpp")
CUDA_SOURCES := $(shell find $(SRC_DIR) -name "*.cu" -not -path "*/test*" -not -path "*/reference/*")
MAIN_SOURCE := $(SRC_DIR)/main.cpp
ALL_SOURCES := $(MAIN_SOURCE) $(COMMON_SOURCES)
REGRESSION_SOURCES := \
	$(SRC_DIR)/test/sampling_filtering_regression.cpp \
	$(SRC_DIR)/models/json_tag_dict.cpp \
	$(SRC_DIR)/tokenizer/Tokenizer.cpp \
	$(SRC_DIR)/evaluation/filtering_thread.cpp

# OBJ
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SOURCES))
CUDA_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CUDA_SOURCES))
DEPS := $(patsubst $(SRC_DIR)/%.cpp,$(DEP_DIR)/%.d,$(ALL_SOURCES))

# include
INCLUDES := -I$(INCLUDE_DIR) -I$(ONNXRUNTIME_INCLUDE)

# default target
.DEFAULT_GOAL := release

.PHONY: all release debug profile clean distclean test install help check-onnx sampling-filtering-regression

# Check ONNX Runtime installation
check-onnx:
	@echo "Checking ONNX Runtime installation..."
	@if [ ! -d "$(ONNXRUNTIME_ROOT)" ]; then \
		echo "Error: ONNX Runtime not found at $(ONNXRUNTIME_ROOT)"; \
		echo "Please install ONNX Runtime or set ONNXRUNTIME_ROOT environment variable"; \
		echo "Example: export ONNXRUNTIME_ROOT=/path/to/onnxruntime"; \
		exit 1; \
	fi
	@if [ ! -d "$(ONNXRUNTIME_INCLUDE)" ]; then \
		echo "Error: ONNX Runtime include directory not found: $(ONNXRUNTIME_INCLUDE)"; \
		exit 1; \
	fi
	@if [ ! -d "$(ONNXRUNTIME_LIB)" ]; then \
		echo "Error: ONNX Runtime library directory not found: $(ONNXRUNTIME_LIB)"; \
		exit 1; \
	fi
	@echo "ONNX Runtime found at: $(ONNXRUNTIME_ROOT)"
	@echo "  Include: $(ONNXRUNTIME_INCLUDE)"
	@echo "  Library: $(ONNXRUNTIME_LIB)"


# build targets
all: release

release: CXXFLAGS += $(RELEASE_FLAGS)

release: check-onnx $(TARGET_PATH)
	@echo "Release build complete: $(TARGET_PATH)"
	@mv $(TARGET_PATH) $(TARGET)

sampling-filtering-regression: $(REGRESSION_TARGET_PATH)
	@echo "Regression harness built: $(REGRESSION_TARGET_PATH)"

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: check-onnx $(TARGET_PATH)
	@echo "Debug build complete: $(TARGET_PATH)"

profile: CXXFLAGS += $(PROFILE_FLAGS)
profile: check-onnx $(TARGET_PATH)
	@echo "Profile build complete: $(TARGET_PATH)"

# main target linking
$(TARGET_PATH): $(OBJECTS) $(CUDA_OBJECTS) | $(BIN_DIR)
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJECTS) $(CUDA_OBJECTS) $(LDFLAGS) -o $@
	@echo "Build successful!"

$(REGRESSION_TARGET_PATH): $(REGRESSION_SOURCES) | $(BIN_DIR)
	@echo "Linking $(REGRESSION_TARGET)..."
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) $(INCLUDES) $(REGRESSION_SOURCES) -lpthread -o $@
	@echo "Regression harness build successful!"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR) $(DEP_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEP_DIR)/$*.d)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -MF $(DEP_DIR)/$*.d -c $< -o $@

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@echo "Compiling CUDA $<..."
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OBJ_DIR): | $(BUILD_DIR)
	@mkdir -p $(OBJ_DIR)

$(DEP_DIR): | $(BUILD_DIR)
	@mkdir -p $(DEP_DIR)

$(BIN_DIR): | $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

-include $(DEPS)

# install target
install: release
	@echo "Installing $(TARGET) to /usr/local/bin/"
	sudo cp $(TARGET_PATH) /usr/local/bin/$(TARGET)
	sudo chmod +x /usr/local/bin/$(TARGET)
	@echo "Installation complete!"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(DEP_DIR)
	rm -f $(TARGET_PATH)
	@echo "Clean complete!"
