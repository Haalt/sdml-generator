CXX := g++
CXXSTD := c++17
CXXFLAGS := -std=$(CXXSTD) -Wall -Wextra -flto
LDFLAGS := -lpthread

# ONNX Runtime configuration
ONNXRUNTIME_ROOT ?= /opt/onnxruntime-gpu
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

# Source files
COMMON_SOURCES := $(shell find $(SRC_DIR) -name "*.cpp" -not -path "*/test*" -not -path "*/reference/*" -not -name "main.cpp")
MAIN_SOURCE := $(SRC_DIR)/main.cpp
ALL_SOURCES := $(MAIN_SOURCE) $(COMMON_SOURCES)

# OBJ
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SOURCES))
DEPS := $(patsubst $(SRC_DIR)/%.cpp,$(DEP_DIR)/%.d,$(ALL_SOURCES))

# include
INCLUDES := -I$(INCLUDE_DIR) -I$(ONNXRUNTIME_INCLUDE)

# default target
.DEFAULT_GOAL := release

.PHONY: all release debug profile clean distclean test install help check-onnx

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

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: check-onnx $(TARGET_PATH)
	@echo "Debug build complete: $(TARGET_PATH)"

profile: CXXFLAGS += $(PROFILE_FLAGS)
profile: check-onnx $(TARGET_PATH)
	@echo "Profile build complete: $(TARGET_PATH)"

# main target linking
$(TARGET_PATH): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "Build successful!"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR) $(DEP_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEP_DIR)/$*.d)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -MF $(DEP_DIR)/$*.d -c $< -o $@

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
