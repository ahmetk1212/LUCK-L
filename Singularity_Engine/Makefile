CXX = g++
CXXFLAGS = -O2 -std=c++17

SRCS = engine/tensor.cpp engine/layer.cpp engine/model.cpp engine/tokenizer.cpp engine/rwkv.cpp engine/main.cpp

singularity: $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o singularity

clean:
	rm -f singularity

.PHONY: clean
