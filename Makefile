.PHONY: all glm portable test check cuda-test clean

all glm portable test check cuda-test clean:
	$(MAKE) -C c $@
