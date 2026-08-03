.PHONY: all glm portable test check cuda-test clean install uninstall

all glm portable test check cuda-test clean install uninstall:
	$(MAKE) -C c $@