.PHONY: all glm portable test check cuda-test clean install uninstall coli-go

all glm portable test check cuda-test clean install uninstall coli-go:
	$(MAKE) -C c $@
