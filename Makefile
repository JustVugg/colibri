.PHONY: all glm deepseek-v4 portable test check cuda-test clean install uninstall

all glm deepseek-v4 portable test check cuda-test clean install uninstall:
	$(MAKE) -C c $@