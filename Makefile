
SUITE = tests_*

test:
	python -m unittest tests/$(SUITE) -v


.PHONY: test
