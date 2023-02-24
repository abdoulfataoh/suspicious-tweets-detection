help:
	@echo "make mypy	    - Checks static typing with mypy"
	@echo "make flake8          - Checks flake8"
	@echo "make test            - Runs the test suite"
	

mypy:
	mypy src train_test.py
flake8:
	flake8 src train_test.py

test:
	pytest tests