.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = rjtk/dwglasso_cweeds
PROJECT_NAME = dwglasso_cweeds
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pip install -r requirements.txt

# These .mklog files are a hack for make to keep track of
# what scripts have been run.  I feel like make isn't
# really the right tool for this job.  I need something
# to keep track of the DAG that ins't fundamentally build
# on creating and updating particular files.
data/interim/locations.mklog: src/data/load_locations.py
	$(PYTHON_INTERPRETER) src/data/load_locations.py
	touch data/interim/locations.mklog

data/interim/temperatures.mklog: data/interim/locations.pkl src/data/load_data.py
	$(PYTHON_INTERPRETER) src/data/load_data.py
	touch data/interim/temperatures.mklog

data/interim/interpolate.mklog: src/data/interpolate_data.py
	$(PYTHON_INTERPRETER) src/data/interpolate_data.py
	$touch data/interim/interpolate.mklog

data/interim/clean.mklog: src/data/clean_data.py data/interim/interim_data.hdf
	$(PYTHON_INTERPRETER) src/data/clean_data.py
	touch data/interim/clean.mklog

data/interim/sinusoid_regression.mklog: src/data/sinusoid_regression.py
	$(PYTHON_INTERPRETER) src/data/sinusoid_regression.py
	touch data/interim/sinregress.mklog

data/interim/filter_data.mklog: src/data/filter_data.py
	$(PYTHON_INTERPRETER) src/data/filter_data.py
	touch data/interim/filter_data.mklog

data/interim/final_dataset.mklog: src/data/final_dataset.py
	$(PYTHON_INTERPRETER) src/data/final_dataset.py
	touch data/interim/final_dataset.mklog

data/interim/covars.mklog: data/interim/interim_data.hdf src/data/calculate_covars.py
	$(PYTHON_INTERPRETER) src/data/calculate_covars.py
	touch data/interim/covars.mklog

data/interim/dwglasso.mklog: src/models/dwglasso.py
	$(PYTHON_INTERPRETER) src/models/dwglasso.py
	touch data/interim/dwglasso.mklog

data/interim/plots.mklog: src/models/plot_results.py
	$(PYTHON_INTERPRETER) src/models/plot_results.py
	touch data/interim/plots.mklog


## Read location data
read_locations: data/interim/locations.mklog

## Read temperature data
read_temperatures: data/interim/temperatures.mklog

## Interpolate missing temperature data
interpolate_temperatures: data/interim/interpolate.mklog

## Clean temperature data and produce dT feature
clean_temperatures: data/interim/clean.mklog

## Sinusoidal regression
sinusoid_regression: data/interim/sinusoid_regression.mklog

## Filter data
filter_data: data/interim/filter_data.mklog

## Create final data set
final_dataset: data/interim/final_dataset.mklog

## Calculate covariances
calculate_covars: data/interim/covars.mklog

## Test DWGLASSO
test_dwglasso: data/interim/dwglasso.mklog

## Plot Results
plot_results: data/interim/plots.mklog

# ALL data
#all_data: read_locations read_temperatures interpolate_temperatures clean_temperatures final_data

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Upload Data to S3
sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/

## Download Data from S3
sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data --debug

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.5
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
