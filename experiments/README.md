# Experiments

This directory contains the code for the experiments in the paper. There are 3 main experiments, as well as an
exploratory analysis of the data.

- [`0. Data exploration`](0.%20Data%20exploration.ipynb) contains the exploratory analysis of the data;
- [`1. Single individual experiment.ipynb`](1.%20Single%20individual%20experiment.ipynb) looks at a single mouse and
  tries to predict its sleep stages over 3 days by learning over 1 day;
- [`2. Single mouse line experiment.ipynb`](2.%20Single%20mouse%20line%20experiment.ipynb) evaluates the performance of
  the training of the models over multiple individuals of the same line; and
- [`3. Cross-line experiment.ipynb`](3.%20Cross-line%20experiment.ipynb) evaluates the performance of the models when
  training over multiple lines of mice.

## Models

Each experiment is provided with a pre-trained neural network model. **By default, running the notebooks will retrain the model, but not save it nor load it.**
Therefore, you should uncomment the indicated lines cells if you wish to load the existing model.
