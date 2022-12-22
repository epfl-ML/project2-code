# Shedding Light on Mouse Sleep: Automated Sleep Phase Detection using Machine Learning

In this report, we use a dataset recorded by the Center for Integrative Genomics at UNIL to infer the sleep phase of
mice from miscellaneous genetic strains and evaluate the performance of our approach in this classification problem. We
use Electroencephalogram (EEG) and Electromyogram (EMG) signals to differentiate between wake, _NREM_ and _REM_ sleep
phases via random forests and neural networks. Our highest classification accuracy for a single individual, 93%, is
obtained using a random forest model, whereas our highest classification precision for a cross-individual model is 95%.

## Report

The report is available [here](report.pdf).

## Code structure

The project is structured as follows:

- the `data/` folder contains information to obtain the dataset;
- the `tools/` folder contains two scripts to preprocess the `.smo` files and export them as `.csv` files;
- the `experiments/` folder contains the code used to train the models and run the experiments;
- the `main.ipynb`

### Dependencies and requirements

This project uses the latest version of Python 3, and the Anaconda distribution is required to run the code. You
can install Anaconda from [here](https://www.anaconda.com/products/distribution).

Additionally, you will need to install the TensorFlow library. You can do so by running the following command in
your terminal:

```bash
conda install tensorflow
```

Here's an overview of the code structure:

```text
├── data
│   ├── README.md                               # Instructions to obtain the full dataset
├── experiments
│   ├── README.md
│   ├── lib                                     # Library of functions used in the experiments
│   │   ├── __init__.py 
│   │   ├── breeds.py
│   │   ├── features.py
│   │   ├── models.py
│   │   ├── plots.py
│   ├── 0. Data exploration.ipynb               # Data exploration
│   ├── 1. Single individual experiment.ipynb   # First experiment
│   ├── 2. Single mouse line experiment.ipynb   # Second experiment
│   ├── 3. Cross-line experiment.ipynb          # Third experiment
│   ├── a. Model exploration.ipynb
├── tools
│   ├── README.md
│   ├── data-extraction-kotlin                  # Main pre-processing script
│   ├── data-extraction-python
├── main.ipynb
├── report.pdf                                  # The project report
├── README.md
```

## Authors

+ Matthieu Burguburu matthieu.burguburu@epfl.ch
+ Lars Barmettler lars.barmettler@epfl.ch
+ Alexandre Piveteau alexandre.piveteau@epfl.ch
