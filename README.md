# MLE-Image-Classification
Meta-learning ensemble framework for image classification with convolutional neural networks


## Purpose
This project creates an ensemble of CNN models for the MINST, Fashion-MNIST, and CIFAR10 dataset. The predictions from the ensemble are evaluated by a CNN super-learner and a weight matrix optimized by a genetic algorithm.


A special case is used for binary classification where a MLP detects errors in the aggregation of prediction matrices and flips the final predictions accordingly.

All base learners in the ensemble are saved to disk along with the prediction matrices and final predictions from the GA and CNN meta learners.

---

## Set-up
### Set up a virtual environment ([source](https://docs.python.org/3/tutorial/venv.html#virtual-environments-and-packages))
*A virtual environment, a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages.*

**Unix/MacOS**
* `python3 -m pip install --user virtualenv` - if you don't have virtualenv installed
* `python3 -m venv venv` - Create a virtual environment
* `source venv/bin/activate` - Activate virtual environment
* `python3 -m pip install -r requirements.txt ` - Install required libraries for project

**Windows**
* `py -m pip install --user virtualenv` - if you don't have virtualenv installed
* `py -m venv venv` - Create a virtual environment
* `.\venv\Scripts\activate` - Activate virtual environment
* `py -m pip install -r requirements.txt ` - Install required libraries for project

---

## Config settings
The project uses `dafault.ini` to set configurations for dataset, ensemble and GA prediction. The following table shows the allowed values for the configurations:

| Section  | Setting              | Allowed values                                                              | Description                                                                                                                        |
|----------|----------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| General  | Dataset              | any one of MNIST, MNIST_BIN, F_MNIST, F_MNIST_BIN, CIFAR10, CIFAR10_BIN     | Dataset to use                                                                                                                     |
| General  | ValidationProportion | decimal number in range [0.0, 1.0]. Recommended 0.2                         | Proportion of training data to use to train meta-learners                                                                          |
| Ensemble | Type                 | full or stacked                                                             | full: no meta-learner and full training set is used to train ensemble; stacked: validation data is left for training meta-learners |
| Ensemble | NumBaseLearners      | Any positive integer > 2. Recommended a value above 30                      | Number of base learners in ensemble                                                                                                |
| Ensemble | UseDynamicLoss       | True or False                                                               | Boolean value specifying whether or not to use dynamic loss to train base learners in the ensemble                                 |
| Ensemble | DynamicLossThreshold | decimal number in range [0.0, 1.0]. Recommended a value between 0.6 and 0.9 | Threshold to use for dynamic loss                                                                                                  |
| GA       | GAGenerations        | Any positive integer. Recommended a value above 150                         | Number of generations to run GA meta-learner for                                                                                   |
| GA       | GAPopulation         | Any positive integer. Recommended a value above 100                         | Population to maintain in GA optimizer                                                                                             |