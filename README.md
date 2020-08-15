# Dence_NN

###
###



## Project Description
*Note: This project was highly motivated by topics covered in [Deep Learning Specialisation](https://www.coursera.org/specializations/deep-learning?) by [Deeplearning.ai](https://www.deeplearning.ai/)*

#### Overview:
To develop a dence Neural Network (NN) library from scratch.

In this instance - a user can call the MyDenceNN class to build, train and test NN models. The use can specify the chosen Metrics, Optimisers Activations and Cost functions they want from a list oof existing function in MyDenceNN. The user can specify hoow deep they wish their NN too be aswell as howmany units they wish to have in each layer. The class also come equipt with many options for tunning hyperparameter.

**Files:**
* `idea_NN.ipymb`: Developing a basic idea.
* `prototype_NN.ipymb`: Testing basic version.
* `MyDenceNN.py`: The Neural Network Library.
* `test_NN.ipymb`: Testing the function or the library.

#### Insight:
* When building a model for a paticular task use linear regression as a bottom line (it saves time).
* Models can only ever be as good as the data they are trained on -- so finding good data is 80%+ of the work.

###
###



## Prerequisites

* [IDE](https://jupyter.org/install) - Jupyter Lab

* [Python:](https://www.python.org/downloads/) - Version 3.7

* [Numpy:](https://numpy.org/) - Latest version

* [Pandas:](https://pandas.pydata.org/) - Latest version

* [Matplotlib:](https://matplotlib.org/) - Latest version

* [H5py:](https://www.h5py.org/) - Latest version

* [Requierments Folder:]() - requiments.txt

For installation, run the following:
```
>>> pip install -r requirements.txt
```


# Implementation

```python
new_dence = MyDenceNN()
```

## Initialisation

The class requires the following peices of information to Initialise the model:
* `hiden_units`: the number of hidden units in each layer.
* `X` and `Y`: the input and output features.
* *Additionoal Features*: such as weighted initialisation.

```python
dence.add_inputs(X1)
dence.add_outputs(Y1)
```
or
```python
dence.add_data(X2, Y2)
```


```python
hidden_units = {
    "1" : 64,
    "2" : 8
}

activations = {
    "1" : "relu",
    "2" : "relu",
    "3" : "sigmoid"    
}
```

```python
dence.add_units(hidden_units)
```

```python
dence.build()
```

###
###

## Forward Propagation

```python
dence.forward_propagation()
```

###
###


## Back Propagation

```pythoon
dence.backward_propagation()
```

###
###

```python
new_dence = MyDenceNN()
```

```python
cost_fun = "logistic_regression"
alpha = 0.01
epochs = 2000
display=False
print_cost = True

## L2 Reg Needs Fixing
L2_reg = False
lambd = 0.01
keep_prob = 1
normalise = True

# Experiment with weighted initiation
weighted_init = False
```

```python
new_dence.solve(
                train_x, train_y, hidden_units, activations, 
                cost_fun, alpha, epochs, display, print_cost, 
                L2_reg, lambd, keep_prob, normalise, weighted_init
                )
```

###
###



## Looking Foward

###
###



## Authors

* **[Tobiloba Adeniyi](https://github.com/TobiAdeniyi)** - *Initial work* - [Dence_NN](https://github.com/TobiAdeniyi/Dence_NN)



## Acknowledgments

* [Deeplearning.ai](https://www.deeplearning.ai/).
* [Coursera](https://www.coursera.org/specializations/deep-learning?skipBrowseRedirect=true).
