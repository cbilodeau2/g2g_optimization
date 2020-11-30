# g2g_optimization

This repository implements the Hierarchical Graph Neural Network (https://arxiv.org/pdf/2002.03230.pdf) for graph-to-graph translation (https://arxiv.org/pdf/1812.01070.pdf) with and without iterative stochastic augmentation (https://arxiv.org/pdf/2002.04720.pdf). These approaches can be used to 1) train a model to translate a starting molecule into a related molecule optimized with respect to a target property and 2) generate a library of molecules for identifying a molecule optimized with respect to a target property. In addition to the previously published methods, this repository also includes a molecule pairing algorithm and support for incorporating additional secondary constraints. This repository is intended to be used along with ChemProp for molecular property prediction (https://github.com/chemprop).

**Previous Repository**:

**Full Documentation**:

# Usage/Examples:

## Training:
In order to train a model, you must provide a csv file containing two columns. The first column must contained smiles and have the header "SMILES", and the second column must have labels for each molecule target header specified in the argument file. The model will aim to maximize the quantity in the target column. To run the example in the tests folder:

```
mkdir save_dir
python train.py --data_path test/aqsol_short --args_file tests/aqsol_short/input.dat --save_dir save_dir
```
Model/workflow parameters can be adjusted in the argument file, tests/aqsol_short/input.dat.

### Training with iterations:
In addition to simply training the model, you can use the model to translate every molecule in your starting dataset to iteratively grow your dataset. This can be helpful for improving model performance and for searching for generally optimal molecules. To run multiple iterations of the example in the tests folder:

```
mkdir save_dir
python train.py --data_path tests/aqsol_short --args_file test/aqsol_short/input.dat --save_dir save_dir --n_iterations 2
```
## Model Evaluation:
### Translate a given molecule list:
After training a graph-to-graph translator model, you can input a new list of molecules to produce a set of improved molecules. Using the output of the training example:
```
python evaluate_test.py --test tests/aqsol_short/test.csv --vocab save_dir/inputs/vocab.txt --model save_dir/test/models/model.10 --args_file tests/aqsol_short/input.dat --output_file mols_decoded.csv
```

### Evaluate model performance using a given test set:
Alternatively, in addition to translating a given test set, you can calculate model performance statistics by referencing a previously trained ChemProp model. Using the output of the training example (note: you need a separate ChemProp model to run this):

```
python evaluate_test_chemprop.py --test tests/aqsol_short/test.csv --model model.10 --checkpoint_path save_dir --args_file tests/aqsol_short/input.dat --fold_path <path to ChemProp model> --chemprop_path <path to ChemProp predict.py script>
```

### Evaluate model performance based on subsets of input dataset:
Alternatively, it may be useful to calculate model performance generally on subsets of the training set. evaluate_chemprop.py can be used to generate 5 test sets that are subsets of the training set: the top 25 molecules with respect to the target, the bottom 25 molecules with respect to the target, and molecules falling around the 25th, 50th, and 75th percentiles of the target distribution. evaluate_chemprop.py then generates separate statistitcs for each of these test sets. Using the output of the training example (note: you need a separate ChemProp model to run this):

```
python evaluate_chemprop.py --model model.10 --checkpoint_path save_dir --args_file tests/aqsol_short/input.dat --fold_path <path to ChemProp model> --chemprop_path <path to ChemProp predict.py script>
```


