# Random Forest
random forest is a model that you can use to make predictions of the classification of the eadibility of certain mushroom.
## auththor
the model builder:**Liming Jiang**.

email: Liming_Jiang@student.uml.edu
## quick start
* to implement a Decision Tree
  ```
  python path/to/main.py -m0 -t /path/to/training/file -e /path/to/testing/file
  ```
* to implement a Random Forest
  ```
  python path/to/main.py -m1 -t /path/to/training/file -e /path/to/testing/file -n 20
  ```
## data
### Sources:
1. Mushroom records drawn from The Audubon Society Field Guide to North American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred A. Knopf
2. Donor: Jeff Schlimmer (Jeffrey.Schlimmer@a.gp.cs.cmu.edu)
3. Date: 27 April 1987
### data information
1. 22 attributes;
2. 8124 observation totally, with 5694 in training file and 2430 in testing file;
3. Missing Attribute Values: 2480 of them (denoted by "?"), all for attribute 11.

## frequent asked questions
1. why is the result of Decision tree not constant given the same training and testing datasets?

Answer: Because the "deep bootstrapping" approach is being used in the tree_growth method of Decision Tree class. And some randomness is introduced.

2. What is "deep bootstrapping"?

Answer: In top-down decision tree building process, at each node of the tree, we sample (with replacement) 75% of the training records that arrive at that node.

## note
1. please use python 3+ to implement model files.
2. for more information, please refer to comments inside py. file.
