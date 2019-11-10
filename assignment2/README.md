# Sentiment Analysis

The code enclosed in this folder is based on a Python script provided as a base material
for completing a sentiment analysis task on two datasets: Rotten Tomatoes movie reviews
and Nokia product reviews.

Compared to the original version, the code has been rearranged and extended.

### Rearrangement and refactoring
In the original version, the code was a single Python script. This has been divided 
into four smaller files. Three of them contain functions defined for model training
and testing (bayes.py and rule_based.py) as well as auxiliary functions
(aux_functions.py) The fourth one is the main execution part of the original script,
rewritten as a Jupyter Notebook (sentiments.ipynb).

The code was also slightly refactored to extract some repeated fragments 
as new functions.

### Extending the original functionalities
The code was developed to cover the tasks specified in the assignment instructions.

The main extension was new implementation of the rule-based system (rule_based_new.py),
which still significantly relies on the original (albeit simple) version.
The application of the new algorithm to the data was appended at the end of
the main file (sentiments.ipynb).

Another development was implementation of n-grams (for any n) alongside with
unigrams and bigrams and providing a simple way (a single constant in the main file)
to control the type of n-grams investigated.

### Data files
The data files needed for execution of the code have been moved to a subfolder,
which is also reflected in the code.

There are also two additional data files added to support the new implementation
of the rule-based model: but-words.txt and negation-words.txt.