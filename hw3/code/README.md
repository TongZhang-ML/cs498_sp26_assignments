
## Implementation

The students are required to implement all functions start with doc
string "Implement". Remove body of such functions in the solution code
and ask the students to implement. 


## sent


### Solution code

`sent_solution.py`

### Data files  

`sent_gendata.py` generates the data files.

data/train.txt data/val.txt which will be visiable to students
data/test.txt will be hide from students

###  Auograder

The hidden test set should be evaluated on Gradescope.
The hidden test results are not disclosed to students until the end.

sent_solution.py will save ckpt to outputs/sent_model.pt
sent_solution.py --mode test will read ckpt from outputs/sent_model.pt and evaluate on data/train.txt data/val.txt data/test.txt


## textgen

### Solution code

`textgen_solution.py`

### Data files

`textgen_gendata.py` generates data/train.txt data/val.txt

