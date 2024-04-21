### Use full python implementation 
The notebook `main.ipynb` contains the full python implementation of the algorithms and the utils are in `/utils.py`. 
 
### Run the C++/ cython version 
The cpp files can be compiled and used as a 'C++' library. Alternatively, there is a 'cython3' binding. 
To compile the cpp version with `cython`: 

**Step 1** : Install the requirements with

`pip install -r requirements.txt`

**Step 2**: check the compiler settings in `setup.py` and run from the current directory 

 `python3 setup.py build_ext --inplace`
 
**Step 3** import and use as in the notebook file `cython/main.ipynb`