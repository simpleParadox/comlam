# comlam
Code for the CoMLaM project

Doing some sanity checks now.

## Install python packages and run code.

1. First create a conda environment or virtual env.
```
conda create -n comlam
```

2. Activate the environment
```
conda activate comlam
```

3. Install the following packages (run each command in sequential order).
```
pip install -U scikit-learn
pip install pandas
pip install gensim
pip install nilearn
pip install matplotlib
pip install argparse
pip install scipy
pip install seaborn
pip install mne
```


4. Run code (make sure the anaconda environment is activate before running this)
```
python main.py
```
NOTE: If you get an error saying 'a' package is not found/installed, then run the following command
```
pip install <package_name> # (e.g., pip install numpy)
```

## How to change arguments, currently, the arguments have to be changed manually by going into the main.py file (support for command line arguments will be added soon).
Find the following set of lines in main.py. It should be somewhere near the end of the file (see the line numbers). Modify the arguments accordingly.
![image](https://user-images.githubusercontent.com/17592815/227997117-40ec8064-2e44-4240-a119-49346aa86bbc.png)




