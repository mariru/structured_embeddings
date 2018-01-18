## Instructions for data preprocessing for structured Bernoulli embeddings.

Preprocess the data with 4 simple steps described below.

Depending on how large your data is, you should set aside ca. 20 minutes for these preprocessing steps. 

The goal of preprocessing is to truncate the vocabulary, remove all the words that are not in the vocabulary and then put the text into numpy arrays. Then we split the data into training validation and test set and compute statistics of the data that we need to use in the algorithm (e.g the number of words in each group, and the names of the groups).

These are the steps. They are described in detail below:

  0. Decide on filenames for groups
  1. Create vocabulary file and save text data in numpy arrays
  2. Subsample the data and split into training and testing
  3. Create `dat_stats.pkl`
  4. Generate negative samples for evaluation

An example dataset called `lorem_ipsum` is included to demonstrate the steps.

### Reqired Input

The following folder structure is required to run structured Bernoulli embeddings.

#### Before Preprocessing:
Assumption: Your text is in text files in the `dat/[dataset_name]/raw/` subfolder.

```
dat/
    [dataset_name]/
        raw/
            *.txt
```

#### After Preprocessing:

The preprocessing scripts will add the following folders and files 
```
dat/
    [dataset_name]/
        unigram.txt
        dat_stats.pkl
        raw/
            *.txt
            *.npy
        train/
            *.npy
        test/
            *.npy
            /neg
                *.npy
        valid/
            *.npy
            /neg
                *.npy
```            

The `train/`, `test/` and `valid/` folders will contain the `.npy` files with the data.
The file `unigram.txt` contains the vocabulary and the vocabulary counts and the file `dat_stats.pkl` contains a pickle object that holds a python dictionary with information about the data required to run the algorithm.

#### Example: Lorem Ipsum

This folder contains an example dataset. Text files are in `lorem_ipsum/raw/*.txt`.
Running the python scripts 

```
python step_1_count_words.py
python step_2_split_data.py
python step_3_create_data_stats.py
python step_4_negative_samples.py
```
without modification (and in numerical order) will prepare the dataset for fitting structured embeddings.
  
 ### 0. Decide on filenames for the groups
 
Your text files are in the `raw/` subfolder. Decide now how many groups you want and give each group its own name.
Group names cannot contain underscores as the underscore separates the group name from the rest of the file name.

For example, your folder could contain the file names:

```
Group1_rest_of_filename_xxx.txt
Group1_rest_of_filename_xyz.txt
Group1_rest_of_filename_yyy.txt
...
Group2_rest_of_filename_xxx.txt
Group2_rest_of_filename_xyz.txt
...
...
GroupN_rest_of_filename_zzz.txt
GroupN_rest_of_filename_yyy.txt
```

Save the list of group names `[Group1, Group2, Group3, ..., GroupN]`. You will need it in step 3.

Tip: Instead of having many short files. You might want to have few longer files in each group. Simply concatenate multiple short files into longer text files.

#### Example: Lorem Ipsum

For the lorem ipsum example, the raw filenames (in `lorem_ipsum/raw/`) either start with `A_` or `B_`. This means that there are 2 groups with names `'A'` and `'B'`.


### 1. Create vocabulary file and save text data in numpy arrays

In this step you will run the script `step_1_count_words.py`.
The script counts distinct words and truncates the resulting vocabulary.
Then each word is replaced whith its index in the vocabulary and the resulting numpy arrays are saved.

#### 1.1 Modify step_1_count_words.py

Go into this file and change `dataset_name` to the name of the folder in which the data is.
It should be the subfolder unted `dat/` (as specified above your data is in `dat/[dataset_name]/raw/`).
The current default dataset name `lorem_ipsum`.
Also modify the vocabulary size you want to use a good size for English corpora is 'V = 10000'.


#### 1.3 Run step_1_count_words.py
Assuming you are in `dat/` simply run 

```
   python step_1_count_words.py
```

Tip: In this script data preprocesing is handled. Punctuation is removed, line-breaks are handled and everything is lower cased. Depending on your dataset, additional or differnt preprocessing steps might be required. We also recommend extracting bigrams and adding them to the vocabulary.

### 2. Subsample the data and split into train test and validation split

As in step 1.1 open the file `step_2_split_data.py` and change the data_set name.
Then form `dat/` run
```
    python step_2_split_data.py
```

### 3. Create data statistics.

In this step, you simply have to run `step_3_create_data_stats.py`.
Again, go into the file and change the `dataset_name`. 
Then add the list of group names from step 0 under `states`
(e.g. `states = [Group1, Group2, Group3, ..., GroupN]`).

Then run
```
    python step_3_create_data_stats.py
```

### 4. Generate negative samples.

In this step, you will generate negative samples to be used during model evaluation. By drawing negative samples before running any code, you ensure that all methods are evaluated on the same set of negative examples.

Open `step_4_negative_samples.py` and change the `dataset_name`. 
Then run
```
    python step_4_negative_samples.py

```

You will get an error during training if you train with more negative samples then you generated in step 4.
