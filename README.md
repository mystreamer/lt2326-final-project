# ML4NLPA: Final Project
Machine learning for statistical NLP: Advanced

This is the code for the project *GNN-based Dialogical Argument Mining: A First Step*. The project is an attempt of performing the task of **predicting argument structures grounded in dialogue**.

The underlying dataset is **QT-30** by Hautli-Janisz et al. (2022).

We frame the task as a node prediction problem, by harnessing an existing approach for link prediction proposed by Binder et al. (2024).

More background information on this repository can be found in the accompanying paper in the `paper` folder.

## Running the project 

<!-- Since we use as initial pre-processing an external method, we provide that data out-of-the-box as a parquet file. This also makes running the script on **mltgpu** more friendly, since we can rely on a much simpler virtual environment. In the notebook.py file detailed steps are however provided for replicating the experiments using a different dataset for example, that follows the same annotation standard. -->

To make the script running we will need some lightweight modules on top of the usual ML-stack (pytorch, etc.), hence I setup a virtual environment on **mltgpu**. In case the environment got wiped because it is installed on scratch use the below command to set it up again.

First, `cd code` and then...

```
python -m venv /scratch/gusmasdy/envs/ari
source /scratch/gusmasdy/envs/ari/bin/activate
# Just to be sure we are up-to-date
pip install --upgrade pip
pip install -r requirements.txt
```

If it is already installed we can just use:

```
source /scratch/gusmasdy/envs/ari/bin/activate
```

## References

[DFKI-MLST at DialAM-2024 Shared Task: System Description](https://aclanthology.org/2024.argmining-1.9/) (Binder et al., ArgMining 2024)

[QT30: A Corpus of Argument and Conflict in Broadcast Debate](https://aclanthology.org/2022.lrec-1.352/) (Hautli-Janisz et al., LREC 2022)
