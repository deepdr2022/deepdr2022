## Code repo for paper: `DeepDR: Deep Learning-based Data Recovery for Network Measurement`

### Get the data

> 1. Go to **Releases** section and download the two data files therein
> 2. Make a new directory called `data`
> 3. Put these two data files in this directory

### Train
```python
python train_deepdr.py --dataname ab  # train the DeepDR model for Abilene
python train_deepdr.py --dataname ge  # train the DeepDR model for GEANT
```

### Test

```python
python test.py --dataname ab --model DeepDR --modelpath ab_deepdr # get test results for Abilene
python test.py --dataname ge --model DeepDR --modelpath ge_deepdr # get test results for GEANT
```