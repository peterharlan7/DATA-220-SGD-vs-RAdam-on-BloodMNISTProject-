# DATA 220 – SGD vs RAdam on BloodMNIST (Group 7)

How do plain SGD and RAdam differ in accuracy, convergence speed, learning-rate
sensitivity and memory when training the same ResNet-8 on BloodMNIST?

Team: Ashritha, Jason, Peter, Sai

## Repo layout

```
src/          shared code, imported by every notebook
  utils.py    seeds, device, optimizer-state memory      
  data.py     BloodMNIST loading + DataLoaders           
  model.py    ResNet-8                                   
  train.py    training / evaluation loop + CSV logging   
notebooks/    one notebook per experiment or person
results/      CSV logs and PNG plots (committed)
init.pt       shared starting weights (committed, never regenerate)
```

## Run it in Colab

1. Open `notebooks/00_colab_template.ipynb` in Colab
   (File → Open notebook → GitHub tab → paste this repo's URL).
2. File → Save a copy in Drive, then rename it for your experiment.
3. Run all cells. The first cell clones this repo so `from src import ...` works.

## Team rules

1. Edit only the files your Trello card owns. Need a change in someone else's file? Ask them.
2. Pull (re-run the clone cell) before you start; commit when you finish.
3. Clear notebook outputs before committing (Edit → Clear all outputs).
4. Never change `init.pt`. Both optimizers start from these exact weights.
5. Results go in `results/` as CSV or PNG and do get committed.

## Fixed experiment settings

| Setting | Value |
|---|---|
| Dataset | BloodMNIST 28×28, official split 11,959 / 1,712 / 3,421 |
| Model | ResNet-8 (16 → 32 → 64 channels) |
| Starting weights | `init.pt` |
| Batch size | TBD |
| Seed(s) | TBD |
| LR screen | 3 rates per optimizer |
| Final training | 8–10 epochs at best validation LR |
| Hardware | Colab CPU |

## Dataset

BloodMNIST downloads automatically through the `medmnist` package.
Source: https://github.com/MedMNIST/MedMNIST
(Yang et al., *MedMNIST v2*, Scientific Data, 2023)
