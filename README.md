# EP15 — Weight Initialization Explained (Xavier, He & Why It Matters)

**Deep Learning Series · Module 4: Training Deep Networks · Episode 15 of 72**

## What this episode covers
- Why naive random initialization breaks deep networks
- The variance problem (signal explosion / collapse)
- Xavier / Glorot initialization (math + intuition)
- He / Kaiming initialization (the ReLU correction)
- PyTorch defaults and when to override them
- Side-by-side training curves
- Clean PyTorch code to apply both methods

## Challenge
Compare three initializations on a 10-layer MLP trained on MNIST:

1. Bad random (Normal std=1.0) + Tanh  
2. Xavier Uniform + Tanh  
3. He / Kaiming Uniform + ReLU  

Train each for 20 epochs and post the three final validation accuracies in the comments.

**File:** `ep15_init_comparison.py`

```bash
python ep15_init_comparison.py
