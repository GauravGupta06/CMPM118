# Router.py Usage Guide

## Table of Contents
1. [Quick Start Examples](#quick-start-examples)
2. [How Command-Line Arguments Work](#how-command-line-arguments-work)
3. [All Available Options](#all-available-options)
4. [How Argparse Works (Detailed Explanation)](#how-argparse-works-detailed-explanation)

---

## Quick Start Examples

### Example 1: Basic Usage (Minimum Required Arguments)
```bash
cd rockpoolDevelopment

python router.py \
  --sparse_model_path ./results/small/models/Rockpool_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth \
  --dense_model_path ./results/large/models/Rockpool_Non_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth
```
**What this does:** Runs the router with default settings (700 frequency bins, 100 frames, default hyperparameters)

---

### Example 2: Custom Hyperparameters
```bash
python router.py \
  --sparse_model_path ./results/small/models/Sparse_Custom.pth \
  --dense_model_path ./results/large/models/Dense_Custom.pth \
  --tau_mem_sparse 0.015 \
  --tau_mem_dense 0.025 \
  --spike_lam_sparse 1e-5 \
  --spike_lam_dense 1e-7
```
**What this does:** Uses custom membrane time constants and spike regularization values

---

### Example 3: Different Dataset Path
```bash
python router.py \
  --sparse_model_path ./results/small/models/Sparse.pth \
  --dense_model_path ./results/large/models/Dense.pth \
  --dataset_path /path/to/my/data
```
**What this does:** Uses a different location for the cached dataset

---

### Example 4: See All Options and Help
```bash
python router.py --help
```
**What this does:** Shows all available command-line options with descriptions

---

## How Command-Line Arguments Work

### Basic Anatomy of a Command
```bash
python router.py --argument_name value --another_argument value
```

**Breaking it down:**
- `python` = Run Python interpreter
- `router.py` = The script to run
- `--argument_name value` = An argument with its value
- You can chain multiple arguments together

### Required vs Optional Arguments

**Required Arguments** (you MUST provide these):
- `--sparse_model_path` - Path to your sparse model file
- `--dense_model_path` - Path to your dense model file

**Optional Arguments** (have default values):
- `--input_size` (default: 700)
- `--n_frames` (default: 100)
- `--tau_mem_sparse` (default: 0.01)
- etc.

### Multi-Line Commands (The `\` Character)

When commands get long, you can split them across multiple lines using `\`:

```bash
python router.py \
  --sparse_model_path ./results/small/models/Sparse.pth \
  --dense_model_path ./results/large/models/Dense.pth \
  --input_size 700
```

**Important:** The `\` must be the LAST character on the line (no spaces after it)

Without `\`, you'd have to write it all on one line:
```bash
python router.py --sparse_model_path ./results/small/models/Sparse.pth --dense_model_path ./results/large/models/Dense.pth --input_size 700
```

---

## All Available Options

### Required Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--sparse_model_path` | string | Path to sparse model .pth file | `./results/small/models/Sparse.pth` |
| `--dense_model_path` | string | Path to dense model .pth file | `./results/large/models/Dense.pth` |

### Dataset Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_path` | string | `./data` | Directory where SHD dataset is cached |
| `--input_size` | int | 700 | Number of frequency bins |
| `--n_frames` | int | 100 | Number of temporal bins |

### Model Hyperparameter Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tau_mem_sparse` | float | 0.01 | Membrane time constant for sparse model |
| `--tau_mem_dense` | float | 0.02 | Membrane time constant for dense model |
| `--spike_lam_sparse` | float | 1e-6 | Spike regularization for sparse model |
| `--spike_lam_dense` | float | 1e-8 | Spike regularization for dense model |

---

## How Argparse Works (Detailed Explanation)

### What is Argparse?

`argparse` is a Python library that makes it easy to write command-line tools. Instead of hardcoding values in your script, you can pass them as arguments when you run it.

### Step-by-Step: How It Works in router.py

#### Step 1: Create the Parser
```python
parser = argparse.ArgumentParser(
    description="Router for Rockpool SHD models"
)
```
This creates an object that will handle parsing command-line arguments.

#### Step 2: Add Arguments
```python
parser.add_argument('--sparse_model_path', type=str, required=True,
                   help='Path to pre-trained sparse model (.pth file)')
```

**Breaking this down:**
- `'--sparse_model_path'` = The name you'll use on command line
- `type=str` = This argument expects a string value
- `required=True` = User MUST provide this argument
- `help='...'` = Description shown when user runs `--help`

**Different argument types:**

**String argument:**
```python
parser.add_argument('--dataset_path', type=str, default='./data')
```
Usage: `--dataset_path /my/path`

**Integer argument:**
```python
parser.add_argument('--input_size', type=int, default=700)
```
Usage: `--input_size 700`

**Float argument:**
```python
parser.add_argument('--tau_mem_sparse', type=float, default=0.01)
```
Usage: `--tau_mem_sparse 0.015`

#### Step 3: Parse the Arguments
```python
args = parser.parse_args()
```

This line does the magic! It:
1. Looks at what the user typed on command line
2. Matches it to the arguments you defined
3. Converts values to correct types (str, int, float)
4. Creates an `args` object with all the values

#### Step 4: Use the Values
```python
sparse_model = SHDSNN_FC(
    input_size=args.input_size,      # Gets value from --input_size
    n_frames=args.n_frames,          # Gets value from --n_frames
    tau_mem=args.tau_mem_sparse,     # Gets value from --tau_mem_sparse
    ...
)
```

**Key Point:** `args.argument_name` gives you the value the user provided (or the default if they didn't provide it)

---

### Example Flow

Let's trace what happens when you run:
```bash
python router.py --sparse_model_path models/sparse.pth --dense_model_path models/dense.pth --n_frames 100
```

1. **Python starts router.py**
2. **Creates parser** with all the argument definitions
3. **Calls `args = parser.parse_args()`**
   - Sees `--sparse_model_path models/sparse.pth` → `args.sparse_model_path = "models/sparse.pth"`
   - Sees `--dense_model_path models/dense.pth` → `args.dense_model_path = "models/dense.pth"`
   - Sees `--n_frames 100` → `args.n_frames = 100`
   - Doesn't see `--input_size`, so uses default → `args.input_size = 700`
4. **Your code uses these values:**
   ```python
   sparse_model.load_model(args.sparse_model_path)  # Uses "models/sparse.pth"
   ```

---

### Why Use Argparse?

**Without argparse (hardcoded values):**
```python
SPARSE_MODEL_PATH = "./results/small/models/Take1.pth"
# To change it, you have to EDIT the code
```

**With argparse:**
```bash
python router.py --sparse_model_path ./results/small/models/Take2.pth
# Just change the command - no code editing!
```

**Benefits:**
1. ✅ No need to edit code for different experiments
2. ✅ Can run multiple experiments in parallel
3. ✅ Easy to script and automate
4. ✅ Built-in help system (`--help`)
5. ✅ Type checking and validation
6. ✅ Professional and standard practice

---

## Common Mistakes & Solutions

### Mistake 1: Forgetting Required Arguments
```bash
python router.py --input_size 700
```
**Error:** `error: the following arguments are required: --sparse_model_path, --dense_model_path`

**Solution:** Provide both model paths

---

### Mistake 2: Wrong Path Format
```bash
python router.py --sparse_model_path results\small\models\Sparse.pth  # Windows-style backslashes
```
**Solution:** Use forward slashes in paths: `results/small/models/Sparse.pth`

---

### Mistake 3: Spaces After `\` in Multi-Line Commands
```bash
python router.py \
  --sparse_model_path models/sparse.pth  # Space after \ will cause error
```
**Solution:** Make sure `\` is the LAST character (no trailing spaces)

---

### Mistake 4: Model Paths Don't Exist
```bash
python router.py --sparse_model_path models/nonexistent.pth --dense_model_path models/dense.pth
```
**Error:** File not found error when trying to load

**Solution:** Double-check your paths are correct and files exist

---

## Tips & Tricks

### Tip 1: Use Tab Completion
In most terminals, you can press TAB to auto-complete file paths:
```bash
python router.py --sparse_model_path ./results/sm[TAB]
# Auto-completes to: ./results/small/
```

### Tip 2: Create Shell Scripts for Repeated Runs
Instead of typing long commands, create a file `run_router.sh`:
```bash
#!/bin/bash
python router.py \
  --sparse_model_path ./results/small/models/Sparse_Take1.pth \
  --dense_model_path ./results/large/models/Dense_Take1.pth \
  --input_size 700 \
  --n_frames 100
```

Then run it with:
```bash
bash run_router.sh
```

### Tip 3: Check Argument Values
The script prints all configuration values before running:
```
================================================================================
CONFIGURATION
================================================================================
Sparse model path: ./results/small/models/Sparse.pth
Dense model path:  ./results/large/models/Dense.pth
Input size:        700 frequency bins
...
```
Always check this to make sure arguments are correct!

---

## What the Router Does (Quick Overview)

1. **Loads both models** (sparse and dense)
2. **Loads test dataset**
3. **Evaluates every test sample** on both models
4. **Computes complexity** for each sample (LZC score)
5. **Finds optimal threshold** using ROC analysis
6. **Routes samples** based on complexity:
   - Simple samples (low LZC) → Sparse model
   - Complex samples (high LZC) → Dense model
7. **Generates results:**
   - ROC curve graph
   - LZC vs Accuracy graph
   - JSON file with all metrics
   - LaTeX table for papers
8. **Shows energy savings:** How many spikes saved by routing

The goal: **Maintain accuracy while reducing energy consumption!**
