# Kubernetes Deployment Guide for Rockpool SNN Training

## Overview
This guide explains how to deploy and run the Rockpool SNN training on the Nautilus Kubernetes cluster with GPU support.

## Directory Structure
```
/home/gaurav/CMPM118/
├── nautilus/              # Kubernetes configuration files
│   ├── Dockerfile         # Docker image configuration
│   ├── pvc.yaml          # Persistent Volume Claim
│   └── deployment.yaml   # Kubernetes deployment config
├── core/                 # Core SNN infrastructure
├── datasets/             # Dataset loaders
├── models/               # Model architectures
├── train.py             # Training script
├── router.py            # Router for sparse/dense models
└── evaluate.py          # Evaluation script
```

## Step 1: Build and Push Docker Image

### Build the Docker image
```bash
cd /home/gaurav/CMPM118
docker build -t registry.gitlab.com/gauravgupta07/gauravgupta07-snn/train-snn:latest -f nautilus/Dockerfile .
```

### Push to GitLab registry
```bash
docker push registry.gitlab.com/gauravgupta07/gauravgupta07-snn/train-snn:latest
```

## Step 2: Deploy Persistent Volume Claim (PVC)

The PVC provides persistent storage for your models and results.

```bash
kubectl apply -f nautilus/pvc.yaml
```

**Verify PVC is created:**
```bash
kubectl get pvc gaurav-storage
```

You should see:
```
NAME              STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
gaurav-storage    Bound    pvc-xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx     100Gi      RWX            rook-cephfs    Xs
```

## Step 3: Deploy the Training Pod

```bash
kubectl apply -f nautilus/deployment.yaml
```

**Verify deployment:**
```bash
kubectl get pods
```

You should see:
```
NAME                                READY   STATUS    RESTARTS   AGE
gaurav-training-xxxxxxxxxx-xxxxx    1/1     Running   0          Xs
```

## Step 4: Access the Pod

### Exec into the pod
```bash
kubectl exec -it $(kubectl get pod -l app=gpu-app -o jsonpath='{.items[0].metadata.name}') -- /bin/bash
```

### **IMPORTANT: Understanding the PVC Mount**
When you exec into the pod, the PVC is mounted at `/workspace`, **NOT** at a folder called "pvc".

To see your persistent storage:
```bash
ls /workspace
```

If this is your first time, `/workspace` will be empty. The training script will create the necessary directory structure.

## Step 5: Run Training Inside the Pod

### Training a Sparse Model
```bash
python train.py \
  --model_type sparse \
  --epochs 200 \
  --batch_size 32 \
  --output_path /workspace \
  --dataset_path ./data
```

### Training a Dense Model
```bash
python train.py \
  --model_type dense \
  --epochs 200 \
  --batch_size 32 \
  --output_path /workspace \
  --dataset_path ./data
```

## Step 6: Verify Models are Saved to PVC

After training completes, check that models were saved:

```bash
ls -la /workspace/small/models/   # For sparse models
ls -la /workspace/large/models/   # For dense models
```

You should see files like:
```
Rockpool_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth
Rockpool_Non_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth
```

## Step 7: Running the Router

After training both sparse and dense models, you can run the router to analyze routing efficiency:

```bash
python router.py \
  --sparse_model_path /workspace/small/models/Rockpool_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth \
  --dense_model_path /workspace/large/models/Rockpool_Non_Sparse_Take1_Input700_T100_FC_Rockpool_Epochs200.pth \
  --input_size 700 \
  --n_frames 100
```

Router outputs will be saved to:
- `/workspace/results/ROC_curves/` - ROC curve graphs
- `/workspace/results/LZC_vs_Accuracy/` - LZC vs Accuracy graphs
- `/workspace/results/run_logs/` - JSON logs with detailed metrics

## Directory Structure in PVC

After training, your `/workspace` directory will look like:
```
/workspace/
├── small/                      # Sparse model outputs
│   ├── models/                # .pth model files
│   ├── graphs/                # Training curves
│   └── experiment_counter.txt # Experiment counter
├── large/                     # Dense model outputs
│   ├── models/
│   ├── graphs/
│   └── experiment_counter.txt
└── results/                   # Router outputs (if you run router)
    ├── ROC_curves/
    ├── LZC_vs_Accuracy/
    └── run_logs/
```

## Common Issues and Solutions

### Issue 1: "Can't see PVC folder"
**Problem:** User execs into pod and doesn't see a "pvc" folder

**Solution:** The PVC is mounted at `/workspace`, not "pvc". Use:
```bash
cd /workspace
ls
```

### Issue 2: "No such file or directory" when saving models
**Problem:** Directory structure doesn't exist in `/workspace`

**Solution:** The training script now automatically creates the directory structure. Just run train.py with `--output_path /workspace` and it will create:
- `/workspace/small/models/`
- `/workspace/small/graphs/`
- `/workspace/large/models/`
- `/workspace/large/graphs/`

### Issue 3: "Counter file not found"
**Problem:** `experiment_counter.txt` doesn't exist

**Solution:** The save_model method now automatically initializes the counter file if it doesn't exist.

### Issue 4: Pod keeps restarting
**Problem:** GPU resources not available

**Solution:** Check GPU availability:
```bash
kubectl describe pod $(kubectl get pod -l app=gpu-app -o jsonpath='{.items[0].metadata.name}')
```

The deployment requests:
- 1x NVIDIA GPU (L4, A10, or RTX 4090)
- 64Gi memory
- 18 CPUs

### Issue 5: Can't pull Docker image
**Problem:** `ImagePullBackOff` error

**Solution:** Verify GitLab registry secret:
```bash
kubectl get secret gitlab-secret
```

If missing, create it:
```bash
kubectl create secret docker-registry gitlab-secret \
  --docker-server=registry.gitlab.com \
  --docker-username=<your-username> \
  --docker-password=<your-token>
```

## Monitoring Training Progress

### View logs in real-time
```bash
kubectl logs -f $(kubectl get pod -l app=gpu-app -o jsonpath='{.items[0].metadata.name}')
```

### Check GPU usage
Inside the pod:
```bash
nvidia-smi
```

### Check dataset download progress
```bash
ls -lh /app/data/SHD/
```

## Copying Files from PVC to Local Machine

To retrieve trained models from the cluster:

```bash
kubectl cp $(kubectl get pod -l app=gpu-app -o jsonpath='{.items[0].metadata.name}'):/workspace/small/models/ ./local_models/
```

## Cleanup

### Delete the deployment (keeps PVC and data)
```bash
kubectl delete deployment gaurav-training
```

### Delete everything including PVC (DELETES ALL DATA!)
```bash
kubectl delete deployment gaurav-training
kubectl delete pvc gaurav-storage
```

## Resource Specifications

### GPU Node Affinity
The deployment requests nodes with:
- NVIDIA L4
- NVIDIA A10
- NVIDIA GeForce RTX 4090

### Resource Limits
- **GPU:** 1x NVIDIA GPU
- **Memory:** 64Gi
- **CPU:** 18 cores
- **Shared Memory:** 48Gi (for data loading)
- **PVC Storage:** 100Gi

## Tips for Efficient Training

1. **Use the PVC for all outputs**: Always use `--output_path /workspace` to persist results
2. **Monitor GPU usage**: Run `nvidia-smi` to ensure GPU is utilized
3. **Adjust batch size**: If you run out of memory, reduce `--batch_size`
4. **Use screen/tmux**: For long training runs, use screen or tmux to avoid disconnection
5. **Copy data periodically**: Backup important models to local machine using `kubectl cp`

## Next Steps

1. Build and push your Docker image
2. Deploy PVC and deployment
3. Exec into pod and verify `/workspace` is accessible
4. Run training for both sparse and dense models
5. Run router to analyze routing efficiency
6. Copy results back to local machine for analysis

For more details on router usage, see `ROUTER_USAGE_GUIDE.md`.
