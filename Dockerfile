FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Only install the packages NOT already in the PyTorch image
RUN pip install tonic matplotlib snntorch Lempel-Ziv-Complexity rockpool samna scikit-learn

# Copy core infrastructure
COPY core/ ./core/
COPY datasets/ ./datasets/
COPY models/ ./models/
COPY results/ ./results/

# Copy training and evaluation scripts
COPY train_shd.py .
COPY train_UCI_HAR.py .
COPY train_dvsgesture.py .

COPY evaluate.py .
COPY router.py .


# Copy data directory (for dataset caching)
COPY data/ ./data/

# Copy requirements for reference
COPY requirements.txt .


CMD ["sleep", "infinity"]