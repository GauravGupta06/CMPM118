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
COPY train.py .
COPY evaluate.py .
COPY router.py .

# Copy data directory (for dataset caching)
COPY data/ ./data/

# Copy requirements for reference
COPY requirements.txt .

# Copy documentation
COPY ROUTER_USAGE_GUIDE.md .

CMD ["sleep", "infinity"]