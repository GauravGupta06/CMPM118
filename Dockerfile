FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Only install the packages NOT already in the PyTorch image
RUN pip install tonic matplotlib snntorch Lempel-Ziv-Complexity rockpool samna scikit-learn

# Copy your code and data
COPY data/ ./data/
COPY results/ ./results/

COPY train_snn.py .
COPY LoadDataset.py .
COPY router.py .
COPY run_snn.py .
COPY SNN_model.py .
COPY requirements.txt .

CMD ["sleep", "infinity"]