#!/bin/bash

# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install Miniconda silently
bash miniconda.sh -b -p $HOME/miniconda

# Initialize conda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

# (Optional) Add conda to PATH for future use
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc

# Create and activate your environment
conda env create -f environment.yml
conda activate your_env_name  # Replace with the name inside environment.yml

echo "✅ Environment setup complete."

#!/bin/bash

# Make setup script executable and run it
chmod +x setup.sh
./setup.sh

# Activate conda environment
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate py312  # ← replace with your env name from environment.yml

# Launch Streamlit app
streamlit run app.py
