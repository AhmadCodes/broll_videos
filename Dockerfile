# Base image
FROM runpod/base:0.4.2-cuda11.8.0

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Update and upgrade the system packages (Worker Template)
RUN apt-get update && \
    apt-get upgrade -y

# Install System Packages
RUN apt-get install ffmpeg -y


# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python3.11 /cache_models.py && \
    rm /cache_models.py

# Add src files (Worker Template)
# ADD src .
ADD app .

CMD python3.11 -u /rp_handler.py