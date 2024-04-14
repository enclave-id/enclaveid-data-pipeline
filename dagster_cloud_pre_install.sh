# HDBScan needs gcc and g++ to build the wheel for installation
apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*
