FROM nvcr.io/nvidia/rapidsai/rapidsai-core:22.08-cuda11.0-runtime-ubuntu20.04-py3.9

RUN apt update

RUN apt install -y wget git python3 vim gcc g++ make nvidia-cuda-toolkit

RUN pip uninstall -y cupy && pip install cupy-cuda110

#Build and install CUDA-aware OpenMPI
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz && \
    tar -xzf openmpi-4.1.4.tar.gz && \
    cd openmpi-4.1.4 && \
    ./configure --with-cuda=/usr/ --prefix=/opt/openmpi-4.1.4 && \
    make all install
ENV PATH=/opt/openmpi-4.1.4/bin/:$PATH
ENV LD_LIBRARY_PATH=/opt/openmpi-4.1.4/lib:$LD_LIBRARY_PATH

RUN pip install genepattern-python mpi4py h5py fastcluster

    
