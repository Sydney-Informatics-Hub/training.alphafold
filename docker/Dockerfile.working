# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications for Sydney Uni:
# Set base image to be ubuntu 16.04 and cuda10.2 to run on our HPC
# contact sih.info@sydney.edu.au
# Conda version now hardcoded.
#
# Build with:
# sudo docker build -t sydneyinformaticshub/alphafold .
#
# Run with:
# sudo docker run -it --gpus=all sydneyinformaticshub/alphafold /bin/bash -c "python /app/alphafold/alphafold-2.3.2/run_alphafold.py \
#   --data_dir=${ALPHADB} \
#   --uniref90_database_path=${ALPHADB}/uniref90/uniref90.fasta \
#   --mgnify_database_path=${ALPHADB}/mgnify/mgy_clusters_2018_12.fa \
#   --template_mmcif_dir=${ALPHADB}/pdb_mmcif/mmcif_files/ \
#   --obsolete_pdbs_path=${ALPHADB}/pdb_mmcif/obsolete.dat \
#   --fasta_paths=/project/Training/DATA/input.fasta \
#   --output_dir=${WORKDIR}/output_directory_gpu \
#   --db_preset=full_dbs \
#   --max_template_date=2022-03-23 \
#   --use_gpu_relax=True \
#   --model_preset=monomer \
#   --pdb70_database_path=${ALPHADB}/pdb70/pdb70 \
#   --bfd_database_path=${ALPHADB}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
#   --uniref30_database_path=${ALPHADB}/uniclust30/uniclust30_2018_08/uniclust30_2018_0 \
#  "

ARG CUDA=10.2
FROM nbutter/pytorch:ubuntu1604
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        git \
        hmmer \
        kalign \
        tzdata \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh

# Install conda/pip packages.
ENV PATH="/opt/conda/bin:$PATH"

RUN conda install -y -c conda-forge libgcc
RUN conda install -y -c anaconda libstdcxx-ng
RUN conda install -y -c conda-forge pdbfixer pip
RUN conda install -y -c conda-forge openmm==7.5.0 cudatoolkit==10.2.89
RUN pip install absl-py==1.0.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.9 dm-tree==0.1.6 immutabledict jax==0.1.71 ml-collections==0.1.0 numpy==1.21.6 pandas==1.3.4 protobuf==3.20.1 scipy==1.7.0 tensorflow-cpu==2.9.0
RUN pip install --upgrade --no-cache-dir "jax[cuda102]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install mock

RUN mkdir -p /app && cd /app && wget https://github.com/deepmind/alphafold/archive/refs/tags/v2.2.0.zip && unzip v2.2.0.zip -d /app/alphafold
RUN mkdir -p /app && cd /app && wget https://github.com/deepmind/alphafold/archive/refs/tags/v2.3.2.zip && unzip v2.3.2.zip -d /app/alphafold

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

WORKDIR /app/alphafold

# Add Artemis HPC folders. !! Already inherited from nbutter/pytorch !!
# RUN mkdir /project /scratch && touch /usr/bin/nvidia-smi
