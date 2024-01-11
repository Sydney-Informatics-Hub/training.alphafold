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
# sudo docker build -t sydneyinformaticshub/alphafold:alphafold_v2.2.3
# sudo docker push sydneyinformaticshub/alphafold:alphafold_v2.2.3.
#  sudo docker run -it  sydneyinformaticshub/alphafold:alphafold_v2.2.3
# Run with:
# sudo docker run -it --gpus=all sydneyinformaticshub/alphafold /bin/bash -c "/app/run_alphafold.sh \
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


#Based on https://github.com/google-deepmind/alphafold/tree/86a0b8ec7a39698a7c2974420c4696ea4cb5743a/
# 25 August 2022 v2.2.3
ARG CUDA=10.2
FROM nbutter/pytorch:ubuntu1604
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

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
  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py37_4.10.3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py37_4.10.3-Linux-x86_64.sh

# There is another version of conda in the base image, remove it.
RUN rm -rf /build/miniconda3
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /app/alphafold

RUN wget https://github.com/deepmind/alphafold/archive/refs/tags/v2.2.3.zip && unzip v2.2.3.zip && mv alphafold-2.2.3/* . && rm -rf alphafold-2.2.3 && rm v2.2.3.zip

RUN wget -q -P /app/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN conda install pip -y
# RUN pip install -r /app/alphafold/requirements.txt
RUN conda install -c conda-forge cudatoolkit==10.2.89 -y
RUN conda install -c conda-forge openmm==7.5.0 -y
RUN conda install -c conda-forge pdbfixer==1.7 -y
RUN pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 ml-collections==0.1.0 numpy==1.19.5 pandas==1.3.4 protobuf==3.20.1 scipy==1.7.0 tensorflow-cpu==2.5.0
RUN pip install -U https://storage.googleapis.com/jax-releases/cuda102/jaxlib-0.1.71+cuda102-cp37-none-manylinux2010_x86_64.whl
RUN pip install jax==0.2.25

# # Apply OpenMM patch.
RUN cd /opt/conda/lib/python3.7/site-packages && patch -p0 < /app/alphafold/docker/openmm.patch
#
# # Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

RUN echo $'#!/bin/bash\n\
ldconfig\n\
python /app/alphafold/run_alphafold.py "$@"' > /app/run_alphafold.sh \
  && chmod +x /app/run_alphafold.sh

ENV PATH="/app/:$PATH"
# ENTRYPOINT ["/app/run_alphafold.sh"]
