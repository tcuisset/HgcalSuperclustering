FROM registry.hub.docker.com/cmssw/el8:x86_64

RUN dnf install -y glx-utils mesa-dri-drivers && dnf clean all

RUN echo 'source /cvmfs/cms.cern.ch/cmsset_default.sh' >> ~/.bashrc && \
    echo 'export EDITOR=nano' >> ~/.bashrc && \
    git config --global --add safe.directory '*'

RUN wget -nv https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh &&\
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3 && \
    /opt/miniforge3/bin/mamba init && \
    rm -f Miniconda3-latest-Linux-x86_64.sh


COPY env.yml env.yml
RUN --mount=type=cache,target=/opt/miniforge3/pkgs /opt/miniforge3/bin/mamba env create -f env.yml



#COPY docker-cmsenv-entrypoint.sh /
#RUN chmod +x /docker-cmsenv-entrypoint.sh

#ENTRYPOINT [ "/docker-cmsenv-entrypoint.sh" ]
#CMD [ "sleep", "infinity" ]
