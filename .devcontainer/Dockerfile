FROM registry.hub.docker.com/cmssw/el8:x86_64

RUN dnf install -y glx-utils mesa-dri-drivers && dnf clean all

# certificates for voms-proxy-init (adapted from cmssw-el9 apptainer script)
RUN rm -r /etc/vomses && ln -s /cvmfs/grid.cern.ch/etc/grid-security/vomses /etc/ &&  rm -r /etc/grid-security  && ln -s /cvmfs/grid.cern.ch/etc/grid-security /etc/


ARG USERNAME=cuisset
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /home/cuisset

USER $USERNAME

RUN echo 'source /cvmfs/cms.cern.ch/cmsset_default.sh' >> ~/.bashrc && \
    echo 'export EDITOR=nano' >> ~/.bashrc && \
    git config --global --add safe.directory '*'

# root mode
# RUN wget -nv https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh &&\
#     bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3 && \
#     /opt/miniforge3/bin/mamba init && \
#     /opt/miniforge3/bin/conda config --set auto_activate_base false && \
#     rm -f Miniconda3-latest-Linux-x86_64.sh

# non-root mode
RUN wget -nv https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh &&\
     bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/conda" && \
     "$HOME/conda/bin/mamba" init && \
     "$HOME/conda/bin/conda" config --set auto_activate_base false && \
     rm -f Miniconda3-latest-Linux-x86_64.sh


COPY env.yml env.yml
RUN --mount=type=cache,target="$HOME/conda/pkgs" "$HOME/conda/bin/mamba" env create -f env.yml


RUN git config --global user.email theo.cuisset@polytechnique.edu && \
git config --global user.name Théo Cuisset && \
git config --global user.github tcuisset


#COPY docker-cmsenv-entrypoint.sh /
#RUN chmod +x /docker-cmsenv-entrypoint.sh

#ENTRYPOINT [ "/docker-cmsenv-entrypoint.sh" ]
#CMD [ "sleep", "infinity" ]
