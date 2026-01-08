FROM python:3.12-bookworm

ARG TZ
ENV TZ="$TZ"

# Install basic development tools and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  unzip \
  less \
  procps \
  vim \
  fzf \
  zsh \
  curl \
  ca-certificates \
  sudo \
  # Install networking tools for firewall management  
  # iptables \
  # iproute2 \
  # ipset \
  # dnsutils \
  # aggregate \
  jq \
  # OpenCV dependencies
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libgomp1 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create dev user with sudo access
RUN useradd -m -s /bin/zsh -G sudo dev \
  && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
  && chmod 0440 /etc/sudoers.d/dev

# Persist bash history for dev user
RUN mkdir -p /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R dev:dev /commandhistory

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  curl -fsSL -O "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Set the default shell to zsh rather than sh
ENV SHELL=/bin/zsh

# Set the default editor and visual
ENV EDITOR=vim
ENV VISUAL=vim

# Create workspace directory with proper ownership
RUN mkdir -p /workspace && chown -R dev:dev /workspace

# Switch to dev user for installations
USER dev
ENV HOME=/home/dev

# Install nvm and Node.js 24
ENV NVM_DIR="$HOME/.nvm"
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash \  
  && \. "$NVM_DIR/nvm.sh" \ 
  && nvm install 24

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash 

# Default powerline10k theme
ARG ZSH_IN_DOCKER_VERSION="$ZSH_IN_DOCKER_VERSION"
RUN sh -c "$(curl -fsSL https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Add poetry, nvm, npm, Claude Code to PATH, and load nvm
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> $HOME/.zshrc \
  && echo 'export NVM_DIR="$HOME/.nvm"' >> $HOME/.zshrc \
  && echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> $HOME/.zshrc \
  && echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> $HOME/.zshrc

# Set working directory
WORKDIR /workspace
