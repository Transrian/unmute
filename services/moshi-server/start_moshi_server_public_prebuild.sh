#!/bin/bash
# Binary is pre-compiled at Docker build time — just authenticate and run.
set -ex

export LD_LIBRARY_PATH=$(python3.12 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

uvx hf auth login --token $HUGGING_FACE_HUB_TOKEN

# Subtle detail here: We use the full path to `moshi-server` because there is a `moshi-server` binary
# from the `moshi` Python package. We'll fix this conflict soon.
/root/.cargo/bin/moshi-server $@
