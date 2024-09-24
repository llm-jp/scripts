#!/bin/bash
#
# This script sets up the Lustre settings on mdx to mount `/data` and `/model` directories.
# This script is for testing purposes to verify if the third line of `ip -br addr` contains the STORAGE_NETWORK1_IPV4.
# No guarantees on its behavior.
#
# Usage: Run `sudo bash no_arg_setup.sh`
#
set -eux -o pipefail

# Get the STORAGE_NETWORK1_IPV4 from the third line of `ip -br addr`
STORAGE_NETWORK1_IPV4=$(ip -br addr | sed -n 3p | awk '{print $3}' | cut -d'/' -f1)
bash setup.sh -f $STORAGE_NETWORK1_IPV4
