#!/bin/bash
#
# This script sets up the Lustre settings on mdx to mount `/data` and `/model` directories.
# This script is for testing purposes to verify if the third line of `ip -br addr` contains the STORAGE_NETWORK1_IPV4.
# No guarantees on its behavior.
#
# Usage: Run `sudo bash no_arg_setup.sh`
#
set -eux -o pipefail


USAGE_MSG="Usage: sudo bash no_arg_setup.sh"

# Check that no arguments are passed to the script
if [ $# -ne 0 ]; then
  echo "$USAGE_MSG"
  exit 1
fi

# Check if running on a login node
if [[ "$(uname -n)" == "login-node" ]]; then
  echo >&2 "This script cannot be run on a login node."
  exit 1
fi

# Ensure the script is being run as root (via sudo)
if [ "$(id -u)" -ne 0 ]; then
  echo >&2 "This script must be run as root. Use sudo."
  exit 1
fi

# Network configuration setup
cp /etc/lnet.conf.ddn.j2 /etc/lnet.conf.ddn

# Get the network name from the third line of `ip -br addr`
NETWORK_NAME=$(ip -br addr | sed -n 3p | awk '{print $1}')
STORAGE_NETWORK1_IPV4=$(ip -br addr | sed -n 3p | awk '{print $3}' | cut -d'/' -f1)

# Error checking for network name detection
if [ -z "$NETWORK_NAME" ] || [ -z "$STORAGE_NETWORK1_IPV4" ]; then
  echo >&2 "Error: Network interface or IP address could not be detected."
  exit 1
fi

# Update lnet configuration file with detected network interface and IP address
sed -i "s/{{ tcp_src_ipaddr }}/$STORAGE_NETWORK1_IPV4/g" /etc/lnet.conf.ddn
sed -i "s/{{ ib_src_ipaddr }}/$STORAGE_NETWORK1_IPV4/g" /etc/lnet.conf.ddn
sed -i "s/{{ ib_netif }}/$NETWORK_NAME/g" /etc/lnet.conf.ddn
sed -i "s/{{ tcp_netif }}/$NETWORK_NAME/g" /etc/lnet.conf.ddn

# Modify /etc/fstab to uncomment relevant Lustre entries
sed -i '/^#172\.17\.8\./s/^#//' /etc/fstab

# Enable Lustre client service
systemctl enable lustre_client

# Reboot the system to apply the changes
reboot