#!/bin/bash
#
# This script sets up the MDX environment by mounting `/data` and `/model` directories.
#
# Usage: Run `sudo bash setup_mdx.sh [-f] [-v] NETWORK_STORAGE1_IPV4`
#

USAGE_MSG="Usage: sudo bash setup_mdx.sh [-f] [-v] STORAGE_NETWORK1_IPV4"

# Default values
FORCE_REBOOT=false
ENABLE_DEBUG=false

# Parse options using getopts
while getopts "fv" opt; do
  case $opt in
    f)
      FORCE_REBOOT=true
      ;;
    v)
      ENABLE_DEBUG=true
      ;;
    \?)
      echo "$USAGE_MSG"
      exit 1
      ;;
  esac
done

# Shift arguments to get the remaining non-option arguments
shift $((OPTIND - 1))

# Check if the STORAGE_NETWORK1_IPV4 argument is provided
if [ $# -ne 1 ]; then
  echo "$USAGE_MSG"
  exit 1
fi

STORAGE_NETWORK1_IPV4=$1

# Enable debug mode if '-v' is specified
if [ "$ENABLE_DEBUG" == true ]; then
  set -eux -o pipefail
else
  set -e
fi

# Check if the script is being run as root (via sudo)
if [ "$(id -u)" -ne 0 ]; then
  set +x
  echo >&2 "This script must be run as root. Use sudo."
  echo >&2 "$USAGE_MSG"
  exit 1
fi

# Network configuration setup
cp /etc/lnet.conf.ddn.j2 /etc/lnet.conf.ddn
NETWORK_NAME=$(ip -br addr | grep "$STORAGE_NETWORK1_IPV4" | awk '{print $1}')

# Error checking for network name detection
if [ -z "$NETWORK_NAME" ]; then
  echo >&2 "Error: Network interface for IP $STORAGE_NETWORK1_IPV4 not found."
  exit 1
fi

sed -i "s/{{ tcp_src_ipaddr }}/$STORAGE_NETWORK1_IPV4/g" /etc/lnet.conf.ddn
sed -i "s/{{ ib_src_ipaddr }}/$STORAGE_NETWORK1_IPV4/g" /etc/lnet.conf.ddn
sed -i "s/{{ ib_netif }}/$NETWORK_NAME/g" /etc/lnet.conf.ddn
sed -i "s/{{ tcp_netif }}/$NETWORK_NAME/g" /etc/lnet.conf.ddn

# /etc/fstab modification
sed -i '/^#172\.17\.8\./s/^#//' /etc/fstab

# Enable Lustre client service
systemctl enable lustre_client

# Reboot confirmation
if [ "$FORCE_REBOOT" == true ]; then
  echo "Forcing system reboot..."
  reboot
else
  echo "Do you want to reboot the system now? (y/n)"
  read -r answer
  if [[ "$answer" == "y" || "$answer" == "yes" ]]; then
    echo "Rebooting system..."
    reboot
  else
    echo "Skipping reboot. You need to reboot the system manually to apply the changes."
  fi
fi