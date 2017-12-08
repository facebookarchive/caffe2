#!/bin/bash

set -ex

# Use AWS mirror if running in EC2
if [ -n "${EC2:-}" ]; then
  A="archive.ubuntu.com"
  B="us-east-1.ec2.archive.ubuntu.com"
  perl -pi -e "s/${A}/${B}/g" /etc/apt/sources.list
fi

apt-get update
apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        sudo

apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
