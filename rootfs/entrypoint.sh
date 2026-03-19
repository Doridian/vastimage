#!/bin/sh
set -e

chown -R fox:fox /models

# Generate (or regenerate) dropbear host keys
rm -frv /etc/dropbear/*
echo '===BEGIN HOST PUBLIC KEYS==='
makekey() {
    local type="$1"
    dropbearkey -t "$type" -f "/etc/dropbear/dropbear_${type}_host_key" 2>/dev/null | grep -F 'Public key portion is:' -A1 | tail -1
}
makekey rsa
makekey ecdsa
makekey ed25519
echo '===END HOST PUBLIC KEYS==='

export -p > /run/container-env

exec dropbear -E -F -s -p 2222
