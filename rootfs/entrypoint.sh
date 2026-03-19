#!/bin/sh
set -e

chown -R fox:fox /models

exec dropbear -F -E -s -p 2222 -R
