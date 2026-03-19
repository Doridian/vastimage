#!/bin/sh
set -e

exec dropbear -F -E -s -p 2222 -R
