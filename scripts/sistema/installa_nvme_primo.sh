#!/bin/bash
set -e
[ "$(id -u)" = 0 ] || { echo "Lancialo con sudo"; exit 1; }
D=$(dirname "$(readlink -f "$0")")
install -m 755 "$D/nvme_primo.sh" /usr/local/sbin/nvme_primo.sh
install -m 644 "$D/nvme_primo.service" /etc/systemd/system/nvme_primo.service
systemctl daemon-reload
systemctl enable --now nvme_primo.service
efibootmgr | head -4
