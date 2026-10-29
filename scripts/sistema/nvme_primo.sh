#!/bin/bash
N=$(efibootmgr | awk '/^Boot[0-9A-Fa-f]+\*/ && /NVMe/ {print substr($1,5,4); exit}')
if [ -z "$N" ]; then
  echo "nvme_primo: voce NVMe non trovata, non cambio niente"
  exit 0
fi
ORD=$(efibootmgr | awk -F': ' '/^BootOrder/{print $2}')
NEW="$N,$(echo "$ORD" | tr ',' '\n' | grep -vix "$N" | paste -sd,)"
NEW=${NEW%,}
if [ "${1:-}" = prova ]; then
  echo "NVMe = Boot$N"
  echo "ordine attuale: $ORD"
  echo "ordine nuovo:   $NEW"
  echo "BootNext:       $N"
  exit 0
fi
[ "$ORD" = "$NEW" ] || efibootmgr -q -o "$NEW"
efibootmgr -q -n "$N"
echo "nvme_primo: BootOrder $NEW, BootNext $N"
