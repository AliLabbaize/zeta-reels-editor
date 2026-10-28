#!/usr/bin/env bash
# Exit (so whoever launched it gets woken) when a long job looks stuck.
#
#   scripts/watchdog.sh '<pgrep -f pattern>' [check_s=120] [stall_checks=3]
#
# Stuck = the job's CPU time advances < 10 s over stall_checks checks (the
# 2026-10-03 hang sat at ~13% CPU in swap for 20 min), or swap use > 10 GB
# (8 GB Mac). Also exits when no matching process is left: the job ended.
pat=$1 every=${2:-120} limit=${3:-3}
cpu() { ps -o time= -p "$(pgrep -f "$pat" | tr '\n' ',' | sed 's/,$//')" 2>/dev/null |
        awk -F'[:.]' '{ s += (NF==4 ? $1*3600+$2*60+$3 : $1*60+$2) } END { print s+0 }'; }
last=$(cpu) slow=0
while pgrep -f "$pat" >/dev/null; do
  sleep "$every"
  pgrep -f "$pat" >/dev/null || break
  now=$(cpu)
  swap=$(sysctl -n vm.swapusage | awk '{ sub("M","",$6); print int($6) }')
  d=$((now - last))
  # A worker restarting (next render piece) resets the counter: that is progress.
  [ "$d" -lt 0 ] && d=999
  if [ "$d" -lt 10 ]; then slow=$((slow + 1)); else slow=0; fi
  echo "$(date +%T) cpu+$((now - last))s swap=${swap}M slow=$slow"
  [ "$slow" -ge "$limit" ] && { echo "STALLED: '$pat' barely used CPU for $((every * limit))s"; exit 2; }
  [ "$swap" -gt 10240 ] && { echo "SWAP: ${swap}M used, memory pressure"; exit 3; }
  last=$now
done
echo "ENDED: no '$pat' process left"
