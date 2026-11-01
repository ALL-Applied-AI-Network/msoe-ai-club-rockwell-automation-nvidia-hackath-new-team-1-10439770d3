#!/usr/bin/env bash
bash /tmp/setup.sh > /work/setup_2dgs.log 2>&1
printf '%s\n' "$?" > /work/setup_2dgs.exit

