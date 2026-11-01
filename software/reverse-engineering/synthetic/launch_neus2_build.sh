#!/usr/bin/env bash
bash /tmp/build_neus2.sh > /work/neus2_build.log 2>&1
printf '%s\n' "$?" > /work/neus2_build.exit

