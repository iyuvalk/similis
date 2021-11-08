#!/usr/bin/env bash

domain=${1}
res=$(echo '{"text":"'"${domain}"'"}' | nc -U /tmp/similis.socket)
#echo "${domain}: ${res}" > "/tmp/similis/similis_res_`echo "${domain}" | md5sum | awk '{print $1}'`.log"
#echo "--> ${domain}"
echo "{\"${domain}\": ${res}}"
