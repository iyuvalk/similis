#!/usr/bin/python3
import random
import sys


randomizer = random.Random()
domains_list = []
source_file = sys.argv[1]
malicious_domains_list = [
    "g00gle.com",
    "google.biz",
    "google.con",
    "facebook.net",
    "facebook.dssdfsfhgfh.co.cn",
    "funny-antonelli.172-245-8-169.plesk.page",
    "webmail.zimmail.repl.co",
    "nvidia-plus.com",
    "houseboatconcordiabandb.com",
    "paypayne.no-replyiv.com"
]
domains_to_use = 1000
result_set_size = 100000
with open(source_file) as raw_domains_list:
    file_contents = raw_domains_list.read()
    for line in file_contents.split('\n'):
        domain = line.split(',')[1]
        domains_list.append(domain)
        if len(domains_list) > domains_to_use:
            break

for i in range(0, result_set_size):
    print(domains_list[randomizer.randint(0, len(domains_list) - 1)])

for domain in malicious_domains_list:
    print(domain)
