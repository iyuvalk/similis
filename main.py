#!/usr/bin/python3
import base64
import os
import sys
import socket
import json
import threading
import time
import uuid
from threading import Thread
import textdistance
import imagehash
from PIL import Image, ImageDraw, ImageFont
import collections
import argparse
import hashlib
import tldextract


class SimpleLRUCache:
    def __init__(self, size, max_value_frequency_to_keep=-1):
        self.size = size
        self.id = uuid.uuid4()
        self.cache_lock = threading.Lock()
        self._lru_cache = collections.OrderedDict()
        self.value_frequency_dict = {}
        self.max_value_frequency_to_keep = max_value_frequency_to_keep

    def get_value_frequency(self, value, total_tasks_count):
        value_hash = self.hash_value(value)
        with self.cache_lock:
            value_frequency = -1
            if value_hash in self.value_frequency_dict:
                value_frequency = self.value_frequency_dict[value_hash] / total_tasks_count * 100
            return value_frequency

    def hash_value(self, value):
        return hashlib.md5(value.encode()).hexdigest()

    def get(self, key):
        # print("DEBUG: Retrieving item (" + key + ") from cache (" + str(self.id) + ")...")
        value_hash = hashlib.md5(key)
        with self.cache_lock:
            try:
                value = self._lru_cache.pop(key)
                self._lru_cache[key] = value
                value_frequency = -1
                if value_hash in self.value_frequency_dict:
                    value_frequency = self.value_frequency_dict[value_hash]
                return value.copy(), value_frequency
            except KeyError:
                return -1

    def __put(self, value):
        try:
            self._lru_cache.pop(value)
        except KeyError:
            if len(self._lru_cache) >= self.size:
                self._lru_cache.popitem(last=False)
        self._lru_cache[value] = value

    def put(self, value):
        # print("DEBUG: Appending item (" + value + ") to cache (" + str(self.id) + ")...")
        with self.cache_lock:
            self.__put(value)

    def put_if_not_exist(self, value):
        # print("DEBUG: Appending item (" + value + ") to cache (" + str(self.id) + ") if it does not exist...")
        value_hash = self.hash_value(value)
        with self.cache_lock:
            if value not in self._lru_cache:
                self.__put(value)
            if value_hash not in self.value_frequency_dict:
                self.value_frequency_dict[value_hash] = 1
            else:
                self.value_frequency_dict[value_hash] = self.value_frequency_dict[value_hash] + 1
            if self.max_value_frequency_to_keep > 0 and self.value_frequency_dict[value_hash] > self.max_value_frequency_to_keep:
                del self.value_frequency_dict[value_hash]

    def exists(self, key):
        print("DEBUG: Checking if item (" + key + ") exists in cache (" + str(self.id) + ")...")
        with self.cache_lock:
            return key in self._lru_cache

    def dump(self):
        # print("DEBUG: Dumping items from cache (" + str(self.id) + ")...")
        with self.cache_lock:
            return self._lru_cache.copy().items()

    def len(self):
        # print("DEBUG: Getting length of cache (" + str(self.id) + ")...")
        with self.cache_lock:
            return len(self._lru_cache)


class SimilisServer:
    def __init__(self, socket_path, min_cache_size, max_cache_size, sufficient_similarity_rate=90, min_similarity_to_report=95, max_frequency_to_report=2, report_members_of_same_domain=False):
        self.socket_path = socket_path
        self.cache = SimpleLRUCache(max_cache_size)
        self.min_cache_size = min_cache_size
        self.sufficient_similarity_rate = sufficient_similarity_rate
        self.min_similarity_to_report = min_similarity_to_report
        self.max_frequency_to_report = max_frequency_to_report
        self.report_members_of_same_domain = report_members_of_same_domain
        self.max_time_taken_ms = 0
        self.tasks_counter = 0
        self.time_taken_recent5 = []
        self.time_taken_recent15 = []
        self.time_taken_recent45 = []
        self.time_taken_mgmt_lock = threading.Lock()
        self.algorithms = {
            "textual_starts_with": self.get_similarity_startswith,
            "textual_prefix": self.get_similarity_text_distance_prefix,
            "textual_contains": self.get_similarity_contains,
            "textual_strcmp95": self.get_similarity_strcmp95,
            # "textual_hamming": self.get_similarity_text_distance_hamming,
            # "textual_damerau_levenshtein": self.get_similarity_text_distance_damerau_levenshtein,
            # "textual_levenshtein": self.get_similarity_text_distance_levenshtein,
            # "textual_jaccard": self.get_similarity_jaccard,
            # "textual_smith_waterman": self.get_similarity_smith_waterman
            # "visual_phash_free_mono": self.get_similarity_visual_free_mono,
            # "visual_phash_free_sans": self.get_similarity_visual_free_sans
        }
        self.threads_count = 0
        self.threads_lock = threading.Lock()

    def start(self):
        # Make sure the socket does not already exist
        try:
            os.unlink(self.socket_path)
        except OSError:
            if os.path.exists(self.socket_path):
                raise

        # Create a UDS socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Bind the socket to the address
        print('starting up on {}'.format(self.socket_path))
        sock.bind(self.socket_path)

        # Listen for incoming connections
        sock.listen(1)

        while True:
            # Wait for a connection
            connection, client_address = sock.accept()
            Thread(target=self.connection_handler, args=[connection]).start()

    def get_similarity_contains(self, text1, text2):
        if text2 in text1:
            return len(text2) / len(text1) * 100
        else:
            return 0

    def get_similarity_startswith(self, text1, text2):
        if str(text1).startswith(text2):
            return len(text2) / len(text1) * 100
        else:
            return 0

    def get_similarity_text_distance_hamming(self, text1, text2):
        if len(text1) == len(text2):
            return textdistance.hamming.similarity(text1, text2) / len(text1) * 100
        else:
            return 0

    def get_similarity_text_distance_levenshtein(self, text1, text2):
        if len(text1) > len(text2):
            return textdistance.levenshtein.similarity(text1, text2) / len(text1) * 100
        else:
            return textdistance.levenshtein.similarity(text1, text2) / len(text2) * 100

    def get_similarity_text_distance_damerau_levenshtein(self, text1, text2):
        if len(text1) > len(text2):
            return textdistance.damerau_levenshtein.similarity(text1, text2) / len(text1) * 100
        else:
            return textdistance.damerau_levenshtein.similarity(text1, text2) / len(text2) * 100

    def get_similarity_text_distance_prefix(self, text1, text2):
        if len(text1) > len(text2):
            return textdistance.prefix.similarity(text2, text1) / len(text1) * 100
        else:
            return textdistance.prefix.similarity(text2, text1) / len(text2) * 100

    def get_similarity_smith_waterman(self, text1, text2):
        if len(text1) > len(text2):
            return textdistance.smith_waterman.similarity(text2, text1) / len(text1) * 100
        else:
            return textdistance.smith_waterman.similarity(text2, text1) / len(text2) * 100

    def get_similarity_strcmp95(self, text1, text2):
        return textdistance.strcmp95.similarity(text2, text1) * 100

    def get_similarity_jaccard(self, text1, text2):
        return textdistance.jaccard.similarity(text2, text1) * 100

    def get_similarity_visual_free_mono(self, text1, text2):
        text1image_hash = self.text_to_image_hash(text1, '/usr/share/fonts/truetype/freefont/FreeMono.ttf', 40)
        text2image_hash = self.text_to_image_hash(text2, '/usr/share/fonts/truetype/freefont/FreeMono.ttf', 40)
        return 100 - (((text1image_hash - text2image_hash) / 35) * 100)

    def get_similarity_visual_free_sans(self, text1, text2):
        text1image_hash = self.text_to_image_hash(text1, '/usr/share/fonts/truetype/freefont/FreeSans.ttf', 40)
        text2image_hash = self.text_to_image_hash(text2, '/usr/share/fonts/truetype/freefont/FreeSans.ttf', 40)
        return 100 - (((text1image_hash - text2image_hash) / 35) * 100)

    def text_to_image_hash(self, text2draw, font_name, font_size):
        font = ImageFont.truetype(font_name, size=font_size)
        text_width, text_height = font.getsize(text2draw)
        img = Image.new('RGB', (text_width + 25, text_height + 25), 'white')
        img_draw = ImageDraw.Draw(img)
        img_draw.text((10, 10), text2draw, 'black', font=font)
        # img.save('/tmp/' + text2draw + '_' + str(self.id) + '.png', "PNG")
        text1image_hash = imagehash.phash(img)
        return text1image_hash

    def get_highest_similarity(self, text1, text2):
        highest_similarity = 0
        most_similar_value = None
        highest_similarity_algorithm = None
        for algorithm in self.algorithms:
            cur_similarity_test_res = self.algorithms[algorithm](text1, text2)
            if cur_similarity_test_res > highest_similarity:
                highest_similarity = cur_similarity_test_res
                highest_similarity_algorithm = algorithm
                most_similar_value = text2
            if cur_similarity_test_res > self.sufficient_similarity_rate:
                break
        return {
            "highest_similarity": int(highest_similarity),
            "most_similar_value": most_similar_value,
            "highest_similarity_algorithm": highest_similarity_algorithm,
            "cache_size": self.cache.size,
            "cache_len": self.cache.len()
        }

    def measure(self, text):
        self.cache.put_if_not_exist(text)
        items = self.cache.dump()
        if self.cache.len() >= self.min_cache_size:
            highest_similarity_found = 0
            highest_similarity_res = None
            for cached_item in items:
                if text != cached_item[0]:
                    res = self.get_highest_similarity(text, cached_item[0])
                    if res["highest_similarity"] > highest_similarity_found:
                        highest_similarity_res = res
                        highest_similarity_found = res["highest_similarity"]
                if highest_similarity_found > self.sufficient_similarity_rate:
                    # Reduces accuracy to improve performance
                    break
            if highest_similarity_res is None:
                return self.return_empty_similarity_result()
            else:
                highest_similarity_res["value_frequency"] = self.cache.get_value_frequency(text, self.tasks_counter)
                return highest_similarity_res
        else:
            return self.return_empty_similarity_result()

    def return_empty_similarity_result(self):
        return {
            "highest_similarity": -1,
            "most_similar_value": None,
            "highest_similarity_algorithm": None,
            "value_frequency": -1,
            "cache_size": self.cache.size,
            "cache_len": self.cache.len()
        }

    def connection_handler(self, conn):
        highest_similarity = -2
        value_frequency = -2
        time_taken = -1
        with self.threads_lock:
            self.threads_count = self.threads_count + 1
        try:
            raw_data = conn.recv(1024)
            raw_data_string = raw_data.decode()
            command_obj = json.loads(raw_data.decode())
            if "text" not in command_obj:
                print("ERROR: Command misunderstood. Command: " + base64.b64decode(raw_data_string).decode())
                conn.sendall("ERROR: Command misunderstood".encode())
            else:
                time_started = time.time()
                # print("DEBUG: Handling " + raw_data_string)
                result = self.measure(command_obj["text"])
                self.tasks_counter = self.tasks_counter + 1
                time_taken = time.time() - time_started
                with self.time_taken_mgmt_lock:
                    self.time_taken_recent5.append(time_taken)
                    self.time_taken_recent15.append(time_taken)
                    self.time_taken_recent45.append(time_taken)
                    if len(self.time_taken_recent5) > 5:
                        self.time_taken_recent5.remove(self.time_taken_recent5[0])
                    if len(self.time_taken_recent15) > 15:
                        self.time_taken_recent15.remove(self.time_taken_recent15[0])
                    if len(self.time_taken_recent45) > 45:
                        self.time_taken_recent45.remove(self.time_taken_recent45[0])
                    if time_taken > self.max_time_taken_ms:
                        self.max_time_taken_ms = time_taken
                    if len(self.time_taken_recent5) == 5:
                        avg_time_taken_recent5 = sum(self.time_taken_recent5) / len(self.time_taken_recent5)
                    else:
                        avg_time_taken_recent5 = -1
                    if len(self.time_taken_recent15) == 15:
                        avg_time_taken_recent15 = sum(self.time_taken_recent15) / len(self.time_taken_recent15)
                    else:
                        avg_time_taken_recent15 = -1
                    if len(self.time_taken_recent45) == 45:
                        avg_time_taken_recent45 = sum(self.time_taken_recent45) / len(self.time_taken_recent45)
                    else:
                        avg_time_taken_recent45 = -1
                result["time_taken"] = time_taken
                result["members_of_same_domain"] = False

                highest_similarity = result["highest_similarity"]
                value_frequency = result["value_frequency"]
                if highest_similarity >= self.min_similarity_to_report and value_frequency <= self.max_frequency_to_report:
                    extracted_tested_domain_info = tldextract.extract(command_obj["text"])
                    extracted_matched_domain_info = tldextract.extract(result["most_similar_value"])
                    if extracted_tested_domain_info.registered_domain == extracted_matched_domain_info.registered_domain:
                        result["members_of_same_domain"] = True

                    if result["members_of_same_domain"] and not self.report_members_of_same_domain:
                        conn.sendall("{}".encode())
                    else:
                        conn.sendall(json.dumps(result).encode())
                else:
                    conn.sendall("{}".encode())
        except Exception as ex:
            print("ERROR: The following exception has occurred while handling a connection. (ex: " + str(ex) + ")")
        finally:
            try:
                conn.close()
            except:
                pass
            with self.threads_lock:
                self.threads_count = self.threads_count - 1
                print("DEBUG: Finished a task no " + str(self.tasks_counter) + ". Current threads count: " + str(self.threads_count) + ", time taken (ms): " + "{:.2f}".format(time_taken) + ", max time taken (ms): " + "{:.2f}".format(self.max_time_taken_ms) + ", time taken averages (5, 15, 45): " + "{:.2f}".format(avg_time_taken_recent5) + " " + "{:.2f}".format(avg_time_taken_recent15) + " " + "{:.2f}".format(avg_time_taken_recent45) + ", highest_similarity: " + str(highest_similarity) + ", value_frequency: " + str(value_frequency))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Service for detecting similarity between texts")
    args_parser.add_argument("--socket-file", required=True,
                             help="The socket file path on which the service will listen", type=str)
    args_parser.add_argument("--min-cache-size", required=True, help="Minimum cache size to return results", type=int)
    args_parser.add_argument("--max-cache-size", required=True, help="Maximum number of entries to hold in the cache",
                             type=int)
    args_parser.add_argument("--sufficient-similarity-rate", default=90,
                             help="A good-enough similarity rate in percentage at which the service is required to stop looking for better matches. Using a lower value for this (defaults to 90) can improve the service perfomance but reduce its accuracy",
                             type=int)

    args = args_parser.parse_args()
    socket_path = os.path.realpath(args.socket_file)

    print("Starting with default parameters... (socket: " + socket_path + ")")
    if os.path.realpath(sys.argv[0]) == socket_path:
        print("ERROR: The socket path refers to the script itself. Something is wrong in the arguments list.")
        args_parser.print_help()
        exit(8)

    while True:
        try:
            server = SimilisServer(socket_path=socket_path, min_cache_size=args.min_cache_size,
                                   max_cache_size=args.max_cache_size,
                                   sufficient_similarity_rate=args.sufficient_similarity_rate)
            server.start()
        except KeyboardInterrupt:
            break
