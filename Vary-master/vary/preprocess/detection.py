import os
import json
import sys
import requests
import multiprocessing
from tqdm import tqdm
from glob import glob


def detect(files, progress_queue):
    url = "http://10.33.10.63:5003/image_to_json"
    headers = {}
    payload = {}
    for path in files:
        try:
            files = [('image', (path, open(path, 'rb'), 'image/jpeg'))]
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            data = json.loads(response.text)
            json.dump(data, open(path[:-4]+'_det.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
            progress_queue.put(('success', path))
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path))
            continue


def detect_parallel(files, progress_queue):
    chunk_size = len(files) // 8
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=detect, args=(chunk, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'])
        except Exception as e:
            if all(not runner.is_alive() for runner in runners):
                break
            continue
        progress_bar.update(1)
    progress_bar.close()
    print(counts)


def detect_tiku(parallel):
    files = glob('/mnt/ceph2/datasets/tiku7/images*/*.jpg', recursive=True)
    print(len(files))
    progress_queue = multiprocessing.Queue()
    if parallel:
        detect_parallel(files, progress_queue)
    else:
        detect(files, progress_queue)


if __name__ == '__main__':
    parallel = False if sys.gettrace() is not None else True
    detect_tiku(parallel)
