import json
from tqdm import tqdm


def convert():
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_box.json', encoding='utf-8'))
    conversations2 = []
    for c in tqdm(conversations):
        conversations2.append({
            'image': '/mnt/ceph2/datasets/' + c['image'],
            'conversations': [
                {
                    'role': 'user',
                    'content': c['conversations'][0]['value']
                },
                {
                    'role': 'assistant',
                    'content': c['conversations'][1]['value']
                }
            ]
        })
    json.dump(conversations2, open('conversations_tiku_box.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    convert()
