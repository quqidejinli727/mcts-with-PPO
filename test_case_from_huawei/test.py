import json
from collections import Counter
from pathlib import Path

def load_chains(file_path: str | Path):
    """把 pingroup.json 读成 Python 对象"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fingerprint(chain):
    """
    单条链的指纹：每个节点用 (parent_module, parent_inst) 表示
    整条链就是一个不可变的元组
    """
    return tuple((item['parent_module'], item['parent_inst']) for item in chain)

def count_same_topology(chains):
    """统计指纹完全相同的链各出现了多少次"""
    return Counter(fingerprint(ch) for ch in chains)

def main():
    chains = load_chains('pin_assignment/test _case from huawei/pingroup.json')
    stats = count_same_topology(chains)

    # 按出现次数降序打印
    for topo, cnt in stats.most_common():
        # 把指纹格式化成易读字符串
        path_str = ' -> '.join(f'{m}[{i}]' for m, i in topo)
        print(f'{cnt:>4} 次 : {path_str}')

if __name__ == '__main__':
    main()