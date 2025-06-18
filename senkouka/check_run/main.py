# :issue モジュールの相対パス
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.random import Random as r

table = [1, 2, 3, 4, 5]  # 例としてのテーブル
result = r.roulett(table)
print(f"選ばれたインデックス: {result}")
