# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as anime
# import openpyxl
# import random
# from sklearn.cluster import KMeans
# from mpl_toolkits.mplot3d import Axes3D

# from benchmark_function import BenchmarkFunction as BF

# def roulettMin(table, offset=0):
#     total = 0
#     for i in range(len(table)):
#         total += 1 / (offset + table[i])
#     rand = random.uniform(0.0, total)
#     sum = 0
#     for i in range(len(table)):
#         sum += 1 / (offset + table[i])
#         if(sum > rand):
#             return i
        

# if __name__ == "__main__":
#     import openpyxl

#     # Excelファイルを読み込む
#     write_wb = openpyxl.load_workbook("results/Book_write.xlsx")
#     write_ws = write_wb["Sheet1"]

#     # 数値に変換したいセル範囲を指定
#     for row in write_ws.iter_rows(min_row=1, max_row=1000, min_col=1, max_col=1):
#         for cell in row:
#             if isinstance(cell.value, str):
#                 # 数値に変換可能かチェックし、変換する
#                 try:
#                     converted_value = float(cell.value.replace(",", ""))  # カンマを除去してから数値に変換
#                     cell.value = converted_value
#                     cell.number_format = 'General'  # 書式を数値形式に設定
#                 except ValueError:
#                     # 数値に変換できない場合はそのままにする
#                     pass

#     # Excelファイルを保存
#     write_wb.save("results/Book_write_mastered_pso_converted.xlsx")

import os
print("path=\n\n")
print(os.path.abspath(__file__))