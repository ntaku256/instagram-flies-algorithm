import numpy as np
import path_planning as pp

# 環境作成
env = pp.Environment(width=100, height=100, robot_radius=1, start=[5,5], goal=[95,95])

# 6次元の最適化変数
position_6d = [0.3, 0.4, 0.6, 0.7, 0.8, 0.2]
print('6次元最適化変数:', position_6d)
print()

# 同じ6次元から複数回経路生成
sol1 = pp.SplinePath.from_list(env, position_6d, 50, normalized=True)
sol2 = pp.SplinePath.from_list(env, position_6d, 50, normalized=True)
sol3 = pp.SplinePath.from_list(env, position_6d, 50, normalized=True)

path1 = sol1.get_path()
path2 = sol2.get_path()
path3 = sol3.get_path()

# 経路が完全に同じかチェック
def paths_identical(p1, p2, tolerance=1e-10):
    return np.allclose(p1, p2, atol=tolerance)

print('経路の同一性チェック:')
print(f'経路1と経路2が同じ: {paths_identical(path1, path2)}')
print(f'経路1と経路3が同じ: {paths_identical(path1, path3)}')
print(f'経路2と経路3が同じ: {paths_identical(path2, path3)}')
print()

# 数値的な比較
max_diff_12 = np.max(np.abs(path1 - path2))
max_diff_13 = np.max(np.abs(path1 - path3))
max_diff_23 = np.max(np.abs(path2 - path3))

print('経路間の最大絶対差:')
print(f'経路1と経路2: {max_diff_12:.2e}')
print(f'経路1と経路3: {max_diff_13:.2e}')
print(f'経路2と経路3: {max_diff_23:.2e}')
print()

# いくつかの点を確認
print('特定点の座標比較:')
for i in [0, 10, 25, 40, 49]:
    print(f'点{i+1}:')
    print(f'  経路1: [{path1[i,0]:.6f}, {path1[i,1]:.6f}]')
    print(f'  経路2: [{path2[i,0]:.6f}, {path2[i,1]:.6f}]')
    print(f'  経路3: [{path3[i,0]:.6f}, {path3[i,1]:.6f}]')
    print()

# 制御点の確認
print('制御点の比較:')
print(f'sol1の制御点: {sol1.control_points}')
print(f'sol2の制御点: {sol2.control_points}')
print(f'sol3の制御点: {sol3.control_points}')

# さらに確認：異なる6次元から異なる経路が生成されることの確認
print('\n' + '='*50)
print('異なる6次元変数での比較:')

position_6d_different = [0.3, 0.4, 0.1, 0.9, 0.5, 0.6]
sol_diff = pp.SplinePath.from_list(env, position_6d_different, 50, normalized=True)
path_diff = sol_diff.get_path()

max_diff_original_vs_different = np.max(np.abs(path1 - path_diff))
print(f'元の経路と異なる6次元からの経路の最大差: {max_diff_original_vs_different:.2f}')
print(f'異なる6次元変数: {position_6d_different}') 