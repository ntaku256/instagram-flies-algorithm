import numpy as np
import path_planning as pp
from scipy.optimize import minimize

# 環境作成
env = pp.Environment(width=100, height=100, robot_radius=1, start=[5,5], goal=[95,95])

print("="*60)
print("6次元と50点の双方向変換分析")
print("="*60)

# ===== 1. 6次元 → 50点変換 (順変換) =====
print("\n1. 順変換：6次元 → 50点")
print("-" * 30)

original_6d = [0.3, 0.4, 0.6, 0.7, 0.8, 0.2]
print(f"元の6次元変数: {original_6d}")

# 50点経路生成
sol = pp.SplinePath.from_list(env, original_6d, 50, normalized=True)
path_50points = sol.get_path()

print(f"生成された50点経路の形状: {path_50points.shape}")
print(f"制御点: {sol.control_points}")

# ===== 2. 50点 → 6次元変換 (逆変換) の試行 =====
print("\n2. 逆変換：50点 → 6次元")
print("-" * 30)

def path_to_6d_objective(variables_6d, target_path, env):
    """50点経路に最も近い6次元変数を見つける目的関数"""
    try:
        # 6次元から経路生成
        sol_test = pp.SplinePath.from_list(env, variables_6d, 50, normalized=True)
        path_test = sol_test.get_path()
        
        # 目標経路との差の二乗和
        diff = np.sum((path_test - target_path)**2)
        return diff
    except:
        return 1e10

# 最適化による逆変換
print("最適化による逆変換を試行中...")

# 初期推定値
initial_guess = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
bounds = [(0, 1)] * 6

result = minimize(
    path_to_6d_objective, 
    initial_guess, 
    args=(path_50points, env),
    bounds=bounds,
    method='L-BFGS-B'
)

recovered_6d = result.x
print(f"回復された6次元変数: {recovered_6d}")
print(f"最適化成功: {result.success}")
print(f"目的関数値: {result.fun:.2e}")

# ===== 3. 回復確認 =====
print("\n3. 回復確認")
print("-" * 30)

# 回復された6次元から経路生成
sol_recovered = pp.SplinePath.from_list(env, recovered_6d, 50, normalized=True)
path_recovered = sol_recovered.get_path()

# 比較
max_diff = np.max(np.abs(path_50points - path_recovered))
rmse = np.sqrt(np.mean((path_50points - path_recovered)**2))

print(f"元の6次元:     {original_6d}")
print(f"回復された6次元: {[f'{x:.6f}' for x in recovered_6d]}")
print()
print(f"経路の最大絶対差: {max_diff:.2e}")
print(f"経路のRMSE:      {rmse:.2e}")

# ===== 4. 一意性の検証 =====
print("\n4. 一意性の検証")
print("-" * 30)

# 異なる初期値から複数回最適化
print("異なる初期値から複数回逆変換:")
for i in range(3):
    initial_random = np.random.rand(6)
    result_i = minimize(
        path_to_6d_objective, 
        initial_random, 
        args=(path_50points, env),
        bounds=bounds,
        method='L-BFGS-B'
    )
    print(f"試行{i+1}: {[f'{x:.6f}' for x in result_i.x]} (目的関数値: {result_i.fun:.2e})")

# ===== 5. 情報量の比較 =====
print("\n5. 情報量の比較")
print("-" * 30)

print(f"6次元変数:")
print(f"  要素数: 6")
print(f"  自由度: 6 (各要素0-1の範囲)")
print(f"  情報量: 連続値×6")

print(f"\n50点経路:")
print(f"  要素数: 50×2 = 100")
print(f"  制約: スタート/ゴール固定、スプライン曲線")
print(f"  有効自由度: 理論的には6 (制御点由来)")

print(f"\n変換の性質:")
print(f"  6次元 → 50点: 決定論的、一意")
print(f"  50点 → 6次元: 理論的には可能、実用的には困難") 