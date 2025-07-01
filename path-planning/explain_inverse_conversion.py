import numpy as np
import path_planning as pp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 環境作成
env = pp.Environment(width=100, height=100, robot_radius=1, start=[5,5], goal=[95,95])

print("="*70)
print("50点→6次元変換の詳細解説")
print("="*70)

# ===== ステップ1: 目標となる50点経路の準備 =====
print("\nステップ1: 目標となる50点経路の準備")
print("-" * 50)

# 真の6次元変数
true_6d = [0.2, 0.8, 0.7, 0.3, 0.9, 0.1]
print(f"真の6次元変数: {true_6d}")

# 真の6次元から50点経路を生成（これが目標）
sol_true = pp.SplinePath.from_list(env, true_6d, 50, normalized=True)
target_path = sol_true.get_path()

print(f"目標経路の形状: {target_path.shape}")
print(f"目標経路の最初の3点:")
for i in range(3):
    print(f"  点{i+1}: [{target_path[i,0]:.2f}, {target_path[i,1]:.2f}]")

# ===== ステップ2: 推測による6次元変数の試行 =====
print("\nステップ2: 推測による6次元変数の試行")
print("-" * 50)

def calculate_error(guess_6d, target_path, env, verbose=False):
    """推測した6次元から生成される経路と目標経路の誤差を計算"""
    try:
        # 推測6次元から経路生成
        sol_guess = pp.SplinePath.from_list(env, guess_6d, 50, normalized=True)
        guess_path = sol_guess.get_path()
        
        # 各点での距離の二乗を計算
        point_errors = np.sum((guess_path - target_path)**2, axis=1)
        total_error = np.sum(point_errors)
        
        if verbose:
            print(f"  推測6次元: {[f'{x:.3f}' for x in guess_6d]}")
            print(f"  総誤差: {total_error:.2e}")
            print(f"  最大点誤差: {np.max(point_errors):.2e}")
            print(f"  平均点誤差: {np.mean(point_errors):.2e}")
        
        return total_error
    except:
        return 1e10

# いくつかの推測を試してみる
print("様々な推測での誤差:")
guesses = [
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 中央値
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 最小値
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 最大値
    [0.1, 0.9, 0.8, 0.2, 0.8, 0.2],  # ランダム1
    [0.3, 0.7, 0.6, 0.4, 0.9, 0.0],  # ランダム2
]

for i, guess in enumerate(guesses):
    error = calculate_error(guess, target_path, env, verbose=True)
    print()

# ===== ステップ3: 最適化アルゴリズムの動作 =====
print("\nステップ3: 最適化アルゴリズムの動作")
print("-" * 50)

# 最適化の各ステップを記録
optimization_history = []

def objective_with_history(variables_6d, target_path, env):
    """履歴を記録する目的関数"""
    error = calculate_error(variables_6d, target_path, env)
    optimization_history.append({
        'variables': variables_6d.copy(),
        'error': error
    })
    return error

print("最適化を実行中...")

# 初期推定値
initial_guess = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
bounds = [(0, 1)] * 6

result = minimize(
    objective_with_history,
    initial_guess,
    args=(target_path, env),
    bounds=bounds,
    method='L-BFGS-B',
    options={'maxiter': 100}
)

print(f"最適化完了: {len(optimization_history)}回の評価")
print(f"初期誤差: {optimization_history[0]['error']:.2e}")
print(f"最終誤差: {optimization_history[-1]['error']:.2e}")
print(f"誤差改善率: {optimization_history[0]['error'] / optimization_history[-1]['error']:.1e}倍")

# ===== ステップ4: 最適化過程の可視化 =====
print("\nステップ4: 最適化過程の詳細")
print("-" * 50)

print("最適化の主要ステップ:")
key_steps = [0, len(optimization_history)//4, len(optimization_history)//2, 
             3*len(optimization_history)//4, len(optimization_history)-1]

for step in key_steps:
    if step < len(optimization_history):
        h = optimization_history[step]
        print(f"ステップ{step+1:3d}: {[f'{x:.3f}' for x in h['variables']]} -> 誤差: {h['error']:.2e}")

# ===== ステップ5: 結果の検証 =====
print("\nステップ5: 結果の検証")
print("-" * 50)

recovered_6d = result.x
print(f"真の6次元:     {true_6d}")
print(f"回復された6次元: {[f'{x:.6f}' for x in recovered_6d]}")

# 差の計算
diff_6d = np.abs(np.array(true_6d) - np.array(recovered_6d))
print(f"6次元での差:   {[f'{x:.6f}' for x in diff_6d]}")
print(f"最大差:       {np.max(diff_6d):.6f}")

# 経路での検証
sol_recovered = pp.SplinePath.from_list(env, recovered_6d, 50, normalized=True)
path_recovered = sol_recovered.get_path()

path_diff = np.max(np.abs(target_path - path_recovered))
print(f"経路での最大差: {path_diff:.2e}")

# ===== ステップ6: なぜこの方法が機能するのか =====
print("\nステップ6: なぜこの方法が機能するのか")
print("-" * 50)

print("1. 目的関数の意味:")
print("   - 推測した6次元 → 50点経路生成")
print("   - 生成経路と目標経路の各点での距離を計算")
print("   - 50個の距離の二乗和が目的関数")

print("\n2. 最適化の仕組み:")
print("   - 6次元空間で目的関数を最小化")
print("   - 勾配降下法的手法で改善方向を探索")
print("   - 目的関数=0に近づくほど経路が一致")

print("\n3. 一意性の理由:")
print("   - 6次元→50点変換は単射関数（一対一対応）")
print("   - 同じ経路を生成する異なる6次元は存在しない")
print("   - よって逆変換も一意に決まる")

print("\n4. 成功の条件:")
print("   - 目標経路がスプライン経路として表現可能")
print("   - 制御点が3個で十分な複雑度")
print("   - 最適化が局所最適解に陥らない") 