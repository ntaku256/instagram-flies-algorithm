import numpy as np
import path_planning as pp
import matplotlib.pyplot as plt

print("="*80)
print("6次元変数による経路表現の複雑さ限界テスト")
print("="*80)

# 環境作成（複雑な障害物配置）
env = pp.Environment(width=100, height=100, robot_radius=1, start=[5,5], goal=[95,95])

# 複雑な迷路のような障害物
complex_obstacles = [
    {'center': [20, 20], 'radius': 8},
    {'center': [30, 40], 'radius': 6},
    {'center': [25, 60], 'radius': 7},
    {'center': [45, 25], 'radius': 9},
    {'center': [40, 55], 'radius': 8},
    {'center': [60, 35], 'radius': 7},
    {'center': [70, 15], 'radius': 6},
    {'center': [75, 50], 'radius': 8},
    {'center': [55, 70], 'radius': 7},
    {'center': [80, 75], 'radius': 6},
]

for obs in complex_obstacles:
    env.add_obstacle(pp.Obstacle(**obs))

print("複雑な障害物環境を作成しました")

# ===== テスト1: 3つの制御点（6次元）での経路表現 =====
print("\n" + "="*60)
print("テスト1: 3つの制御点（6次元）での経路表現")
print("="*60)

def test_path_complexity(num_control_points, title):
    print(f"\n--- {title} ---")
    
    # 迷路を避けるような複雑な経路を手動設計
    if num_control_points == 3:
        # 6次元: 3つの制御点
        complex_path_6d = [0.15, 0.8, 0.5, 0.3, 0.85, 0.6]
        sol = pp.SplinePath.from_list(env, complex_path_6d, 50, normalized=True)
    elif num_control_points == 5:
        # 10次元: 5つの制御点
        complex_path_10d = [0.12, 0.7, 0.25, 0.4, 0.4, 0.2, 0.65, 0.8, 0.85, 0.6]
        sol = pp.SplinePath.from_list(env, complex_path_10d, 50, normalized=True)
    elif num_control_points == 8:
        # 16次元: 8つの制御点
        complex_path_16d = [0.1, 0.6, 0.2, 0.3, 0.3, 0.8, 0.45, 0.2, 0.6, 0.9, 0.75, 0.3, 0.85, 0.7, 0.9, 0.8]
        sol = pp.SplinePath.from_list(env, complex_path_16d, 50, normalized=True)
    
    path = sol.get_path()
    
    # 経路の複雑さ指標を計算
    def calculate_complexity_metrics(path):
        # 1. 方向変化の総和
        diff_vectors = np.diff(path, axis=0)
        angles = np.arctan2(diff_vectors[:, 1], diff_vectors[:, 0])
        angle_changes = np.abs(np.diff(angles))
        # 角度の不連続性を修正
        angle_changes[angle_changes > np.pi] = 2*np.pi - angle_changes[angle_changes > np.pi]
        total_direction_change = np.sum(angle_changes)
        
        # 2. 経路の曲率
        curvature_sum = np.sum(angle_changes)
        
        # 3. 最大角度変化
        max_angle_change = np.max(angle_changes) if len(angle_changes) > 0 else 0
        
        # 4. 経路長
        path_length = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
        
        return {
            'total_direction_change': total_direction_change,
            'curvature_sum': curvature_sum,
            'max_angle_change': max_angle_change,
            'path_length': path_length,
            'num_segments': len(path) - 1
        }
    
    metrics = calculate_complexity_metrics(path)
    
    print(f"制御点数: {num_control_points}")
    print(f"総方向変化: {metrics['total_direction_change']:.3f} ラジアン")
    print(f"最大角度変化: {metrics['max_angle_change']:.3f} ラジアン ({np.degrees(metrics['max_angle_change']):.1f}度)")
    print(f"経路長: {metrics['path_length']:.2f}")
    print(f"平均角度変化: {metrics['total_direction_change'] / metrics['num_segments']:.4f} ラジアン/セグメント")
    
    # 衝突検出
    collision_count, details = env.count_violations(path)
    print(f"衝突回数: {details['collision_violation_count']}")
    
    # 経路の描画
    fig, ax = plt.subplots(figsize=(10, 10))
    pp.plot_environment(env, ax=ax)
    pp.plot_path(sol, ax=ax, color='red', linewidth=3, label=f'{num_control_points}制御点')
    
    # 制御点もプロット
    for i, cp in enumerate(sol.control_points):
        ax.plot(cp[0], cp[1], 'bo', markersize=10, label=f'制御点{i+1}' if i == 0 else "")
    
    ax.set_title(f'{title}\n総方向変化: {metrics["total_direction_change"]:.2f}, 衝突: {details["collision_violation_count"]}')
    ax.legend()
    ax.grid(True)
    plt.savefig(f'complexity_test_{num_control_points}cp.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics, details

# 異なる制御点数でテスト
results = []
results.append(test_path_complexity(3, "3つの制御点（6次元）"))
results.append(test_path_complexity(5, "5つの制御点（10次元）"))
results.append(test_path_complexity(8, "8つの制御点（16次元）"))

# ===== 結果の比較 =====
print("\n" + "="*60)
print("結果の比較とまとめ")
print("="*60)

print("\n制御点数と表現能力の関係:")
print("-" * 40)
for i, ((metrics, details), num_cp) in enumerate(zip(results, [3, 5, 8])):
    print(f"{num_cp}制御点: 総方向変化={metrics['total_direction_change']:.3f}, "
          f"最大角度変化={np.degrees(metrics['max_angle_change']):.1f}度, "
          f"衝突={details['collision_violation_count']}")

# ===== 理論的限界の解説 =====
print("\n" + "="*60)
print("6次元変数（3制御点）の理論的限界")
print("="*60)

print("\n1. 表現能力の限界:")
print("   - 5点のスプライン補間: start → cp1 → cp2 → cp3 → goal")
print("   - 3次スプライン: 非常に滑らかな曲線のみ")
print("   - 急激な方向転換は不可能")
print("   - ジグザグや複雑な迂回は困難")

print("\n2. 複雑な経路が必要な場面:")
print("   - 迷路のような環境")
print("   - 密集した障害物")
print("   - 狭い通路の連続")
print("   - U字ターンが必要な場合")

print("\n3. 解決策:")
print("   - 制御点数を増やす（10次元、16次元など）")
print("   - 異なる補間方法（折れ線、B-spline）")
print("   - 階層的経路計画")
print("   - 局所的な軌道修正")

print("\n4. 6次元変数の適用範囲:")
print("   ✅ 適している: シンプルな環境、少数の大きな障害物")
print("   ❌ 不適切: 複雑な迷路、密集障害物、急カーブ")

print("\n" + "="*80)
print("テスト完了！画像ファイルを確認してください。")
print("="*80) 