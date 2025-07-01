import numpy as np
import path_planning as pp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

print("="*80)
print("6次元変換と経路描画の詳細解析")
print("="*80)

# 環境作成
env = pp.Environment(width=100, height=100, robot_radius=1, start=[5,5], goal=[95,95])

# ===== 部分1: 6次元変数の準備 =====
print("\n" + "="*50)
print("部分1: 6次元変数の準備")
print("="*50)

original_6d = [0.3, 0.4, 0.6, 0.7, 0.8, 0.2]
print(f"元の6次元変数: {original_6d}")
print(f"データ型: {type(original_6d)}")
print(f"要素数: {len(original_6d)}")
print(f"構造: [x1, y1, x2, y2, x3, y3] = 3つの制御点の正規化座標")

# ===== 部分2: SplinePath.from_list の詳細解析 =====
print("\n" + "="*50)
print("部分2: SplinePath.from_list の詳細解析")
print("="*50)

print("ステップ2-1: np.array(xy).reshape(-1, 2)")
xy_array = np.array(original_6d)
print(f"  np.array変換後: {xy_array}")
print(f"  形状: {xy_array.shape}")

control_points_raw = xy_array.reshape(-1, 2)
print(f"  reshape(-1, 2)後: \n{control_points_raw}")
print(f"  形状: {control_points_raw.shape}")

print("\nステップ2-2: 正規化座標 → 実座標変換")
print(f"  環境サイズ: width={env.width}, height={env.height}")

control_points = control_points_raw.copy()
control_points[:,0] *= env.width   # x座標のスケーリング
control_points[:,1] *= env.height  # y座標のスケーリング

print(f"  変換前: \n{control_points_raw}")
print(f"  変換後: \n{control_points}")

print("\nステップ2-3: SplinePathオブジェクト作成")
sol = pp.SplinePath(env, control_points, resolution=50)
print(f"  作成されたオブジェクト: {type(sol)}")
print(f"  制御点: \n{sol.control_points}")
print(f"  解像度: {sol.resolution}")

# ===== 部分3: get_path() の詳細解析 =====
print("\n" + "="*50)
print("部分3: get_path() の詳細解析")
print("="*50)

print("ステップ3-1: スタートとゴールの追加")
start = env.start
goal = env.goal
print(f"  スタート地点: {start}")
print(f"  ゴール地点: {goal}")
print(f"  制御点: \n{sol.control_points}")

points = np.vstack((start, sol.control_points, goal))
print(f"  結合後の点群: \n{points}")
print(f"  点群の形状: {points.shape}")
print(f"  点の順序: start → 制御点1 → 制御点2 → 制御点3 → goal")

print("\nステップ3-2: スプライン補間パラメータの設定")
t = np.linspace(0, 1, len(points))
print(f"  点の数: {len(points)}")
print(f"  パラメータt: {t}")
print(f"  各点に対応: {[f't{i}={t[i]:.2f}' for i in range(len(t))]}")

print("\nステップ3-3: CubicSpline作成")
cs = CubicSpline(t, points, bc_type='clamped')
print(f"  スプラインオブジェクト: {type(cs)}")
print(f"  境界条件: clamped (両端で1次微分=0)")

print("\nステップ3-4: 50点の経路生成")
tt = np.linspace(0, 1, sol.resolution)
print(f"  補間点数: {sol.resolution}")
print(f"  補間パラメータttの範囲: {tt[0]:.3f} → {tt[-1]:.3f}")
print(f"  補間パラメータttの例: {tt[:5]}")

path = cs(tt)
print(f"  生成された経路形状: {path.shape}")
print(f"  最初の5点: \n{path[:5]}")

print("\nステップ3-5: 環境クリッピング")
path_clipped = env.clip_path(path)
print(f"  クリッピング前後の形状: {path.shape} → {path_clipped.shape}")
print(f"  クリッピング後の最初の5点: \n{path_clipped[:5]}")

# ===== 部分4: 経路描画の詳細解析 =====
print("\n" + "="*50)
print("部分4: 経路描画の詳細解析")
print("="*50)

# 実際の描画テスト
fig, ax = plt.subplots(figsize=(8, 8))

print("ステップ4-1: 環境の描画")
pp.plot_environment(env, ax=ax)
print("  ✓ 環境、障害物、スタート、ゴールを描画")

print("\nステップ4-2: 経路の描画")
path_line = pp.plot_path(sol, ax=ax, color='blue', linewidth=2)
print(f"  描画された経路オブジェクト: {type(path_line)}")
print(f"  経路の点数: {len(path_line.get_xdata())}")

print("\nステップ4-3: 描画データの確認")
x_data = path_line.get_xdata()
y_data = path_line.get_ydata()
print(f"  X座標データ形状: {x_data.shape}")
print(f"  Y座標データ形状: {y_data.shape}")
print(f"  最初の5点のX座標: {x_data[:5]}")
print(f"  最初の5点のY座標: {y_data[:5]}")

ax.set_title("6次元変数から生成された経路")
ax.grid(True)
plt.savefig('path_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ 画像を'path_analysis.png'として保存")
plt.close()

# ===== 部分5: 変換プロセスの総まとめ =====
print("\n" + "="*50)
print("部分5: 変換プロセスの総まとめ")
print("="*50)

print("完全な変換フロー:")
print("1. 6次元変数 [0.3, 0.4, 0.6, 0.7, 0.8, 0.2]")
print("    ↓")
print("2. reshape → [[0.3, 0.4], [0.6, 0.7], [0.8, 0.2]]")
print("    ↓")
print("3. スケーリング → [[30, 40], [60, 70], [80, 20]]")
print("    ↓")
print("4. start+制御点+goal → [[5,5], [30,40], [60,70], [80,20], [95,95]]")
print("    ↓")
print("5. CubicSpline補間 → 50個の滑らかな経路点")
print("    ↓")
print("6. matplotlib描画 → 画面上の経路線")

print("\n各段階での情報:")
print(f"  6次元変数:     6個の float値")
print(f"  制御点:       3個の (x,y) 座標")
print(f"  補間点群:     5個の (x,y) 座標 (start+制御点+goal)")
print(f"  最終経路:     50個の (x,y) 座標")
print(f"  描画データ:   50個の連続線分")

# ===== 部分6: 実装コードの場所 =====
print("\n" + "="*50)
print("部分6: 実装コードの場所")
print("="*50)

print("重要なメソッドの場所:")
print("1. SplinePath.from_list():")
print("   ファイル: path_planning/solution.py")
print("   行: 20-26")
print("   機能: 6次元リスト → SplinePathオブジェクト")

print("\n2. SplinePath.get_path():")
print("   ファイル: path_planning/solution.py") 
print("   行: 29-44")
print("   機能: 制御点 → 50点経路")

print("\n3. plot_path():")
print("   ファイル: path_planning/plots.py")
print("   行: 3-7")
print("   機能: 経路の描画")

print("\n4. update_path():")
print("   ファイル: path_planning/plots.py")
print("   行: 9-14")
print("   機能: 経路の更新（アニメーション用）")

print("\n" + "="*80)
print("解析完了！")
print("="*80) 