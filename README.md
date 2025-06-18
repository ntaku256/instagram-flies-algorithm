# オセロ

# PSO(粒子群最適化)
- 群知能の一種で、生物の群れを粒子として模倣している
- 昆虫の群れなどにおいては，適当な経路を発見した一匹に残りの群れが素早く追従することができるが，これを多次元空間において位置と速度を持つ粒子群でモデル化したもの

<img src="https://github.com/ntaku256/AI/blob/main/Source/PSO.png" width="80%">

https://qiita.com/opticont/items/04a5b4ff41483966987f



# オセロAI
### 評価マップ
- マップに価値をつけ、おける場所の中からvalueが高い場所を選択する。
- 裏返したマスの価値の総和が最大となる手を選択してゆく。
- アルゴリズムでは、この価値マップをオセロで多くの駒をとれるように何度も試行錯誤して値を変えていく。
<img src="https://github.com/ntaku256/AI/blob/main/Source/StoneMap.png" width="50%">

### スコアの計算
- 左の図の例では、右の図のような場所を選択[7][1]するとvalueが (-3) + (-1) + (-1) = -5 になる。
<img src="https://github.com/ntaku256/AI/blob/main/Source/OseroMap.png" width="100%">

```
置ける場所とvalue
  [2][5].value = -1
  [4][5].value = -1
  [5][5].value = -1
  [6][3].value = -1
  [6][4].value = -3
  [6][5].value = -1
  [7][1].value = -5

  ※同じ値では上から選ばれるので、今回は[2][5]が選ばれる
```
# オセロAIの評価の種類
### 定石
### 候補数(おけるマスの数)
- 置けるマスが自分が多く、相手が少ない方が評価が高い
### 囲い具合(どれだけ自石が空マスに接しているか)
### 評価マップ
- 8*8のマスすべてに価値を付ける
1. 自身の色があるマスの値を加算、相手の色があるマスの値を減算して評価
2. 裏返したマスの価値の総和が最大となる手で評価
- 引用文献

https://www.info.kindai.ac.jp/~takasi-i/thesis/2012_09-1-037-0133_S_Shiota_thesis.pdf

# PSO
1. それぞれの点(vector)に、各マスのvalueが-30~30の価値マップをランダムに生成する。
```python
  def InitIndivisual():
    #8*8のマスそれぞれに-30 ~ 30 の価値を付与する。
    return np.random.uniform(-30,30,(8*8))
```
- 評価マップ
```
vector =
[  24.50813671  -29.2878167     9.04794791  -20.67778499    9.04018821   -2.40647408   -8.3115887    27.01061412
   -5.23585605  -10.35255287  -22.7822244    16.51181067  -15.84991705    6.78862383  -21.22315287   -7.65030773
  -27.69997706   22.56736182  -14.53154593   19.6416465     9.66053401   10.03510166  -27.20319744   -1.45975759
   29.59072925   25.34566733  -27.62213873   22.82363253   28.25320931    6.08110748  -13.87657254   13.78007481
  -20.27975359   -8.29864857   21.10178124   29.11842049   -6.60816536    9.95253957    8.94232069   25.37094617
    7.3748964    -7.15071738   10.60151921    5.9180152    15.87395565   -1.39311164  -17.46658634   26.40022369
   10.29530317   29.42593735  -14.95372671  -28.78418615   19.09774394  -16.58776741  -20.73929498    7.31926295
   17.8158737    24.30464656   22.28706443    8.22550678   24.77724914    5.77699278  -26.36270084    1.7537956 ]
```
2. 敵(enemy)とし評価マップ(vector)をランダムに一つ選び、全ての評価マップ(vector)と順番にオセロで対戦させる。
3. 順番に対戦した評価マップ(vector)は、結果のスコア(自分の色のマス+空白のマス)と自分の評価マップ(vector)内で一番スコアが高かったものと比較し、それよりも高かった場合はp_bestを更新してその評価マップ(vector)をp_best_vectorとして記録する。
4. 結果のスコア(自分の色のマス+空白のマス)と全部の評価マップ(vector)の中で一番スコアが高かったものとも比較し、それよりも高ければg_bestを更新してその評価マップ(vector)をg_best_vectorとして記録する。
```python
  def CalcScores(self):
          #対戦する相手をランダムに決める
          enemy = np.random.randint(0,self.n_swarm,1)

          #決まった評価マップ(vector)と全部の評価マップ(vector)をそれぞれ対戦させる
          for i in range(self.n_swarm):
              new_score = EvalIndivisual(self.vectors[i],self.vectors[enemy])

              #自身の評価マップ(vector)のベストスコアと比較し、大きければ更新、評価マップを保存
              if new_score > self.p_best_scores[i]:
                  self.p_best_scores[i] = new_score
                  self.p_best_vectors[i] = np.copy(self.vectors[i])

              #全評価マップ(vector)のベストスコアと比較し、大きければ更新、評価マップを保存
              if new_score > self.g_best_score:
                  self.g_best_score = new_score
                  self.g_best_vector = np.copy(self.vectors[i])
```
```
  スコア : (オセロの結果=自分の色があるマス+開いているマス)
  p_best_score[12] = 40 : 12番目の評価マップ(vector)内の自己ベストが40 
  g_best_score = 50 : 全評価マップ(vector)の中でベストスコアが50
```
- 外部モジュール
```python
  # min から max までの整数をランダムに n 回とる
  np.random.randint(mini,max,n)

  # 型、値など全てを複製
  np.copy()

  # min から max までの数をランダムに 8*8=64 回とる
  np.random.uniform(min,max,(8*8))
```
5. 評価マップ(vector)の移動速度を求め、評価マップ(vector)を移動させる
   - 評価マップ(vector)の移動速度は、もとの速度・p_best_vectorと現在の評価マップ(vector)との距離・g_best_vectorと現在の評価マップ(vector)との距離に重みを付けて足し合わせる。
```python
  def UpdateVectors(self):
          #全点(vector)の位置をそれぞれ更新する。
          for i in range(self.n_swarm):
              #p_best(自己ベストスコア)の重みをランダムにつける。
              r1 = np.random.uniform(0,1,self.vectors[0].shape)
              #g_best(全ベストスコア)の重みをランダムにつける。
              r2 = np.random.uniform(0,1,self.vectors[0].shape)

              #最適解(p_best,g_best)に近づくように評価マップ(vector)の移動速度を計算する。
              self.speeds[i] = self.w*self.speeds[i]+r1*(self.p_best_vectors[i]-self.vectors[i])+r2*(self.g_best_vector-self.vectors[i])

              #評価マップ(vector)の位置を更新する。
              self.vectors[i] = self.vectors[i] + self.speeds[i]
```
- 点(vector)の移動速度(speeds)
```
speeds[2] =
[-2.34390529e+01 -3.22353587e+01 -3.22654493e-01  3.61578725e+01
  -7.38940442e+00  2.06039252e+01 -1.60904841e+01 -1.93745703e+01
  -1.54948031e+01  2.41069919e+00  6.75141051e+00 -1.28848487e+00
  -8.13728253e+00  2.89531148e+01 -3.15566174e+01  3.29477609e+01
  -1.30789441e+00  4.51979504e+00  2.81629377e+00 -2.41579737e+01
   1.67538447e+01  3.79618809e+00  3.52619189e+00 -2.12695123e+00
  -5.84342888e+00  9.98335383e-01 -2.46886069e+01 -1.86974098e+00
  -1.69514911e+01 -8.88329312e+00 -9.81804470e+00 -1.67575792e+01
   1.68126120e+01  9.22685255e+00 -1.33518469e+01 -3.14702702e+01
   4.01475877e+00 -1.74841216e+00  3.80391728e+01  6.20107605e+01
  -7.31699943e+00  2.51690505e+00 -1.66437233e+01 -2.37916903e+01
   7.09227242e+00  2.62374791e+01  1.45234694e+01 -2.71682454e+01
  -3.27540550e-01 -8.82356718e+00 -3.30318250e+00  1.36719906e+00
   8.99428786e+00  1.72697724e+00  1.35257254e+01 -3.34575972e+01
   6.67237175e+00  1.31369359e+01  1.04677501e+01  4.64610831e+00
   1.43879724e+00 -3.13585177e+00 -9.26959553e+00 -1.22254282e+01]
```
6. 2~5を繰り返し、最終的にはg_best_vectorの評価マップとそのときのスコアを出力する。(基本使うのは価値マップだけ)
```python
    n_indivisuals = 150  #評価マップ(vector)の数 (= n_swarm)
    n_iters = 10　       #評価マップ(vector)の更新回数 (= n_iter)

    def Run(self):
        for i in range(self.n_iter):
            #スコア計算
            self.CalcScores()
            #評価マップ(vector)位置更新
            self.UpdateVectors()
        #ベストスコアを出力する。
        return self.g_best_score,self.g_best_vector
```



# InstaBae Algorithm
1. 初期化
```python
  def InitFlies(self):
          #点(vector:評価マップ)を初期化
          self.vectors = np.array([InitIndivisual() for _ in range(self.n_flies)])

          #方針決定確率を初期化
          self.strategies = np.zeros([self.n_flies, 3])
          for i in range(self.n_flies):
              randoms = np.random.uniform(1, 100, 3)
              for j in range(3):
                  self.strategies[i][j] = randoms[j]/sum(randoms)

          self.likes = np.zeros(self.n_flies)
          self.best_fly_indices = np.zeros(self.n_clusters)
```
- 方針決定確率(pioneer・faddist・masterのどれにするのか決める)
```python
  #合計すると１になる
  self.strategies[i] = [ 0.52438472 0.36452948 0.1110858 ]
```
2. 敵(enemy)として評価マップ(vector)をランダムに一つ選び、全ての評価マップ(vector)と順番にオセロで対戦させる。
```python
    def EvaluateLikes(self):
        #対戦する相手をランダムに決める
        enemy = np.random.randint(0,self.n_flies,1)

        #決まった点(vector)と全部の点(vector)をそれぞれ対戦させる
        for i in range(self.n_flies):
            self.likes[i] = EvalIndivisual(self.vectors[i],self.vectors[enemy])
        
```
3. 評価マップ(vector)をクラスタリング(グループ分け)して、クラスタの中心を求める
4. 前回計算したクラスタの中心との距離を速度として求める
5. 各クラスター内のベストスコアを記録する
6. 各クラスター内の平均スコアを求める
7. 各クラスターの中心同士の平均距離を求める
```python
    def Clustering(self):
        #中心の点(center)と速度設定
        if self.centers is None:
            #クラスタリング(グループ分け)
            model = KMeans(n_clusters=self.n_clusters)

            #平均を求める
            result = model.fit(self.vectors)

            #重心を求める
            #result.cluster_centers_ :クラスタ中心の座標
            self.centers = result.cluster_centers_
        else:
            model = KMeans(n_init=1,n_clusters=self.n_clusters,init=self.centers)
            result = model.fit(self.vectors)

        #評価マップ(vector)がどのクラスターに分類されるか
        self.labels = result.labels_
        self.center_speeds = result.cluster_centers_ - self.centers
        self.centers = result.cluster_centers_

        # best flies in each cluster
        best = np.zeros(self.n_clusters)
        self.cluster_like_average = np.zeros(self.n_clusters)
        for i in range(self.n_flies):
            label = self.labels[i]
            if (self.likes[i] > best[label]):
                best[label] = self.likes[i]
                self.best_fly_indices[label] = i

        # like average in each cluster
        for i in range(self.n_clusters):
            self.cluster_like_average[i] = np.mean(
                    [self.likes[j] for j in range(self.n_flies) if self.labels[j] == i])

        # average dist between each cluster
        self.center_dist_average = np.zeros_like(self.vectors[0])
        for i in range(self.n_clusters):
            for j in range(i+1,self.n_clusters):
                self.center_dist_average += (self.centers[i]-self.centers[j])
        self.center_dist_average /= sum(range(1,self.n_clusters))
```
### クラスタリング
- 特定のルールに基づいていくつかのグループに分類
- 教師なしの機械学習
```python
  #n_init    :クラスタリングの回数 
  #n_clusters:クラスター(グループ)の数
  #init      :前回の結果(クラスタの中心)を元にクラスタリングする

  model = KMeans(n_init=1,n_clusters=self.n_clusters,init=self.centers)

  #n_init = 1 のとき、
  #result = model.fit(self.vectors)
  #で平均を求めても値は変わらない!
```
### k-means法（k-means method）
- 非階層的クラスタリング ・・・ 母集団の中で近いデータをまとめる
- 入力データ
<img src="https://github.com/ntaku256/AI/blob/main/Source/kmean1.png" width="60%">

1. まず、k個のクラスタの代表点（中心）を決める
<img src="https://github.com/ntaku256/AI/blob/main/Source/kmean2.png" width="60%">

2. 次に、全てのデータを、もっとも中心が近いクラスタに振り分ける
<img src="https://github.com/ntaku256/AI/blob/main/Source/kmean3.png" width="60%">

3. クラスタがまとまったら、再度、各クラスタの重心点（平均値）を計算しし直す。重心点を変えることでデータが分類されるクラスタも変わる。
<img src="https://github.com/ntaku256/AI/blob/main/Source/kmean4.png" width="60%">

4. 全てのデータのクラスタに変更がなくなるまで3の作業を繰り返し、グループ分けを行います。
### k-meansの問題点
- 初回にランダムに生成されたセントロイドの場所によって
  - 極度に偏ったクラスタリングができる
  - なかなか収束しない事態に陥ってしまう
### k-means++法
- 従来のk-meansの初回セントロイド選定方法に改良を加えた手法
1. まず始めにデータ点をランダムに選び1つ目のクラスタ中心とする
2. 全てのデータ点とその最近傍のクラスタ中心の距離を求める
3. その距離の二乗に比例した確率でクラスタ中心として選ばれていないデータ点をクラスタ中心としてランダムに選んでいく
- 引用文献

https://exawizards.com/column/article/ai/clustering/

https://pythonbunseki.com/python-k-means/

- 外部モジュール,その他
```
  #平均を求める
  np.mean()

  #配列内の最大値がある場所を返す
  np.argmax()

  #a = a/b
  a /= b
```

8. 評価マップの更新方法を方針決定確率で決めて更新する
```python
    def UpdateFlieVector(self):
        for i in range(self.n_flies):
            action = roulett(self.strategies[i])
            # pioneer
            if action == 0:
                self.vectors[i] = self.UpdatePioneer(self.vectors[i])
            # faddist
            if action == 1:
                 self.vectors[i] = self.UpdateFaddist(self.vectors[i])
            # master
            if action == 2:
                self.vectors[i] = self.UpdateMaster(self.vectors[i], self.labels[i])
            #評価マップの値の上限を超えないように調整する
            filtIndivisual(self.vectors[i])
```
### Pioneer
- 開拓者の意味。高専系Youtuberなど、発想力で訳わからんジャンルを開拓し、一部のファンからいいねをもらう人。自分の作品からパレート分布に従う乱数によってランダムに次回作を作成。パレート分布にした理由は遠くに探索を行ってほしかったため。
- 現在のクラスタを考慮しながら、新しい特徴を持つクラスタを発見しようとする(実際にクラスタ数は増えない)
```python
    def Pareto(mode,a,shape): 
        return (np.random.pareto(a,size=shape)+1)*mode

    def UpdatePioneer(self, vector):
        #クラスタ間の平均距離
        length = Pareto(1,6,self.center_dist_average.shape)
        #向き(純方向or逆方向)をランダムに決める
        rand01 = np.random.choice([-1,1],length.shape)
        return vector + self.center_dist_average*length*rand01
```
- パレート分布
  - パレート分布のスケールは各クラスタ間の平均距離になっています。画像はクラスタ間ユーグリット距離の乱数を生成していますが、実際は最適化する変数ごとに乱数を生成しています。
<img src="https://github.com/ntaku256/AI/blob/main/Source/Pioneer.png" width="80%">

```python
  #パレート分布
  a, mode = 6, 2  # 分布の幅と、モードを指定   
  (np.random.pareto(a, 1000) + 1) * mode

#パレート分布 (Pareto distribution) は19世紀のイタリア経済学者 Vilfredo Pareto によって考案された確率分布である．
#ベキ分布 (power law     probability distribution) のひとつに分類される．元々は高額所得者の所得分布を示す分布として提案された．
#実際の当てはまりも良く，富の8割は人口の2割によって支配されるという80:20の法則，またはパレートの法則として知られる法則を良く表現している．
#現在においては経済以外にも自然現象や社会現象等の様々な事例に当てはめられることが分かっており，
#人口集積のモデル化，ネットワーク間でやり取りされるデータ容量の分布，生物属あたりの生物種のバリエーションのモデル化等，
#経済学のみならず社会学から生物学まで広範な研究領域に渡って広く用いられている確率分布である．
#パラメーターはα (>0) およびβ (>0) であり，パレート分布は Par(α, β) にて略記される．確率密度関数は以下で与えられる．
```
- 引用文献

https://data-science.gr.jp/theory/tpd_pareto_distribution.html

https://www.ntrand.com/jp/pareto-distribution/

### Master
- 自分が得意な一つのジャンルでいいね数を稼ぐ。自分の作品のジャンルで一番を目指すことを目標として、作品を作成。
- クラスタ内でいいねを稼いでいるYoutuber(いいね数のルーレット選択)、クラスタ重心、クラスタの移動速度を参考に新作動画を作成。
<img src="https://github.com/ntaku256/AI/blob/main/Source/Master.png" width="80%">

- 自信の評価マップがあるクラスタ内で、ランダムに選択した評価マップを参考に位置を更新する
```python
    def UpdateMaster(self, vector, label):
        #参考にするクラスタ内の評価マップを探す
        index_table = [i for i in range(self.n_flies) if(self.labels[i]==label)]
        #各評価マップのスコアを取り出す
        table = [self.likes[i] for i in index_table]
        #ランダムに選択した評価マップを目標にする
        target_fly_index = index_table[roulett(table)]
        center = self.centers[label]
        #クラスタの中心との距離に重み(0~1)を付ける
        center_vector = (center - vector)*np.random.uniform(0,1)
        #目標の評価マップとの距離に重みを付ける
        target_vector = (self.vectors[target_fly_index] - vector)*np.random.uniform(0,1)
        #現在のクラスタの中心と前回のクラスタの中心との距離に重みを付ける
        center_speed_vector = self.center_speeds[label]*np.random.uniform(0,1)
        return vector+center_vector+target_vector+center_speed_vector

```
### Faddist
- 流行を追う人という意味。各世代で熱いジャンルを盛り上げる役割。メントスコーラ+自分の特徴みたいな感じで流行りを自分の作品に組み込む。流行りのクラスタ内でいいねを稼いでいるYoutuber、クラスタ重心、クラスタの移動速度を参考に新作動画を作成。
<img src="https://github.com/ntaku256/AI/blob/main/Source/Faddist.png" width="80%">

- ランダムにクラスタを選び、そのクラスタ内でランダムに選択した評価マップを参考に位置を更新する
```python
    def UpdateFaddist(self, vector):
        cluster = roulett(self.cluster_like_average)
        return self.UpdateMaster(vector,cluster)
```
9. 2~8を繰り返し、最終的には評価マップ(self.vectors[best_arg])とそのときのスコア(self.likes[best_arg])を出力する。(基本使うのは価値マップだけ)
```python
    n_indivisuals = 150  #評価マップの数 (= n_files)
    n_iters = 1000        #評価マップの更新回数 
    n_clusters = 10     #クラスターの数

    def Run(self):
        for i in range(self.n_iters):
            self.EvaluateLikes()
            self.Clustering()
            self.UpdateFlieVector()
        self.EvaluateLikes()
        best_arg = np.argmax(self.likes)
        return self.likes[best_arg],self.vectors[best_arg]
```
