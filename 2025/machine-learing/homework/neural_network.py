from typing import List, Sequence
import numpy as np


class MyMLP:
    """
    多層パーセプトロン（バックプロパゲーション学習）
    """

    def __init__(
        self,
        rho: float = 0.1,
        loss_threshold: float = 1e-4,
        hidden_unit_size: int | Sequence[int] = 10,
        random_seed: int | None = None,
    ) -> None:
        self.rho = float(rho)
        self.loss_threshold = float(loss_threshold)

        # 隠れ層定義を「intのリスト」に正規化
        if isinstance(hidden_unit_size, (list, tuple)):
            self.hidden_sizes: List[int] = list(map(int, hidden_unit_size))
        else:
            self.hidden_sizes = [int(hidden_unit_size)]

        self.random_seed = random_seed

        # fit()で初期化される
        self._W_hidden: list[np.ndarray] = []
        self._W_out: np.ndarray | None = None
        self._n_classes: int | None = None

        self._rng = np.random.default_rng(random_seed)

    @classmethod
    def _sigmoid(cls, z: np.ndarray) -> np.ndarray:
        """要素ごとのロジスティックシグモイド"""
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def g(cls, x: np.ndarray, W: np.ndarray) -> np.ndarray:  
        """
        1 層分の前向き伝播

        パラメータ
        ----------
        x : ndarray, 形状 (n_features,)
            バイアス項を含まない層入力ベクトル
        W : ndarray, 形状 (n_units, n_features+1)
            重み行列（最後の列がバイアス）

        戻り値
        -------
        ndarray, 形状 (n_units,)
            活性化関数を通した層の出力
        """
        x_aug = np.append(x, 1.0)  # バイアス1を追加
        z = W @ x_aug
        return cls._sigmoid(z)

    @classmethod
    def compute_epsilon(
        cls, err: np.ndarray, g: np.ndarray 
    ) -> np.ndarray:
        return err * g * (1.0 - g)

    def update_W(
        self, W: np.ndarray, g_in: np.ndarray, e: np.ndarray  
    ) -> np.ndarray:
        """
        パラメータ
        ----------
        W : ndarray, 形状 (n_units, n_inputs+1)
            現層の重み行列
        g_in : ndarray, 形状 (n_inputs,)
            現層への入力（バイアス項を除く）
        e : ndarray, 形状 (n_units,)
            現層のデルタ

        戻り値
        -------
        ndarray
            更新後の重み行列
        """
        g_aug = np.append(g_in, 1.0)  # バイアス項
        return W + self.rho * np.outer(e, g_aug)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        確率的バックプロパゲーションで学習
      
        X_train : ndarray, 形状 (n_samples, n_features)
            訓練特徴量
        y_train : ndarray, 形状 (n_samples,)
            クラスラベル（0 … C-1）
        """
        if X_train.ndim != 2:
            raise ValueError("X_trainは2次元配列でなければなりません。")
        if y_train.ndim != 1:
            raise ValueError("y_trainは1次元配列でなければなりません。")
        if X_train.shape[0] != y_train.size:
            raise ValueError("X_trainとy_trainのサンプル数が一致しません。")

        n_samples, n_features = X_train.shape
        classes = np.unique(y_train)
        self._n_classes = classes.size

        # ------------------ ターゲットを one-hot 化 ------------------
        T = np.zeros((n_samples, self._n_classes), dtype=float)
        T[np.arange(n_samples), y_train.astype(int)] = 1.0

        # ------------------ 重みの初期化 ------------------
        def rand_matrix(rows: int, cols: int) -> np.ndarray:
            return self._rng.standard_normal((rows, cols))

        self._W_hidden = []
        prev_size = n_features
        for h in self.hidden_sizes:
            self._W_hidden.append(rand_matrix(h, prev_size + 1))
            prev_size = h
        self._W_out = rand_matrix(self._n_classes, prev_size + 1)

        # ------------------ メイン学習ループ ------------------
        epoch = 0
        while True:  # mse < threshold になるまでループ
            sse = 0.0  # 1 エポックの二乗誤差和
            for x, t in zip(X_train, T, strict=True):
                # ---------- 順伝播 ----------
                activations: list[np.ndarray] = [x]  # 入力層
                g_in = x
                for W in self._W_hidden:
                    g_out = self.g(g_in, W)
                    activations.append(g_out)
                    g_in = g_out
                assert self._W_out is not None  # for mypy
                y = self.g(g_in, self._W_out)
                activations.append(y)

                # ---------- 逆伝播 ----------
                err = t - y
                sse += np.dot(err, err) * 0.5

                eps_next = self.compute_epsilon(err, y)

                # 出力層の重み更新
                self._W_out = self.update_W(self._W_out, activations[-2], eps_next)

                # 隠れ層（後ろから前へ）
                for layer_idx in range(len(self._W_hidden) - 1, -1, -1):
                    W_next = (
                        self._W_out
                        if layer_idx == len(self._W_hidden) - 1
                        else self._W_hidden[layer_idx + 1]
                    )
                    # バイアス列を除外して誤差を伝播
                    err_hidden = W_next[:, :-1].T @ eps_next
                    eps_hidden = self.compute_epsilon(
                        err_hidden, activations[layer_idx + 1]
                    )
                    # 現層の重み更新
                    self._W_hidden[layer_idx] = self.update_W(
                        self._W_hidden[layer_idx], activations[layer_idx], eps_hidden
                    )
                    eps_next = eps_hidden  # さらに伝播

            mse = sse / n_samples
            epoch += 1
            if mse < self.loss_threshold:
                break
            # 収束しない場合の安全装置
            if epoch > 10_000:
                raise RuntimeError(
                    "10,000エポック以内に収束しませんでした。"
                    f" 最終MSE = {mse:.6g}"
                )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if X_test.ndim != 2:
            raise ValueError("X_testは2次元配列でなければなりません。")

        preds: list[int] = []
        for x in X_test:
            g_in = x
            for W in self._W_hidden:
                g_in = self.g(g_in, W)
            assert self._W_out is not None
            out = self.g(g_in, self._W_out)
            preds.append(int(np.argmax(out)))

        return np.asarray(preds, dtype=int)

# 動作確認テスト

# if __name__ == "__main__":
#     # test1
#     X_xor = np.array(
#         [
#             [0.0, 0.0],
#             [0.0, 1.0],
#             [1.0, 0.0],
#             [1.0, 1.0],
#         ],
#         dtype=float,
#     )
#     y_xor = np.array([0, 1, 1, 0], dtype=int)

#     mlp_xor = MyMLP(
#         rho=0.5,              # やや高めの学習率
#         hidden_unit_size=4,   # 1隠れ層 4ユニット
#         loss_threshold=1e-3,  # 収束判定を緩めて高速化
#         random_seed=42,
#     )
#     mlp_xor.fit(X_xor, y_xor)
#     preds_xor = mlp_xor.predict(X_xor)
#     print("XOR予測:", preds_xor.tolist())

#     assert np.array_equal(preds_xor, y_xor), "XORテスト失敗: 予測が正解と一致しません"

#     # test2
#     rng = np.random.default_rng(42)
#     n_per_class = 50
#     X_blobs = np.vstack(
#         [
#             rng.normal(loc=(0, 0), scale=0.5, size=(n_per_class, 2)),
#             rng.normal(loc=(3, 0), scale=0.5, size=(n_per_class, 2)),
#             rng.normal(loc=(0, 3), scale=0.5, size=(n_per_class, 2)),
#         ]
#     )
#     y_blobs = np.repeat([0, 1, 2], n_per_class)

#     mlp_blobs = MyMLP(
#         rho=0.1,
#         hidden_unit_size=[8, 8],   # 隠れ層2層 × 8ユニット
#         loss_threshold=1e-3,
#         random_seed=123,
#     )
#     mlp_blobs.fit(X_blobs, y_blobs)
#     preds_blobs = mlp_blobs.predict(X_blobs)
#     acc = (preds_blobs == y_blobs).mean()
#     print(f"Blobs訓練精度: {acc:.3f}")

#     assert acc > 0.90, "Blobsテスト失敗: 精度が 90% 未満です"

#     print("すべてのテストに合格しました。")
