from typing import Optional
import numpy as np


class MyPerceptron:
    # ──────────────────── メンバ変数定義 ────────────────────
    rho: float                    # 学習係数
    W: Optional[np.ndarray]       # 拡張重み行列 
    max_epoch: int                # エポック最大数

    # ──────────────────── コンストラクタ ────────────────────
    def __init__(
        self,
        rho: float = 1.0,
        W: Optional[np.ndarray] = None,
        max_epoch: int = 1000,
    ) -> None:
        self.rho = float(rho)
        self.W = None if W is None else np.asarray(W, dtype=float)
        self.max_epoch = int(max_epoch)

    # ──────────────────── 識別関数 g ────────────────────
    def g(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("モデルが初期化されていません。先にfit()を実行してください。")
        return self.W @ x  

    # ──────────────────── 最大値選択機 ────────────────────
    @classmethod
    def select_max_class(cls, g_out: np.ndarray) -> int:
        return int(np.argmax(g_out))

    # ──────────────────── 重み更新 ────────────────────
    def update_W(self, x: np.ndarray, predicted_class: int, correct_class: int) -> None:
        if predicted_class == correct_class:
            return  # 誤分類でなければ更新不要
        self.W[correct_class] += self.rho * x
        self.W[predicted_class] -= self.rho * x

    # ──────────────────── 学習 ────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        
        if X_train.ndim != 2:
            raise ValueError("X_trainは2次元配列(n_samples, n_features)である必要があります。")
        if y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
            raise ValueError("y_trainはX_trainと同じサンプル数を持つ1次元配列である必要があります。")

        n_samples, n_features = X_train.shape
        # バイアス項を付加して拡張
        X_aug = np.hstack([np.ones((n_samples, 1)), X_train])  

        classes = np.unique(y_train)
        if classes.min() < 0:
            raise ValueError("クラスラベルは0以上の整数である必要があります。")
        n_classes = int(classes.max()) + 1  

        # W を初期化または形状チェック
        if self.W is None:
            self.W = np.zeros((n_classes, n_features + 1))
        else:
            if self.W.shape != (n_classes, n_features + 1):
                raise ValueError(
                    f"Wの形状は({n_classes},{n_features + 1})である必要があります(現在{self.W.shape})。"
                )

        # エポック学習ループ
        for epoch in range(1, self.max_epoch + 1):
            error_count = 0
            for x, t in zip(X_aug, y_train):
                g_out = self.g(x)
                y_pred = self.select_max_class(g_out)
                if y_pred != t:
                    self.update_W(x, y_pred, t)
                    error_count += 1

            # 全サンプル誤分類なし → 収束
            if error_count == 0:
                break

    # ──────────────────── 予測 ────────────────────
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("モデルが未学習です。先にfit()を実行してください。")

        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        m_samples = X_test.shape[0]
        X_aug = np.hstack([np.ones((m_samples, 1)), X_test])
        g_matrix = self.W @ X_aug.T  
        predictions = np.argmax(g_matrix, axis=0)  

        return predictions.astype(int)
