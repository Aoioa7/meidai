from typing import Tuple
import numpy as np

# ----------------------------------------------------------------------
# 例外クラス
# ----------------------------------------------------------------------
class FeatureSpaceError(Exception):
    """feature_spaceモジュールで発生する基底例外"""

class InvalidDimensionError(FeatureSpaceError, ValueError):
    """入力配列の次元が想定外のときに送出"""

class LabelMismatchError(FeatureSpaceError, ValueError):
    """ラベル配列が不正またはサンプル数が合わないときに送出"""

class InvalidComponentsError(FeatureSpaceError, ValueError):
    """n_componentsが1未満のときに送出"""

class NotFittedError(FeatureSpaceError, RuntimeError):
    """fit()が呼ばれていないのにtransform()したときに送出"""


# ----------------------------------------------------------------------
# 基本関数
# ----------------------------------------------------------------------
def compute_S(X: np.ndarray, means: np.ndarray) -> np.ndarray[np.ndarray[float]]:
    """
    変動行列Sを計算する。
    """
    
    X_centered = X - means
    S = X_centered.T @ X_centered
    return S


def compute_w(S_w: np.ndarray, means_0: np.ndarray, means_1: np.ndarray) -> np.ndarray[float]:
    """
    Fisher判別の線形変換ベクトルwを計算する。
    """
    
    # pinvは特異行列でも安定して逆行列の代替を返す
    S_w_inv = np.linalg.pinv(S_w)
    w = S_w_inv @ (means_1 - means_0)
    return w


def compute_sorted_eigen(
    cov_matrix: np.ndarray,
) -> Tuple[np.ndarray[float], np.ndarray[np.ndarray[float]]]:
    """
    分散共分散行列の固有値・固有ベクトルを降順に並べて返す。
    """
    
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)  # eigh: 対称行列用
    sort_idx = np.argsort(eig_vals)[::-1]            # 降順
    eig_vals_sorted = eig_vals[sort_idx]
    eig_vecs_sorted = eig_vecs[:, sort_idx]
    return eig_vals_sorted, eig_vecs_sorted


# ----------------------------------------------------------------------
# Fisher判別
# ----------------------------------------------------------------------
class MyFisher:
    """2クラスFisherの線形判別器。"""

    def __init__(self) -> None:
        self.w_: np.ndarray | None = None
        self.threshold_: float | None = None
        self.means_: Tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        線形判別ベクトルwと判定しきい値を学習する。
        """
        
        if X_train.ndim != 2:
            raise InvalidDimensionError("X_trainは2次元配列である必要があります。")
        if y_train.ndim != 1 or len(y_train) != len(X_train):
            raise LabelMismatchError("y_trainはX_trainと同じサンプル数の1次元配列である必要があります。")

        # クラスごとに分割
        X0, X1 = X_train[y_train == 0], X_train[y_train == 1]
        μ0, μ1 = X0.mean(axis=0), X1.mean(axis=0)

        # クラス内散布行列
        S0 = compute_S(X0, μ0)
        S1 = compute_S(X1, μ1)
        S_w = S0 + S1

        # 判別ベクトルw
        w = compute_w(S_w, μ0, μ1)

        # しきい値: 射影後の2クラス平均の中点
        t0, t1 = w @ μ0, w @ μ1
        threshold = (t0 + t1) / 2.0

        # 保存
        self.w_ = w
        self.threshold_ = threshold
        self.means_ = (μ0, μ1)

    # ------------------
    # 変換（1 次元への射影）
    # ------------------
    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """
        学習済み w で X_test を 1 次元に射影する。
        """
        
        if self.w_ is None:
            raise NotFittedError("fit()を先に呼び出してください。")
        return X_test @ self.w_


# ----------------------------------------------------------------------
# PCA変換
# ----------------------------------------------------------------------
class MyKL:
    """
    分散最大基準（PCA）による次元削減クラス。
    """

    def __init__(self, n_components: int):
        if n_components < 1:
            raise InvalidComponentsError("n_componentsは1以上である必要があります。")
        self.n_components = n_components
        self.components_: np.ndarray | None = None  # (n_features, n_components)
        self.mean_: np.ndarray | None = None        # (n_features,)
        
    def fit(self, X_train: np.ndarray):
        """
        主成分を学習して線形射影行列を求める。
        """
        
        if X_train.ndim != 2:
            raise InvalidDimensionError("X_trainは2次元配列である必要があります。")

        # 平均でセンタリング
        self.mean_ = X_train.mean(axis=0)
        X_centered = X_train - self.mean_

        # 共分散行列
        cov = (X_centered.T @ X_centered) / (len(X_train) - 1)

        # 固有分解（降順）
        eig_vals, eig_vecs = compute_sorted_eigen(cov)

        # 上位n_componentsを保持
        self.components_ = eig_vecs[:, : self.n_components]

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """
        学習済み主成分で X_test を射影する。
        """
        
        if self.components_ is None or self.mean_ is None:
            raise NotFittedError("fit() を先に呼び出してください。")
        X_centered = X_test - self.mean_
        X_reduced = X_centered @ self.components_
        return X_reduced

# ----------------------------------------------------------------------
# 動作確認用(以下コメントアウト)
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     # -------------------------
#     # 2クラスのサンプルデータ
#     # -------------------------
#     rng = np.random.default_rng(0)
#     n_samples = 100
#     # クラス0 : 原点まわりのガウス
#     X0 = rng.normal(loc=[0.0, 0.0], scale=0.6, size=(n_samples, 2))
#     # クラス1 : (2, 2) まわりのガウス
#     X1 = rng.normal(loc=[2.0, 2.0], scale=0.6, size=(n_samples, 2))

#     X_train = np.vstack([X0, X1])
#     y_train = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

#     # -----------------------------------
#     # Fisher判別で1次元に射影 & 判定
#     # -----------------------------------
#     fisher = MyFisher()
#     fisher.fit(X_train, y_train)

#     X_proj = fisher.transform(X_train)
#     preds = (X_proj >= fisher.threshold_).astype(int)
#     acc = (preds == y_train).mean()

#     print("=== Fisher判別 ===")
#     print(f"w             : {fisher.w_}")
#     print(f"threshold     : {fisher.threshold_:.3f}")
#     print(f"train accuracy: {acc * 100:.1f}%\n")

#     # --------------------------------
#     # PCAで2→1次元に圧縮してみる
#     # --------------------------------
#     pca = MyKL(n_components=1)
#     pca.fit(X_train)
#     X_reduced = pca.transform(X_train)

#     print("=== PCA (n_components=1) ===")
#     print(f"principal vector: {pca.components_.ravel()}")
#     print(f"first 5 projected values:\n{X_reduced[:5].ravel()}")
