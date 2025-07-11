import numpy as np

def to_one_hot(y: np.ndarray, class_num: int) -> np.ndarray:
    """
    ラベル配列yをone-hot行列に変換する。
    """
    y = np.asarray(y, dtype=int).ravel()

    if y.size == 0:
        raise ValueError("ラベル配列yが空です。")
    if y.min() < 0 or y.max() >= class_num:
        raise ValueError("ラベルは0以上class_num未満の整数である必要があります。")

    # (n_samples, class_num) すべて0で初期化
    one_hot = np.zeros((y.size, class_num), dtype=int)
    # 行インデックスと列インデックスを同時に指定して1を立てる
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def pseudo_inv(X: np.ndarray, e: float = 1e-8) -> np.ndarray:
    """
    リッジ項eを加えたムーア＝ペンローズ擬似逆行列 (XᵀX + eI)⁻¹ Xᵀ を返す。
    """
    X = np.asarray(X, dtype=float)
    XtX = X.T @ X
    XtX_reg = XtX + e * np.eye(XtX.shape[0], dtype=float)   # 対角にeを加えて可逆化
    XtX_inv = np.linalg.inv(XtX_reg)
    return XtX_inv @ X.T


class MyLinear:
    """
    one-vs-all で学習する多クラス線形分類器
    ------------------------------------------------------------
      g_i(x) = w_i^T [1, x] の最大値をとるクラスを予測
    """
    def __init__(self, class_num: int) -> None:
        self.class_num = int(class_num)
        self.W: np.ndarray | None = None   # shape = (class_num, d + 1)

    # ------------------------------------------------------------
    # クラスメソッド: 1 クラス分の重みを閉形式で計算
    # ------------------------------------------------------------
    @classmethod
    def compute_w(
        cls,
        X_train_ext: np.ndarray,
        b_i: np.ndarray,
        e: float = 1e-8,
    ) -> np.ndarray:
        """
        X_train_ext : shape (n_samples, d + 1)
            先頭列が1のバイアス付き設計行列
        b_i : shape (n_samples,)
            対象クラスなら1, それ以外0の教師信号
        """
        pinv = pseudo_inv(X_train_ext, e)
        return pinv @ b_i      # shape = (d + 1,)

    # ------------------------------------------------------------
    # 学習: 全クラスの重み行列 W を作成
    # ------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        e: float = 1e-8,
    ) -> None:
        n_samples = X_train.shape[0]

        # バイアス項 1 を列方向に追加
        X_ext = np.hstack([np.ones((n_samples, 1), dtype=float), X_train])
        # one-hot で各クラスのターゲット行列 B を作成
        B = to_one_hot(y_train, self.class_num)             # (n_samples, class_num)
        # クラスごとに重み w_i を計算して縦に積む
        W = np.empty((self.class_num, X_ext.shape[1]), dtype=float)
        for i in range(self.class_num):
            W[i] = self.compute_w(X_ext, B[:, i], e)

        self.W = W

    # ------------------------------------------------------------
    # 予測: 最大スコアのクラス ID を返す
    # ------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("fit() が実行されていないため予測できません。")

        m = X_test.shape[0]
        X_ext = np.hstack([np.ones((m, 1), dtype=float), X_test])  # (m, d + 1)
        # スコア行列 shape = (class_num, m)
        scores = self.W @ X_ext.T
        return scores.argmax(axis=0)
