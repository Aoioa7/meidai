import numpy as np
from numpy.typing import NDArray


class BayesError(Exception):
    """ベイズモジュール共通の基底例外。"""


class InputShapeError(BayesError):
    """配列の形が想定と異なる場合の例外。"""


class InvalidLabelError(BayesError):
    """ラベルが0/1以外を含む場合の例外。"""


class UnfittedModelError(BayesError):
    """fit前にpredictなどを呼び出した場合の例外。"""

def compute_log_prior_p(y: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    クラス0,1の事前確率を推定し、対数を返す。
    """
    y = np.asarray(y, dtype=np.int64).ravel()
    if y.ndim != 1:
        raise InputShapeError("yは1次元配列である必要があります。")
    if not np.isin(y, [0, 1]).all():
        raise InvalidLabelError("yには0と1以外を含められません。")

    cnt = np.bincount(y, minlength=2)    
    # 「どの語も必ず1回は出た」という仮の回数を加えることで確率0を避け計算を安定させる
    prior = (cnt + 1) / (cnt.sum() + 2)
    return np.log(prior.astype(np.float64))


def compute_log_post_P(
    X: NDArray[np.int64] | NDArray[np.float64],
    y: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    クラス条件付き確率P(word | class)を推定し、対数を返す。
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64).ravel()
    if X.shape[0] != y.shape[0]:
        raise InputShapeError("Xとyのサンプル数が一致しません。")
    if not np.isin(y, [0, 1]).all():
        raise InvalidLabelError("yには0と1以外を含められません。")

    n_feat = X.shape[1]
    log_post_P = np.empty((2, n_feat), dtype=np.float64)

    for c in (0, 1):
        X_c = X[y == c]
        word_cnt = X_c.sum(axis=0)                 # 各特徴の出現総数
        total = word_cnt.sum()
        prob = (word_cnt + 1) / (total + n_feat)   # 「どの語も必ず1回は出た」という仮の回数を加えることで確率0を避け計算を安定させる
        log_post_P[c] = np.log(prob.astype(np.float64))

    return log_post_P


def compute_likelihoods(
    X: NDArray[np.int64] | NDArray[np.float64],
    log_prior_p: NDArray[np.float64],
    log_post_P: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    各サンプル×各クラスの対数尤度を計算する。
    """
    X = np.asarray(X)
    if X.shape[1] != log_post_P.shape[1]:
        raise InputShapeError("Xとlog_post_Pの特徴数が一致しません。")

    ll0 = log_prior_p[0] + X @ log_post_P[0]
    ll1 = log_prior_p[1] + X @ log_post_P[1]
    return np.vstack((ll0, ll1)).T        

class MyBayes:
    """
    二値分類用の単純ベイズ分類器。
    fit → predict の流れで使用する。
    """

    def __init__(self) -> None:
        self._log_prior_p: NDArray[np.float64] | None = None
        self._log_post_P: NDArray[np.float64] | None = None

    def fit(
        self,
        X_train: NDArray[np.int64] | NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> None:
        """学習データから事前・事後確率を推定して保持する。"""
        self._log_prior_p = compute_log_prior_p(y_train)
        self._log_post_P = compute_log_post_P(X_train, y_train)

    def predict(
        self,
        X_test: NDArray[np.int64] | NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """各サンプルをMAPルールで0/1に分類して返す。"""
        if self._log_prior_p is None or self._log_post_P is None:
            raise UnfittedModelError("predictの前にfitを呼んでください。")

        ll = compute_likelihoods(X_test, self._log_prior_p, self._log_post_P)
        return ll.argmax(axis=1).astype(np.int64)


# 動作確認
# if __name__ == "__main__":
#     # ---------------- 学習用ダミーデータ ----------------
#     X_train = np.array([[2, 0, 1],
#                         [1, 1, 0],
#                         [0, 2, 3],
#                         [0, 1, 1]])
#     y_train = np.array([0, 0, 1, 1])

#     # ---------------- テスト用ダミーデータ ---------------
#     X_test = np.array([[1, 0, 0],   # 期待クラス 0
#                       [0, 2, 2]])  # 期待クラス 1
#     expected = np.array([0, 1])
    
#     model = MyBayes()
#     model.fit(X_train, y_train)

#     pred = model.predict(X_test)

#     # ---------------- 結果表示 --------------------------
#     print("予測結果 :", pred)
#     print("期待値   :", expected)
#     print("テスト   :", "OK" if np.array_equal(pred, expected) else "NG")

#     # 形状確認
#     log_prior = compute_log_prior_p(y_train)
#     log_post = compute_log_post_P(X_train, y_train)
#     ll = compute_likelihoods(X_train, log_prior, log_post)
#     print("\n--- 関数レベル確認 ---")
#     print("事前確率ベクトル shape :", log_prior.shape)
#     print("事後確率行列   shape :", log_post.shape)
#     print("尤度行列       shape :", ll.shape)
