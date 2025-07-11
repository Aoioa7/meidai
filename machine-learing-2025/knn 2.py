from typing import Any, Dict, TypeAlias
import numpy as np

Vector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating]]
Matrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating]]


# 入力：d次元特徴ベクトルを表す1次元配列2つ．
# 出力：与えられた特徴ベクトル間のユークリッド距離．
# i.e., def dist(x_train, x_test) -> float:
# 距離の尺度としてp = 2のミンコフスキー距離，つまりユークリッド距離を用いる．
def dist(x_train: Vector, x_test: Vector) -> float:
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    if x_train.ndim != 1 or x_test.ndim != 1:
        raise ValueError("入力はどちらも1次元配列である必要があります。")
    if x_train.shape[0] != x_test.shape[0]:
        raise ValueError("2つのベクトルの次元数が一致していません。")
    return float(np.linalg.norm(x_train - x_test, ord=2))


# 入力：n個のd次元特徴ベクトルを表す2次元配列およびd次元の特徴ベクトルを表す1次元配列．
# 出力：第2引数の特徴ベクトルとの距離が最も小さい第1引数中の特徴ベクトルの配列上での位置．
# i.e., def get_nearest_pos(X_train, x_test) -> int:
# 検証データと同距離の特徴ベクトルが訓練データ中に複数存在する場合，配列上での位置が先頭に近い方の訓練データへ分類する．
def get_nearest_pos(X_train: Matrix, x_test: Vector) -> int:
    X_train = np.asarray(X_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    if X_train.ndim != 2 or x_test.ndim != 1:
        raise ValueError("X_trainは2次元配列、x_testは1次元配列である必要があります。")
    if X_train.shape[1] != x_test.shape[0]:
        raise ValueError("X_trainとx_testの特徴次元が一致していません。")
    diffs = X_train - x_test
    dists_sq = np.einsum("ij,ij->i", diffs, diffs)
    return int(np.argmin(dists_sq))


class MyKNN:
    def __init__(self) -> None:
        self.k_: int = 1  # k=1の場合，つまり最近傍法に基づき分類を行う．
        self.metric_: str = "euclidean"
        self._X_train: Matrix | None = None
        self._y_train: np.ndarray[Any] | None = None 

    # i.e., def fit(self, X_train, y_train) -> None:
    def fit(self, X_train: Matrix, y_train: np.ndarray[Any]) -> "MyKNN":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)
        if X_train.ndim != 2:
            raise ValueError("X_trainは2次元配列である必要があります。")
        if y_train.ndim != 1:
            raise ValueError("y_trainは1次元配列である必要があります。")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_trainとy_trainのサンプル数が一致していません。")
        self._X_train = X_train
        self._y_train = y_train
        return self

    # i.e., def predict(self, X_test) -> "np.ndarray[int]":
    def predict(self, X_test: Matrix | Vector) -> np.ndarray[int]:
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("分類器がまだ学習されていません。fit()を先に呼び出してください。")
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        if X_test.ndim != 2:
            raise ValueError("X_testは2次元配列、または(d,)形状の1次元配列である必要があります。")
        if X_test.shape[1] != self._X_train.shape[1]:
            raise ValueError("X_testの特徴次元が学習データと一致していません。")
        indices = [get_nearest_pos(self._X_train, x) for x in X_test]
        return self._y_train[indices]

    # 現在の内部パラメータを出力する関数get_parametersを持つ．
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k": self.k_,
            "metric": self.metric_,
            "n_train_samples": None if self._X_train is None else self._X_train.shape[0],
            "n_features": None if self._X_train is None else self._X_train.shape[1],
        }

# 以下、Google colab動作確認用(コメントアウト)
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier

# def _self_test() -> None:
#     iris = load_iris()
#     X_train, X_test, y_train, y_test = train_test_split(
#         iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
#     )
#     my_clf = MyKNN().fit(X_train, y_train)
#     # sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)と同じ答えを出す．
#     skl_clf = KNeighborsClassifier(n_neighbors=1, p=2).fit(X_train, y_train)
#     y_pred_my = my_clf.predict(X_test)
#     y_pred_skl = skl_clf.predict(X_test)
#     if np.array_equal(y_pred_my, y_pred_skl):
#         print("テスト合格: MyKNN の予測は scikit‑learn と一致しました。")
#     else:
#         diff = np.where(y_pred_my != y_pred_skl)[0]
#         print("テスト失敗: 予測が一致しないサンプルが", diff.size, "件あります。")
#         print("Indices:", diff.tolist())

# def main() -> None:
#     _self_test()

# if __name__ == "__main__":
#     main()
