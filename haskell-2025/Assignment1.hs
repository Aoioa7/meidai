--  課題 1 : Tree の基本走査関数（直接再帰版）
module Assignment1 where

--  分岐を持つ　＝　data　／　ただの別名で実行時コスト0 ＝ newtype
-- 	Node (Tree a) a (Tree a): 左・値・右という見慣れた順で書いているだけ。変更しても型的には問題ないが、後続も書き換える必要あり
data Tree a = Node (Tree a) a (Tree a)
            | Empty
            deriving (Eq, Ord, Show, Read)

-- 	Int vs Integer: 固定長で速いか、多倍長で無限か。Numは数字をまとめる型クラス
nodes :: Num n => Tree t -> n
nodes Empty          = 0
nodes (Node l _ r)   = 1 + nodes l + nodes r

height :: (Num n, Ord n) => Tree t -> n
height Empty         = 0
height (Node l _ r)  = 1 + max (height l) (height r)

sumTree :: Num a => Tree a -> a
sumTree Empty        = 0
sumTree (Node l x r) = sumTree l + x + sumTree r

mapTree :: (a -> b) -> Tree a -> Tree b
mapTree _ Empty        = Empty
mapTree f (Node l x r) = Node (mapTree f l) (f x) (mapTree f r)
