import Data.Char (isUpper, isDigit)

----------------------------------------------------------------
-- 課題1.  内包表記を用いた実装 (再帰禁止)
----------------------------------------------------------------

-- 文字列から大文字アルファベットと数字を抽出して略語を作る
abbreviate :: String -> String
abbreviate str = [c | c <- str, isUpper c || isDigit c]

-- 与えられたリストのすべての部分リスト(接尾)を返す
suffixList :: [a] -> [[a]]
suffixList xs = [drop n xs | n <- [0 .. length xs]]

----------------------------------------------------------------
-- 課題2.  再帰を用いた実装 (内包表記禁止)
----------------------------------------------------------------

-- elem と同等
myelem :: Eq a => a -> [a] -> Bool
myelem _ [] = False
myelem x (y:ys)
  | x == y    = True
  | otherwise = myelem x ys

-- リストから指定値をすべて削除
deleteall :: Eq a => a -> [a] -> [a]
deleteall _ [] = []
deleteall x (y:ys)
  | x == y    = deleteall x ys
  | otherwise = y : deleteall x ys

-- 重複要素を1個だけ残す（最後の出現を残す仕様）
unique :: Eq a => [a] -> [a]
unique [] = []
unique (x:xs)
  | myelem x xs = unique xs        
  | otherwise   = x : unique xs

-- 文字列のリストを区切り文字で連結
delimit :: [String] -> Char -> String
delimit [] _       = ""
delimit [s] _      = s
delimit (s:ss) sep = s ++ sep : delimit ss sep
