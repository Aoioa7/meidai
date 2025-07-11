--- 課題1 (再帰呼び出し)

myanyRec :: (t -> Bool) -> [t] -> Bool　--- 型
myanyRec _ []     = False --- 空リストの場合はFalseを返す
myanyRec p (x:xs) = p x || myanyRec p xs 

--- 1.	先頭xを取り出してp xを計算
---　2.	その結果が True ➔ || の左だけで決着
--- 3.	Falseの場合 ➔ 右側myanyRec p xsを必要になったので評価
---　4.	以後リスト末尾まで同じ処理を再帰

suffixListRec :: [a] -> [[a]]
suffixListRec []         = [[]]
suffixListRec xs@(_:xs')  = xs : suffixListRec xs'

--- xs 先頭要素を含む元のリスト
--- xs' 先頭要素を除いた残り
--- @ 「分解」と「元データの保持」を同時に行う

--  課題2 (foldr)

myany :: (t -> Bool) -> [t] -> Bool
myany p = foldr (\x acc -> p x || acc) False

--- (\x acc -> p x || acc) が myanyRecの再帰本体と同じこと
--- ORを畳み込んでいくので結果は1個のBool

deleteall :: Eq a => a -> [a] -> [a]
deleteall y = foldr (\x acc -> if x == y then acc else x : acc) []

--- x==yならaccに入れない

unique :: Eq a => [a] -> [a]
unique xs =
  let step acc x = if x `elem` acc then acc else x : acc
  in  reverse (foldr step [] xs)

suffixList :: [a] -> [[a]]
suffixList = foldr (\x acc@(y:_) -> (x:y) : acc) [[]]

--  課題2 (foldl)

myanyL :: (t -> Bool) -> [t] -> Bool
myanyL p = foldl (\acc x -> acc || p x) False
