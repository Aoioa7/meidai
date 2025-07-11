module Main where

import System.Environment (getArgs)

-- 課題１の factL をここにも定義
factL :: Integer -> [Integer]
factL n = go 1 1
  where
    go i acc
      | i > n     = []
      | otherwise = let acc' = acc * i
                    in acc' : go (i + 1) acc'

main :: IO ()
main = do
  args <- getArgs
  case args of
    [nStr] -> case reads nStr of
      [(n,"")] -> putStrLn $ show (factL n)
      _        -> putStrLn "エラー: 引数が整数ではありません。"
    _      -> putStrLn "使用法: fact <自然数>"
