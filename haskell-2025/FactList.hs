module FactList where

factL :: Integer -> [Integer]
factL n = go 1 1
  where
    go i acc
      | i > n     = []
      | otherwise = let acc' = acc * i
                    in acc' : go (i + 1) acc'

factIL :: [Integer]
factIL = scanl1 (*) [1..]
