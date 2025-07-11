module Mid
  ( mid          
  , midCond     
  , midGuard     
  , midSort      
  ) where

import Data.List (sort)

midCond :: Ord a => a -> a -> a -> a
midCond a b c =
  if (a <= b && b <= c) || (c <= b && b <= a) then b    
  else if (b <= a && a <= c) || (c <= a && a <= b) then a
  else c                                                  

midGuard :: Ord a => a -> a -> a -> a
midGuard a b c
  | (a <= b && b <= c) || (c <= b && b <= a) = b
  | (b <= a && a <= c) || (c <= a && a <= b) = a
  | otherwise                                = c

midSort :: Ord a => a -> a -> a -> a
midSort a b c = sort [a, b, c] !! 1   

mid :: Ord a => a -> a -> a -> a
mid = midGuard   -- ここを書き換えれば実装を変えられる
