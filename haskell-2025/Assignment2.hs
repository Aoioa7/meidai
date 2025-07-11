--  課題 2 : treeFold とそれを用いた走査関数
module Assignment2 where

data Tree a = Node (Tree a) a (Tree a)
            | Empty
            deriving (Eq, Ord, Show, Read)

treeFold :: (b -> a -> b -> b) -> b -> Tree a -> b
treeFold _ z Empty        = z
treeFold f z (Node l x r) = f (treeFold f z l) x (treeFold f z r)

nodesF   :: Num n => Tree t -> n
nodesF   = treeFold (\l _ r -> 1 + l + r) 0

heightF  :: (Num n, Ord n) => Tree t -> n
heightF  = treeFold (\l _ r -> 1 + max l r) 0

sumTreeF :: Num a => Tree a -> a
sumTreeF = treeFold (\l x r -> l + x + r) 0
