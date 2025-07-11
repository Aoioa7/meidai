-- プログラミングHaskell（初版・第2版）の数式パーサー
-- 使用例
--   eval "2*(3+4)"
--
-- 構文規則
--   expr   :== term ( '+ 'expr | e )
--   term   :== factor ( '*' term | e )
--   factor :== '(' expr ')' | nat
--   nat    :== '0' | '1' | ... | '9'

import Parsing

expr :: Parser Int
expr = do t <- term
          do symbol "+"
             e <- expr
             return (t + e)
           <|> return t

term :: Parser Int
term = do f <- factor
          do symbol "*"
             t <- term
             return (f * t)
           <|> return f
             
factor :: Parser Int
factor = do symbol "("
            e <- expr
            symbol ")"
            return e
          <|> natural

eval :: String -> Int
eval xs = case parse expr xs of
            [(n,[])] -> n
            [(_,out)] -> error ("unused input " ++ out)
            [] -> error "invalid input"
