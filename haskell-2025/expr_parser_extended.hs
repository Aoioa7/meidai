import Parsing

-- 左結合汎用パーサー
chainl1 :: Parser a -> Parser (a -> a -> a) -> Parser a
chainl1 p op = do x <- p
                  rest x
  where
    rest x = (do f <- op
                 y <- p
                 rest (f x y))
             <|> return x

-- factor ::= '(' expr ')' | integer
factor :: Parser Int
factor =  do symbol "("
             e <- expr
             symbol ")"
             return e
       <|> integer                  

-- term  ::= factor (('*'|'/') factor)*
term :: Parser Int
term = factor `chainl1` mulop

mulop :: Parser (Int -> Int -> Int)
mulop =  (symbol "*" >> return (*))
     <|> (symbol "/" >> return div)       -- 整数除算

-- expr  ::= term (('+'|'-') term)*
expr :: Parser Int
expr = term `chainl1` addop

addop :: Parser (Int -> Int -> Int)
addop =  (symbol "+" >> return (+))
     <|> (symbol "-" >> return (-))

-- トップレベルの評価関数
eval :: String -> Int
eval xs = case parse expr xs of
            [(n,[])]  -> n
            [(_,out)] -> error ("未使用 " ++ out)
            []        -> error "パース失敗"
