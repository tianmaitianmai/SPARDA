#=
This file contains a set of useful functions that were not present in Julia.
Some of the ideas for these functions were inspired by Haskell,
    and the associative definitions are similar to Haskell's.
=#

fst(x) = x[1]

snd(x) = x[2]

ith(i) = x -> x[i]

unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

curry(f, xs...) = (ys...) -> f(xs..., ys...)

uncurry(f) = xs -> f(xs...)

remove_ith(i, xs) = filter(x -> first(x) != i, xs |> enumerate |> collect) |> curry(map, snd)

scanl(f, b, xs) =
    foldl(xs; init = [b]) do bs, x
        push!(bs, f(bs[end], x))
        bs
    end

scanl1(f, xs) = scanl(f, xs[1], xs[2:end])

scanr(f, b, xs) = foldr(xs; init = [b]) do x, bs
    push!(bs, f(bs[end], x))
    bs
end |> reverse

scanr1(f, xs) = scanr(f, xs[end], xs[1:(end - 1)])