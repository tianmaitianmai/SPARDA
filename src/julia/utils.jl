using FileIO
using ImageIO
using Images

"""
Short function(s)
"""

round4(x) = round(x, digits = 4)
dimension(x) = length(size(x))

"""
    mul(xs::AbstractVector{<:Number})

# Examples
    mul([1,2,3]) = 1*(2*3) = 6

 ```julia-repl
 julia> mul([1,2,3])
 6
 ```
"""
mul(xs::AbstractVector{<:Number}) = foldr(*, xs)

function make_extension(extension::AbstractString)
    e = ""
    if extension[1] == '.'
        e = extension
    else
        e = "." * extension
    end
    return e
end

"""
    have_extension(extension::AbstractString)

Return a function, which takes an `x::AbstractString` and return if `x` have the same extension with `extension`.

# Examples
```julia-repl
julia> have_extension(".bar")("foo.bar")
true
```
"""
function have_extension(extension::AbstractString)
    helper(x::AbstractString) = splitext(x)[end] == make_extension(extension)
    return helper
end

"""
    replace_extension(extension::AbstractString, fname::AbstractString)

Relpace `fname`'s extension with `extension`

# Examples
```julia-repl
julia> replace_extension(".bar", "a/b/c/foo.baz")
"a/b/c/foo.bar"
```
"""
function replace_extension(extension::AbstractString, fname::AbstractString)
    f_, _ = splitext(fname)
    return f_ * make_extension(extension)
end

"""
    mkpath_if_needed(s)

Make a path `s` only when path `s` does not exist.
"""
function mkpath_if_needed(s)
    !ispath(s) && mkpath(s)
    return Nothing
end