using FITSIO
using Images
using AstroImages
import LocalFilters: bilateralfilter
using NPZ

include("utils.jl")

"""
    cut_to_ring(ff)

Receive an ImageHDU and cut out a ring with
height of `πR_{sun}/12` and turn it to a rectangle, starting at `-90ᵒ`, clockwise.

    Note:

Be careful with the orientation of image!
If not sure, check it
with `DS9` or package: `AstroImages`
"""
function cut_to_ring(ff::ImageHDU)
    ffheader = read_header(ff)
    input = read(ff)
    r = ffheader["R_SUN"]
    x = ffheader["CRPIX1"]
    y = ffheader["CRPIX2"]
    xlim = ffheader["NAXIS1"]
    ylim = ffheader["NAXIS2"]
    w = floor(Int, 2π * r)
    h = floor(Int, π * r / 12)
    output = reshape([input[min(max(floor(Int, x - (r + j) * sin(2π * i / w)), 1), xlim),
                            min(max(floor(Int, y - (r + j) * cos(2π * i / w)), 1), ylim)]
                      for i in 1:w for j in h:(-1):1],
                     (h, w))
    return output
end

"""
    linear_rescale(xs::AbstractArray, l, u)
Just linear rescale to `[l, u]`
"""
function linear_rescale(xs::AbstractArray, l, u)
    l, u = min(l, u), max(l, u)
    l0, u0 = minimum(xs), maximum(xs)
    if l0 == u0
        return fill(l, size(xs)...)
    end
    return [(u - l) * (x - l0) / (u0 - l0) + l for x in xs]
end

"""
    gray_rescale(xs::AbstractArray; L = 8)

Just gray rescale with up to `2^L` gray scales.
"""
gray_rescale(xs::AbstractArray; L = 8) = round.(Int, linear_rescale(xs, 0, 2^L - 1))

"""
    hist_stretch(xs::AbstractArray; L = 8)

Histogram stretching for up to `2^L` gray scales.
"""
function hist_stretch(xs::AbstractArray; L = 8)
    function helper(acc, i)
        acc[i + 1] += 1
        return acc
    end
    pdf = reduce(helper, vec(xs); init = zeros(Int, 2^L))
    cpd = cumsum(pdf) ./ (*(size(xs)...))
    return gray_rescale([cpd[x + 1] for x in xs]; L)
end

"""
    minus_minimum(xs::AbstractArray)

Return an array that substract `xs` with the minimum of `xs`.
"""
minus_minimum(xs::AbstractArray) = xs .- minimum(xs)

"""
    reinhard(xs::AbstractArray):
A tone mapping mehtod maps: [0,∞) -> [0,1)
"""
reinhard(xs::AbstractArray) = xs ./ (one(eltype(xs)) .+ xs)

"""
    extended_reinhard(xs::AbstractArray)

A tone mapping mehtod maps: `[0,∞) -> [0,1]`
"""
function extended_reinhard(xs::AbstractArray)
    xs_max_square = maximum(xs)^2
    return @.((one(eltype(xs)) + xs / xs_max_square) *
              xs/(one(eltype(xs)) + xs))
end

"""
    wang_rescale_rect(img, δ = 1e-4)

Rescale a rectangle with method from Wang's paper, averaging along the x-axis of the rectangle.
"""
function wang_rescale_rect(img, δ = 1e-4)
    rows, cols = size(img)
    img_ = zeros(size(img)...)
    for i in 1:rows
        avg = sum(img[i, :]) / cols
        for j in 1:cols
            img_[i, j] = (img[i, j] + δ) / (avg + δ)
        end
    end
    return img_
end

"""
    centering_low_high(img, l = 0.05, h = 0.95)

Sort the intensity of pixels, set the lowest `(100*l)%` to the `(100*l)`th lowest value and similar for the highest ones.
"""
function centering_low_high(img, l = 0.05, h = 0.95)
    len = mul(size(img))
    sorted = sort(reshape(img, len))
    low = sorted[floor(Int, l * len) + 1]
    high = sorted[ceil(Int, h * len)]
    return map(img) do x
        if x < low
            x = low
        elseif x > high
            x = high
        else
            x
        end
    end
end

"""
    process_path_with_f(func, i, o; verbose = false, filtering = have_extension("fits"))

with function `func` , input path `i`, output path `o`,
process files in `i`, using `func`, and store images in folder `o`.
`func` turn an ImageHDU to an image. `filtering` filter the wanted files
"""
function process_path_with_f(func, i, o; verbose = false,
                             filtering = have_extension("fits"))
    fs = filter(filtering, readdir(i))
    map(fs) do x
        verbose && (@info x)
        f = FITS(joinpath(i, x))
        save(joinpath(o,
                      filter(x -> x != "", rsplit(i, '/'))[end],
                      replace_extension(".png", x)),
             func(f[1]))
    end
    return Nothing
end

function cut_fits_and_save(i, o; verbose = false)
    fs = filter(have_extension(".fits"), readdir(i))
    map(fs) do x
        verbose && (@info x)
        f = FITS(joinpath(i, x))
        ring = cut_to_ring(f[1])
        npzwrite(joinpath(o, replace_extension(".npy", x)), ring)
    end
end

function power_reinhard(img, n)
    return @. (img^n / (1 + img^n))
end

function durand(xs; σr = 3, σs = 3, γ = 2, δ = 1e-6)
    xs_ = xs .- minimum(xs) .+ δ
    xs_ = xs_ ./ maximum(xs_)
    xs_ = log.(xs_)
    B = bilateralfilter(xs_, σr, σs)
    D = xs_ .- B
    xs_ = exp.(γ .* (B .- maximum(B)) .+ D)
    return xs_
end

if abspath(PROGRAM_FILE) == @__FILE__
    # channel_1
    my_process_1(x) = (x |> cut_to_ring |> minus_minimum |> wang_rescale_rect |> x->power_reinhard(x, 5) |> gray_rescale) ./ 255

    # channel_2
    durand_(x) = durand(x; σr=3, σs=3, γ=10, δ=1e-6)
    my_process_2(x) = (x |> cut_to_ring |> minus_minimum |> wang_rescale_rect |> x->power_reinhard(x, 10) |> durand_ |> gray_rescale) ./ 255

    # channel_3
    function my_process_3(x)
        (x |> cut_to_ring |> minus_minimum |> wang_rescale_rect |>
         t -> log.(1.0 .+ t) |> gray_rescale) ./ 255
    end

    for (my_process, folder) in zip([my_process_1, my_process_2, my_process_3], joinpath.(@__DIR__, ["../../data/channel_1", "../../data/channel_2", "../../data/channel_3"]))
        @time process_path_with_f.(x -> Gray.(my_process(x)),
                                   readdir(joinpath(@__DIR__, "../../data/FITS"), join=true),
                                   folder; verbose = true)
    end
end
