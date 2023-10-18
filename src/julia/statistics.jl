using Images
using FileIO
using ImageIO
using ProgressMeter: @showprogress
using DataFrames, CSV
using Dates
@enum ObjLabel Unknown=1 Prominence=2 ActiveRegion=3

include("fp_tools.jl")

struct Coord
    x::Float64
    y::Float64
end

struct Box
    left_down::Coord
    right_up::Coord
end

Base.@kwdef mutable struct LabelParams
    date = ""::String
    label = Unknown::ObjLabel
    origin_size = Coord(0.0, 0.0)::Coord
    num = 0::Int
    box = Box(Coord(0.0, 0.0), Coord(0.0, 0.0))::Box
    area = 0.0::Float64
    centroid = Coord(0.0, 0.0)::Coord
end

function stats_components(fig::Matrix, date::String, label::ObjLabel; area_adjust = true)
    img_h, img_w = size(fig)
    # rₛ = img_w / (2π)
    bias = 1 + 2π * img_h / img_w
    w = -2π / img_w
    to_θ(x) = 1.5π - (2π / img_w) * x
    # the det of Jacoobi == r
    to_r(x) = w * x + bias # respective to rₛ = img_w / (2π)
    function rθ(x, y)
        r = sqrt(x^2 + y^2)
        θ = atan(y, x)
        return (r, θ)
    end
    labels = label_components(fig)
    num = maximum(labels)

    Axs = fill(Inf, num)
    Ays = fill(Inf, num)
    Bxs = fill(-Inf, num)
    Bys = fill(-Inf, num)

    areas = zeros(Float64, num)
    cxs = zeros(Float64, num)
    cys = zeros(Float64, num)
    Rs = zeros(Float64, num)

    for index in CartesianIndices(labels)
        k = labels[index]
        if k != 0
            i, j = index[1], index[2]
            θ = to_θ(j)
            r = to_r(i)
            x = r * cos(θ)
            y = r * sin(θ)

            # the above θ is in polar coordinate
            # our coordinate is a litte different
            θ = 1.5π - θ # == 2π*j/img_w

            # area_k is introduced to adjust the area due to
            # the ring to rectangle process
            area_k = 1.0
            if area_adjust
                area_k = r
            end
            areas[k] += area_k
            # Axs, Ays, Bxs, Bys in this loop, store data in solar disk
            Axs[k] = min(Axs[k], r)
            Ays[k] = min(Ays[k], θ)
            Bxs[k] = max(Bxs[k], r)
            Bys[k] = max(Bys[k], θ)
            cxs[k] += x * area_k
            cys[k] += y * area_k
            Rs[k] += r * area_k
        end
    end
    cxs = cxs ./ areas
    cys = cys ./ areas
    Rs = Rs ./ areas

    # Be careful, I might switch the input order
    # due to the choice of coordinate
    cxs, cys = rθ.(cys, cxs) |> unzip
    cys = cys .+ π

    # it shoulde be equivalent to the above
    # cxs, cys = rθ.(cxs, cys) |> unzip
    # cys = @. mod(1.5π - cys, 2π)

    #cxs = round.(Int,cxs./areas)
    #cys = round.(Int,cys./areas)
    pars = Array{LabelParams, 1}()
    for k in 1:num
        p = LabelParams()
        p.label = label
        p.num = k
        p.box = Box(Coord(Axs[k], Ays[k]), Coord(Bxs[k], Bys[k]))
        p.area = areas[k]
        p.centroid = Coord(Rs[k], cys[k])
        p.date = date
        # p.compression_x_y = compression_x_y
        push!(pars, p)
    end
    #return (cxs, cys)
    return pars
end

function generate_dict(target::LabelParams)
    D = Dict{Any, Any}()
    D[:Date] = target.date
    D[:Label] = string(target.label)
    D[:Num] = target.num
    D[:Area] = target.area
    D[:r_1] = target.box.left_down.x
    D[:θ_1] = target.box.left_down.y
    D[:r_2] = target.box.right_up.x
    D[:θ_2] = target.box.right_up.y
    D[:r_c] = target.centroid.x
    D[:θ_c] = target.centroid.y
    return D
end

function df_initialize()
    df = DataFrame(
          Date = String[]
        , Label = String[]
        , Num = Int[]
        , Area = Float64[]
        , r_1 = Float64[]
        , θ_1 = Float64[]
        , r_2 = Float64[]
        , θ_2 = Float64[]
        , r_c = Float64[]
        , θ_c = Float64[]
    )
    return df
end

function image_onehot(img; dims = 1) # (3,H,W) by default
    img_ = zeros(size(img)...)
    indices = argmax(img, dims = dims)
    img_[indices] .= 1
    return img_
end

function name_to_date(s)
    #like `AIA20100513_000008_0304.png'
    yyyy, mm, dd = s[4:7], s[8:9], s[10:11]
    string(yyyy, "-", mm, "-", dd)
end

"""
apply post processing in this function
"""
function df_construct(ŷ_path)
    df = df_initialize()
    ŷ_name = readdir(ŷ_path)
    sort!(ŷ_name)
    @showprogress for file_name in ŷ_name
        ŷ = load(joinpath(ŷ_path, file_name))
        ŷ = channelview(ŷ)
        # ŷ = image_onehot(ŷ)
        r = ŷ[1, :, :]
        g = ŷ[2, :, :]
        date = name_to_date(file_name)
        P = stats_components(g, date, Prominence
                             #; area_adjust=true
                             )
        AR = stats_components(r, date, ActiveRegion
                              #; area_adjust=true
                              )

        foreach(x -> push!(df, generate_dict(x)), P)
        foreach(x -> push!(df, generate_dict(x)), AR)
    end
    return df
end

function full_df_construct(; read_from_file = false)
    if read_from_file
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../../data/stats.csv")))
    else
        y = [t for t in readdir.(joinpath(@__DIR__, "../../data/predict_post/"), join=true) if isdir(t)]
        df = vcat(df_construct.(y)...)
        w_p = joinpath(@__DIR__, "../../data/statistic.csv")
        @info "write to $w_p"
        CSV.write(w_p, df)
    end
    return df
end

function month_average(df::DataFrame, s::Date, e::Date)
    @assert day(s) == day(e) == 1
    split_days = Dates.value.(collect(s:Month(1):e))
    days = @. Dates.value(Date(df.Date, "yyyy_mm_dd"))
    df.days = days
    df = filter(:days => x -> split_days[1] <= x < split_days[end], df)
    j = 0
    while j < length(split_days)
        for i in eachindex(days)
            if split_days[j] <= days[i] <= split_days[j + 1]
                j = j + 1
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        @warn "Passed zero argument. One argument is supposed to be passed, e.g. `julia --project statistic.jl true`"
    elseif length(ARGS) == 1
        full_df_construct(; read_from_file = parse(Bool, ARGS[1]))
    else
        @warn "Passed too many arguments. One argument is supposed to be passed, e.g. `julia --project statistic.jl true`"
    end
end