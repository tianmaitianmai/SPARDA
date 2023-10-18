using Images, FileIO
using ProgressMeter: @showprogress

include("fp_tools.jl")

function recovery_hidden_AR!(ar_mask, p_mask)
    area(p, q) = (q[2] - p[2]) * (q[1] - p[1])
    rect(p, q) = [CartesianIndex(i, j) for i in p[1]:q[1] for j in p[2]:q[2]]
    type = typeof(first(ar_mask))
    function not_all_zeros(x, y)
        if x == 0 && y == 0
            return zero(type)
        end
        return one(type)
    end
    function helper(xs, ys, pq)
        mask = rect(pq...)
        xs[mask] .= not_all_zeros.(ys[mask], xs[mask])
        return xs
    end
    ar_components = label_components(ar_mask, trues(3, 3))
    mtree = MaxTree(ar_components, connectivity = 2)
    boxes = filter(x -> x[1] != x[2], boundingboxes(mtree)) |> unique
    max_idx = argmax([area(box...) for box in boxes])
    boxes = remove_ith(max_idx, boxes)
    for box in boxes
        ar_mask = helper(ar_mask, p_mask, box)
    end
    return nothing
end

function recovery_hidden_AR!(img)
    img_ch = channelview(img)
    r = @view img_ch[1, :, :]
    g = @view img_ch[2, :, :]
    recovery_hidden_AR!(r, g)
    return nothing
end

function recovery_hidden_AR(ar_mask, p_mask)
    ar_mask_ = copy(ar_mask)
    recovery_hidden_AR!(ar_mask_, p_mask)
    return ar_mask_
end

function recovery_hidden_AR(img)
    img_ = copy(img)
    recovery_hidden_AR!(img_)
    return img_
end

function filter_on_area!(mask, thr)
    components = label_components(mask, trues(3, 3))
    for i in 1:maximum(components)
        m_i = (components .== i)
        if sum(m_i) <= thr
            mask[m_i] .= 0
        end
    end
    return nothing
end

function filter_on_area(mask, thr)
    mask_ = copy(mask)
    filter_on_area!(mask_, thr)
    return mask_
end

function post_processing!(img; thr_ar = 500, thr_p = 200)
    img_ch = channelview(img)
    r = @view img_ch[1, :, :]
    g = @view img_ch[2, :, :]
    recovery_hidden_AR!(img)
    filter_on_area!(r, thr_ar)
    filter_on_area!(g, thr_p)
    return nothing
end

function post_processing(img; thr_ar = 500, thr_p = 200)
    img_ = copy(img)
    post_processing!(img_; thr_ar, thr_p)
    return img_
end

function process_folder(src_folder, target_folder; f_p = post_processing)
    files = readdir(src_folder, join = true)
    !ispath(target_folder) && mkdir(target_folder)
    @showprogress for file in files
        img = load(file) |> f_p
        save(joinpath(target_folder, basename(file)), img)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    s_root = joinpath(@__DIR__, "../../data/predict_onehot")
    srcs = readdir(s_root, join = true)
    t_root = joinpath(@__DIR__, "../../data/predict_post")
    !ispath(t_root) && mkdir(t_root)
    targets = joinpath.(t_root, basename.(srcs))
    for (s, t) in zip(srcs, targets)
        process_folder(s, t, f_p = recovery_hidden_AR)
    end
end
