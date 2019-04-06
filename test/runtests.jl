using Printf
using Statistics
using Test
using CUDAdrv, CuArrays, CUDAnative

push!(LOAD_PATH, ".")

using CuDiff

function deriv_num(f, x::T) where T<:Number
    u = f(x)
    eps = T(1e-3)
    du = (f(x+eps) - f(x-eps)) / (2*eps)
    return u, du
end


function kernel(f, d_x, d_a, d_da, d_b, d_db)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    d_a[i], d_da[i] = deriv(f, d_x[i])
    d_b[i], d_db[i] = deriv_num(f, d_x[i])
    return nothing
end

function test_proc(f)
    x = rand(Float32, 1000)

    a = similar(x)
    da = similar(x)
    b = similar(x)
    db = similar(x)

    d_x = CuArray(x)
    d_a = similar(d_x)
    d_da = similar(d_x)
    d_b = similar(d_x)
    d_db = similar(d_x)

    CUDAnative.@cuda threads=1000 kernel(f, d_x, d_a, d_da, d_b, d_db)

    copy!(a, d_a)
    copy!(da, d_da)
    copy!(b, d_b)
    copy!(db, d_db)

    return x, a, da, b, db
end

function test_all()
    x = sort(rand(Float32, 1000))

    a = similar(x)
    da = similar(x)
    b = similar(x)
    db = similar(x)

    d_x = CuArray(x)
    d_a = similar(d_x)
    d_da = similar(d_x)
    d_b = similar(d_x)
    d_db = similar(d_x)

    function run(f)
        CUDAnative.@cuda threads=1000 kernel(f, d_x, d_a, d_da, d_b, d_db)
        copy!(a, d_a)
        copy!(da, d_da)
        copy!(b, d_b)
        copy!(db, d_db)
        err1 = quantile!(abs.(a .- b), 0.99)
        err2 = quantile!(abs.(da .- db), 0.99)
        passed =  err1 < 1e-5 && err2 < 1e-3
        #Printf.@printf "%.6f\t%.6f\n" err1 err2
        if !passed
            Printf.@printf "%.6f\t%.6f\n" err1 err2
        end
        return passed
    end

    @printf "rationals:\t%s\n" @test run(x -> (2*x-0.65f0)/(4.3-3*x))
    @printf "sin:\t%s\n" @test run(x -> CUDAnative.sin(x))
    @printf "sinpi:\t%s\n" @test run(x -> CUDAnative.sinpi(x))
    # @printf "sin_fast:\t%s\n" @test run(x -> CUDAnative.sin_fast(x))
    @printf "cos:\t%s\n" @test run(x -> CUDAnative.cos(x))
    @printf "cospi:\t%s\n" @test run(x -> CUDAnative.cospi(x))
    #@printf "cos_fast:\t%s\n" @test run(x -> CUDAnative.cos_fast(x))
    @printf "tan:\t%s\n" @test run(x -> -CUDAnative.tan(x))
    #@printf "tan_fast:\t%s\n" @test run(x -> -CUDAnative.tan_fast(x))
    @printf "acos:\t%s\n" @test run(x -> CUDAnative.acos(0.9*x))
    @printf "asin:\t%s\n" @test run(x -> CUDAnative.asin(0.9*x))
    @printf "atan:\t%s\n" @test run(x -> CUDAnative.atan(x))
    @printf "cosh:\t%s\n" @test run(x -> CUDAnative.cosh(x))
    @printf "sinh:\t%s\n" @test run(x -> CUDAnative.sinh(x))
    @printf "tanh:\t%s\n" @test run(x -> CUDAnative.tanh(x))
    @printf "acosh:\t%s\n" @test run(x -> CUDAnative.acosh(1.1+x))
    @printf "asinh:\t%s\n" @test run(x -> CUDAnative.asinh(x))
    @printf "atanh:\t%s\n" @test run(x -> CUDAnative.atanh(0.9*x))
    #@printf "atanh2:\t%s\n" @test run(x -> CUDAnative.atanh2(x,x))
    @printf "exp:\t%s\n" @test run(x -> CUDAnative.exp(x))
    @printf "exp_fast:\t%s\n" @test run(x -> CUDAnative.exp_fast(x))
    @printf "exp2:\t%s\n" @test run(x -> CUDAnative.exp2(x))
    @printf "exp10:\t%s\n" @test run(x -> CUDAnative.exp10(x))
    @printf "expm1:\t%s\n" @test run(x -> CUDAnative.expm1(x))
    @printf "log:\t%s\n" @test run(x -> CUDAnative.log(0.1+x))
    #@printf "log_fast:\t%s\n" @test run(x -> CUDAnative.log_fast(0.1+x))
    @printf "log10:\t%s\n" @test run(x -> CUDAnative.log10(0.1+x))
    #@printf "log10_fast:\t%s\n" @test run(x -> CUDAnative.log10_fast(0.1+x))
    @printf "log2:\t%s\n" @test run(x -> CUDAnative.log2(0.1+x))
    #@printf "log2_fast:\t%s\n" @test run(x -> CUDAnative.log2_fast(0.1+x))
    @printf "log1p:\t%s\n" @test run(x -> CUDAnative.log1p(x+0.1))
    @printf "abs:\t%s\n" @test run(x -> CUDAnative.abs(x-0.5))
    #@printf "saturate:\t%s\n" @test run(x -> CUDAnative.saturate(2*(x-0.5)))
    @printf "sqrt:\t%s\n" @test run(x -> CUDAnative.sqrt(x+0.1))
    @printf "rsqrt:\t%s\n" @test run(x -> CUDAnative.rsqrt(x+0.1))
    @printf "cbrt:\t%s\n" @test run(x -> CUDAnative.cbrt(x+0.1))
    @printf "rcbrt:\t%s\n" @test run(x -> CUDAnative.rcbrt(x+0.1))
    @printf "pow:\t%s\n" @test run(x -> CUDAnative.pow(2+x,1.5f0))
    @printf "erf:\t%s\n" @test run(x -> CUDAnative.erf(10*x-1))
    @printf "erfc:\t%s\n" @test run(x -> CUDAnative.erfc(10*x-1))
    @printf "erfinv:\t%s\n" @test run(x -> CUDAnative.erfinv(0.01+x*0.5))
    @printf "erfcinv:\t%s\n" @test run(x -> CUDAnative.erfcinv(0.2+x*0.5))

    @printf "<:\t%s\n" @test run(x -> x < 0.5 ? CUDAnative.sin(x) : CUDAnative.cos(x))
    @printf ">:\t%s\n" @test run(x -> x > 0.5 ? CUDAnative.sin(x) : CUDAnative.cos(x))
    @printf "<=:\t%s\n" @test run(x -> x <= 0.5 ? CUDAnative.sin(x) : CUDAnative.cos(x))
    @printf ">=:\t%s\n" @test run(x -> x >= 0.5 ? CUDAnative.sin(x) : CUDAnative.cos(x))
    @printf "==:\t%s\n" @test run(x -> x == 0.5 ? CUDAnative.sin(x) : CUDAnative.cos(x))
    @printf "!=:\t%s\n" @test run(x -> x != 0.5 ? CUDAnative.sin(x) : CUDAnative.cos(x))
end

test_all()
