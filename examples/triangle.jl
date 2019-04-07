import PyPlot; plt = PyPlot;
using CuArrays, CUDAnative
using CuDiff

function triangle_series(x)
    series_sum = 0f0

    for k = 1000:-1:1
        n = 2*k-1
        series_sum += (k & 1 == 1 ? +1f0 : -1f0) * CUDAnative.sin(n*x) / (n*n)
    end

    return series_sum
end

function kernel(x, y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    y[i] = triangle_series(x[i])
    return nothing
end

function kernel_deriv(x, y, dy)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    y[i], dy[i] = CuDiff.derivative(triangle_series, x[i])
    return nothing
end


x = range(0, 4*pi, length=1024)
y = similar(x)
dy = similar(x)

d_x = CuArray(x)
d_y = similar(d_x)
d_dy = similar(d_x)

CUDAnative.@cuda threads=1024 kernel_deriv(d_x, d_y, d_dy)

copy!(y, d_y)
copy!(dy, d_dy)

plt.plot(x, y)
plt.plot(x, dy)
plt.xlabel("x")
plt.ylabel("y")
