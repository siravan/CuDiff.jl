import CUDAnative

struct Dual{T}
    u::T
    du::T
end

import Base: promote_rule, convert
import Base.+, Base.-, Base.*, Base./, Base.^
import Base.>, Base.<, Base.>=, Base.<=, Base.==, Base.!=
import Base.isapprox

Dual(c::T) where T<:Number = Dual(c, zero(c))
convert(::Type{Dual{T}}, c::T) where T<:Number = Dual(c, zero(c))
promote_rule(::Type{Dual{T}}, ::Type{T}) where T<:Number = Dual{T}

(+)(x::Dual) = Dual(x.u, x.du)
(+)(x::Dual{T}, y::Dual{T}) where T = Dual(x.u+y.u, x.du+y.du)
(+)(x::Dual{T}, c::Number) where T = (+)(promote(x,T(c))...)
(+)(c::Number, y::Dual{T}) where T = (+)(promote(T(c),y)...)

(-)(x::Dual) = Dual(-x.u, -x.du)
(-)(x::Dual{T}, y::Dual{T}) where T = Dual(x.u-y.u, x.du-y.du)
(-)(x::Dual{T}, c::Number) where T = (-)(promote(x,T(c))...)
(-)(c::Number, y::Dual{T}) where T = (-)(promote(T(c),y)...)

(*)(x::Dual{T}, y::Dual{T}) where T = Dual(x.u*y.u, x.u*y.du+y.u*x.du)
(*)(x::Dual{T}, c::Number) where T = (*)(promote(x,T(c))...)
(*)(c::Number, y::Dual{T}) where T = (*)(promote(T(c),y)...)

(/)(x::Dual{T}, y::Dual{T}) where T = Dual(x.u/y.u, (x.du*y.u-x.u*y.du)/sqr(y.u))
(/)(x::Dual{T}, c::Number) where T = (/)(promote(x,T(c))...)
(/)(c::Number, y::Dual{T}) where T = (/)(promote(T(c),y)...)

CUDAnative.sin(x::Dual) = Dual(CUDAnative.sin(x.u), x.du*CUDAnative.cos(x.u))
CUDAnative.sin_fast(x::Dual) = Dual(CUDAnative.sin_fast(x.u), x.du*CUDAnative.cos_fast(x.u))
CUDAnative.sinpi(x::Dual{T}) where T = Dual(CUDAnative.sinpi(x.u), T(pi)*x.du*CUDAnative.cospi(x.u))

CUDAnative.cos(x::Dual) = Dual(CUDAnative.cos(x.u), -x.du*CUDAnative.sin(x.u))
CUDAnative.cos_fast(x::Dual) = Dual(CUDAnative.cos_fast(x.u), -x.du*CUDAnative.sin_fast(x.u))
CUDAnative.cospi(x::Dual{T}) where T = Dual(CUDAnative.cospi(x.u), -T(pi)*x.du*CUDAnative.sinpi(x.u))

function CUDAnative.tan(x::Dual)
    t = CUDAnative.tan(x.u)
    return Dual(t, x.du*(one(x.u)+sqr(t)))
end

function CUDAnative.tan_fast(x::Dual)
    t = CUDAnative.tan_fast(x.u)
    return Dual(t, x.du*(one(x.u)+sqr(t)))
end

CUDAnative.acos(x::Dual) = Dual(CUDAnative.acos(x.u), -x.du*CUDAnative.rsqrt(one(x.u)-sqr(x.u)))
CUDAnative.asin(x::Dual) = Dual(CUDAnative.asin(x.u), x.du*CUDAnative.rsqrt(one(x.u)-sqr(x.u)))
CUDAnative.atan(x::Dual) = Dual(CUDAnative.atan(x.u), x.du/(one(x.u)+sqr(x.u)))

CUDAnative.atan2(x::Dual{T}, y::Dual{T}) where T = Dual(CUDAnative.atan2(x.u,y.u), x.du/(one(x.u)+sqr(x.u/y.u)))
CUDAnative.atan2(x::Dual{T}, c::Number) where T = CUDAnative.atan2(promote(x,T(c))...)
CUDAnative.atan2(c::Number, x::Dual{T}) where T = CUDAnative.atan2(promote(T(c),x)...)

CUDAnative.cosh(x::Dual) = Dual(CUDAnative.cosh(x.u), x.du*CUDAnative.sinh(x.u))
CUDAnative.sinh(x::Dual) = Dual(CUDAnative.sinh(x.u), x.du*CUDAnative.cosh(x.u))

function CUDAnative.tanh(x::Dual)
    t = CUDAnative.tanh(x.u)
    return Dual(t, x.du*(one(x.u)-sqr(t)))
end

CUDAnative.acosh(x::Dual) = Dual(CUDAnative.acosh(x.u), x.du*CUDAnative.rsqrt(sqr(x.u)-one(x.u)))
CUDAnative.asinh(x::Dual) = Dual(CUDAnative.asinh(x.u), x.du*CUDAnative.rsqrt(sqr(x.u)+one(x.u)))
CUDAnative.atanh(x::Dual) = Dual(CUDAnative.atanh(x.u), x.du*inverse(one(x.u)-sqr(x.u)))

function CUDAnative.exp(x::Dual) # = Dual(CUDAnative.exp(x.u), x.du*CUDAnative.exp(x.u))
     t = CUDAnative.exp(x.u)
     return Dual(t, x.du*t)
 end

function CUDAnative.exp_fast(x::Dual)
    t = CUDAnative.exp_fast(x.u)
    return Dual(t, x.du*t)
end

function CUDAnative.exp2(x::Dual{T}) where T
    t = CUDAnative.exp2(x.u)
    return Dual(t, T(0.6931471805599453)*x.du*t)
end

function CUDAnative.exp10(x::Dual{T}) where T
    t = CUDAnative.exp10(x.u)
    return Dual(t, T(2.302585092994046)*x.du*t)
end

function CUDAnative.expm1(x::Dual)
    t = CUDAnative.expm1(x.u)
    return Dual(t, x.du*(one(x.u)+t))
end

CUDAnative.log(x::Dual) = Dual(CUDAnative.log(x.u), x.du/x.u)
CUDAnative.log_fast(x::Dual) = Dual(CUDAnative.log_fast(x.u), x.du/x.u)
CUDAnative.log10(x::Dual{T}) where T = Dual(CUDAnative.log10(x.u), T(0.43429448190325176)*x.du/x.u)
CUDAnative.log10_fast(x::Dual{T}) where T = Dual(CUDAnative.log10_fast(x.u), T(0.43429448190325176)*x.du/x.u)
CUDAnative.log2(x::Dual{T}) where T = Dual(CUDAnative.log2(x.u), T(1.4426950408889634)*x.du/x.u)
CUDAnative.log2_fast(x::Dual{T}) where T = Dual(CUDAnative.log2_fast(x.u), T(1.4426950408889634)*x.du/x.u)
CUDAnative.log1p(x::Dual) = Dual(CUDAnative.log1p(x.u), x.du/(one(x.u)+x.u))

CUDAnative.abs(x::Dual) = Dual(CUDAnative.abs(x.u), x.u > zero(x.u) ? x.du : -x.du)
CUDAnative.saturate(x::Dual) = Dual(CUDAnative.saturate(x.u), one(x.u) >= x.u >= zero(x.u) ? one(x.u) : zero(x.u))
CUDAnative.sqrt(x::Dual{T}) where T = Dual(CUDAnative.sqrt(x.u), T(0.5)*x.du/CUDAnative.sqrt(x.u))
CUDAnative.rsqrt(x::Dual{T}) where T = Dual(CUDAnative.rsqrt(x.u), T(-0.5)*x.du/(x.u*CUDAnative.sqrt(x.u)))
CUDAnative.cbrt(x::Dual{T}) where T = Dual(CUDAnative.cbrt(x.u), T(1/3)*x.du/sqr(CUDAnative.cbrt(x.u)))
CUDAnative.rcbrt(x::Dual{T}) where T = Dual(CUDAnative.rcbrt(x.u), T(-1/3)*x.du/sqr(sqr(CUDAnative.cbrt(x.u))))

function CUDAnative.hypot(x::Dual{T}, y::Dual{T}) where T
    t = CUDAnative.hypot(x.u, y.u)
    return Dual(t, (x.u*x.du+y.u*y.du)/t)
end

CUDAnative.hypot(x::Dual{T}, c::Number) where T = CUDAnative.hypot(promote(x,T(c))...)
CUDAnative.hypot(c::Number, x::Dual{T}) where T = CUDAnative.hypot(promote(T(c),x)...)

CUDAnative.erf(x::Dual{T}) where T = Dual(CUDAnative.erf(x.u), T(1.1283791670955126)*x.du*CUDAnative.exp(-sqr(x.u)))
CUDAnative.erfc(x::Dual{T}) where T = Dual(CUDAnative.erfc(x.u), T(-1.1283791670955126)*x.du*CUDAnative.exp(-sqr(x.u)))

function CUDAnative.erfinv(x::Dual{T}) where T
    t = CUDAnative.erfinv(x.u)
    Dual(t, T(0.8862269254527579)*x.du*CUDAnative.exp(sqr(t)))
end

function CUDAnative.erfcinv(x::Dual{T}) where T
    t = CUDAnative.erfcinv(x.u)
    Dual(t, T(-0.8862269254527579)*x.du*CUDAnative.exp(sqr(t)))
end

function CUDAnative.pow(x::Dual, p)
    t = CUDAnative.pow(x.u, p)
    return Dual(t, x.du*p*t/x.u)
end

function (^)(x::Dual, p)
    t = x.u ^ p
    return Dual(t, x.du*p*t/x.u)
end

#############################################################################

CUDAnative.ceil(x::Dual) = Dual(CUDAnative.ceil(x.u), zero(x.u))
CUDAnative.floor(x::Dual) = Dual(CUDAnative.floor(x.u), zero(x.u))

CUDAnative.min(x::Dual{T}, y::Dual{T}) where T = CUDAnative.min(x.u, y.u)
CUDAnative.min(x::Dual{T}, c::Number) where T = CUDAnative.min(x.u, c)
CUDAnative.min(c::Number, x::Dual{T}) where T = CUDAnative.min(c, x.u)
CUDAnative.max(x::Dual{T}, y::Dual{T}) where T = CUDAnative.max(x.u, y.u)
CUDAnative.max(x::Dual{T}, c::Number) where T = CUDAnative.max(x.u, c)
CUDAnative.max(c::Number, x::Dual{T}) where T = CUDAnative.max(c, x.u)
CUDAnative.isfinite(x::Dual) = CUDAnative.isfinite(x.u)
CUDAnative.isinf(x::Dual) = CUDAnative.isinf(x.u)
CUDAnative.isnan(x::Dual) = CUDAnative.isnan(x.u)
CUDAnative.nearbyint(x::Dual) = CUDAnative.nearbyint(x.u)
CUDAnative.nextafter(x::Dual) = CUDAnative.nextafter(x.u)
CUDAnative.signbit(x::Dual) = CUDAnative.signbit(x.u)

(<)(x::Dual{T}, y::Dual{T}) where T = (x.u < y.u)
(<)(x::Dual{T}, c::Number) where T = (x.u < c)
(<)(c::Number, x::Dual{T}) where T = (c < x.u)

(>)(x::Dual{T}, y::Dual{T}) where T = (x.u > y.u)
(>)(x::Dual{T}, c::Number) where T = (x.u > c)
(>)(c::Number, x::Dual{T}) where T = (c > x.u)

(<=)(x::Dual{T}, y::Dual{T}) where T = (x.u <= y.u)
(<=)(x::Dual{T}, c::Number) where T = (x.u <= c)
(<=)(c::Number, x::Dual{T}) where T = (c <= x.u)

(>=)(x::Dual{T}, y::Dual{T}) where T = (x.u >= y.u)
(>=)(x::Dual{T}, c::Number) where T = (x.u >= c)
(>=)(c::Number, x::Dual{T}) where T = (c >= x.u)

(==)(x::Dual{T}, y::Dual{T}) where T = (x.u == y.u)
(==)(x::Dual{T}, c::Number) where T = (x.u == c)
(==)(c::Number, x::Dual{T}) where T = (c == x.u)

(!=)(x::Dual{T}, y::Dual{T}) where T = (x.u != y.u)
(!=)(x::Dual{T}, c::Number) where T = (x.u != c)
(!=)(c::Number, x::Dual{T}) where T = (c != x.u)

isapprox(x::Dual{T}, y::Dual{T}) where T = isapprox(x.u, y.u)
isapprox(x::Dual{T}, c::Number) where T = isapprox(x.u, c)
isapprox(c::Number, x::Dual{T}) where T = isapprox(c, x.u)

#############################################################################
