import CUDAnative

struct Single{T}
    u::T
end

import Base: promote_rule, convert
import Base.+, Base.-, Base.*, Base./, Base.^
import Base.>, Base.<, Base.>=, Base.<=, Base.==, Base.!=
import Base.isapprox
import Base.zero, Base.one

zero(::Single{T}) where T = Single(zero(T))
one(::Single{T}) where T = Single(one(T))
convert(::Type{Single{T}}, c::T) where T<:Number = Single(c)
promote_rule(::Type{Single{T}}, ::Type{T}) where T<:Number = Single{T}

(+)(x::Single) = Single(x.u)
(+)(x::Single{T}, y::Single{T}) where T = Single(x.u+y.u)
(+)(x::Single{T}, c::Number) where T = (+)(promote(x,T(c))...)
(+)(c::Number, y::Single{T}) where T = (+)(promote(T(c),y)...)

(-)(x::Single) = Single(-x.u)
(-)(x::Single{T}, y::Single{T}) where T = Single(x.u-y.u)
(-)(x::Single{T}, c::Number) where T = (-)(promote(x,T(c))...)
(-)(c::Number, y::Single{T}) where T = (-)(promote(T(c),y)...)

(*)(x::Single{T}, y::Single{T}) where T = Single(x.u*y.u)
(*)(x::Single{T}, c::Number) where T = (*)(promote(x,T(c))...)
(*)(c::Number, y::Single{T}) where T = (*)(promote(T(c),y)...)

(/)(x::Single{T}, y::Single{T}) where T = Single(x.u/y.u)
(/)(x::Single{T}, c::Number) where T = (/)(promote(x,T(c))...)
(/)(c::Number, y::Single{T}) where T = (/)(promote(T(c),y)...)

CUDAnative.sin(x::Single) = Single(CUDAnative.sin(x.u))
CUDAnative.sin_fast(x::Single) = Single(CUDAnative.sin_fast(x.u))
CUDAnative.sinpi(x::Single) = Single(CUDAnative.sinpi(x.u))

CUDAnative.cos(x::Single) = Single(CUDAnative.cos(x.u))
CUDAnative.cos_fast(x::Single) = Single(CUDAnative.cos_fast(x.u))
CUDAnative.cospi(x::Single) = Single(CUDAnative.cospi(x.u))

CUDAnative.tan(x::Single) = Single(CUDAnative.tan(x.u))
CUDAnative.tan_fast(x::Single) = Single(CUDAnative.tan_fast(x.u))

CUDAnative.acos(x::Single) = Single(CUDAnative.acos(x.u))
CUDAnative.asin(x::Single) = Single(CUDAnative.asin(x.u))
CUDAnative.atan(x::Single) = Single(CUDAnative.atan(x.u))

CUDAnative.atan2(x::Single, y::Single) = Single(CUDAnative.atan2(x.u,y.u))
CUDAnative.atan2(x::Single{T}, c::Number) where T = CUDAnative.atan2(promote(x,T(c))...)
CUDAnative.atan2(c::Number, x::Single{T}) where T = CUDAnative.atan2(promote(T(c),x)...)

CUDAnative.cosh(x::Single) = Single(CUDAnative.cosh(x.u))
CUDAnative.sinh(x::Single) = Single(CUDAnative.sinh(x.u))
CUDAnative.tanh(x::Single) = Single(CUDAnative.tanh(x.u))

CUDAnative.acosh(x::Single) = Single(CUDAnative.acosh(x.u))
CUDAnative.asinh(x::Single) = Single(CUDAnative.asinh(x.u))
CUDAnative.atanh(x::Single) = Single(CUDAnative.atanh(x.u))

CUDAnative.exp(x::Single) = Single(CUDAnative.exp(x.u))
CUDAnative.exp_fast(x::Single) = Single(CUDAnative.exp_fast(x.u))
CUDAnative.exp2(x::Single) = Single(CUDAnative.exp2(x.u))
CUDAnative.exp10(x::Single) = Single(CUDAnative.exp10(x.u))
CUDAnative.expm1(x::Single) = Single(CUDAnative.expm1(x.u))

CUDAnative.log(x::Single) = Single(CUDAnative.log(x.u))
CUDAnative.log_fast(x::Single) = Single(CUDAnative.log_fast(x.u))
CUDAnative.log10(x::Single) = Single(CUDAnative.log10(x.u))
CUDAnative.log10_fast(x::Single) = Single(CUDAnative.log10_fast(x.u))
CUDAnative.log2(x::Single) = Single(CUDAnative.log2(x.u))
CUDAnative.log2_fast(x::Single) = Single(CUDAnative.log2_fast(x.u))
CUDAnative.log1p(x::Single) = Single(CUDAnative.log1p(x.u))

CUDAnative.abs(x::Single) = Single(CUDAnative.abs(x.u))
CUDAnative.saturate(x::Single) = Single(CUDAnative.saturate(x.u))
CUDAnative.sqrt(x::Single) = Single(CUDAnative.sqrt(x.u))
CUDAnative.rsqrt(x::Single) = Single(CUDAnative.rsqrt(x.u))
CUDAnative.cbrt(x::Single) = Single(CUDAnative.cbrt(x.u))
CUDAnative.rcbrt(x::Single) = Single(CUDAnative.rcbrt(x.u))

CUDAnative.hypot(x::Single{T}, y::Single{T}) where T = Single(CUDAnative.hypot(x.u, y.u))
CUDAnative.hypot(x::Single{T}, c::Number) where T = CUDAnative.hypot(promote(x,T(c))...)
CUDAnative.hypot(c::Number, x::Single{T}) where T = CUDAnative.hypot(promote(T(c),x)...)

CUDAnative.erf(x::Single) = Single(CUDAnative.erf(x.u))
CUDAnative.erfc(x::Single) = Single(CUDAnative.erfc(x.u))
CUDAnative.erfinv(x::Single) = Single(CUDAnative.erfinv(x.u))
CUDAnative.erfcinv(x::Single) = Single(CUDAnative.erfcinv(x.u))
CUDAnative.pow(x::Single, p) = Single(CUDAnative.pow(x.u, p))
(^)(x::Single, p) = Single(x.u ^ p)

#############################################################################

CUDAnative.ceil(x::Single) = Single(CUDAnative.ceil(x.u))
CUDAnative.floor(x::Single) = Single(CUDAnative.floor(x.u))

CUDAnative.min(x::Single{T}, y::Single{T}) where T = CUDAnative.min(x.u, y.u)
CUDAnative.min(x::Single{T}, c::Number) where T = CUDAnative.min(x.u, c)
CUDAnative.min(c::Number, x::Single{T}) where T = CUDAnative.min(c, x.u)
CUDAnative.max(x::Single{T}, y::Single{T}) where T = CUDAnative.max(x.u, y.u)
CUDAnative.max(x::Single{T}, c::Number) where T = CUDAnative.max(x.u, c)
CUDAnative.max(c::Number, x::Single{T}) where T = CUDAnative.max(c, x.u)
CUDAnative.isfinite(x::Single) = CUDAnative.isfinite(x.u)
CUDAnative.isinf(x::Single) = CUDAnative.isinf(x.u)
CUDAnative.isnan(x::Single) = CUDAnative.isnan(x.u)
CUDAnative.nearbyint(x::Single) = CUDAnative.nearbyint(x.u)
CUDAnative.nextafter(x::Single) = CUDAnative.nextafter(x.u)
CUDAnative.signbit(x::Single) = CUDAnative.signbit(x.u)

(<)(x::Single{T}, y::Single{T}) where T = (x.u < y.u)
(<)(x::Single{T}, c::Number) where T = (x.u < c)
(<)(c::Number, x::Single{T}) where T = (c < x.u)

(>)(x::Single{T}, y::Single{T}) where T = (x.u > y.u)
(>)(x::Single{T}, c::Number) where T = (x.u > c)
(>)(c::Number, x::Single{T}) where T = (c > x.u)

(<=)(x::Single{T}, y::Single{T}) where T = (x.u <= y.u)
(<=)(x::Single{T}, c::Number) where T = (x.u <= c)
(<=)(c::Number, x::Single{T}) where T = (c <= x.u)

(>=)(x::Single{T}, y::Single{T}) where T = (x.u >= y.u)
(>=)(x::Single{T}, c::Number) where T = (x.u >= c)
(>=)(c::Number, x::Single{T}) where T = (c >= x.u)

(==)(x::Single{T}, y::Single{T}) where T = (x.u == y.u)
(==)(x::Single{T}, c::Number) where T = (x.u == c)
(==)(c::Number, x::Single{T}) where T = (c == x.u)

(!=)(x::Single{T}, y::Single{T}) where T = (x.u != y.u)
(!=)(x::Single{T}, c::Number) where T = (x.u != c)
(!=)(c::Number, x::Single{T}) where T = (c != x.u)

isapprox(x::Single{T}, y::Single{T}) where T = isapprox(x.u, y.u)
isapprox(x::Single{T}, c::Number) where T = isapprox(x.u, c)
isapprox(c::Number, x::Single{T}) where T = isapprox(c, x.u)

#############################################################################
