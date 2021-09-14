//
// Created by Acacia on 9/12/2021.
//

#ifndef MPM_MATRIX_H
#define MPM_MATRIX_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <iostream>

template<class T>
CUDA_CALLABLE_MEMBER void swap(T& a, T& b)
{
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

template<typename T>
class Mat2d;

template<typename T>
class Vec2d
{
    T elem[2]{ 0, 0 };
public:
    CUDA_CALLABLE_MEMBER Vec2d(T a, T b)
    {
        elem[0] = a;
        elem[1] = b;
    }

    CUDA_CALLABLE_MEMBER explicit Vec2d(T a)
    {
        elem[0] = a;
        elem[1] = a;
    }

    Vec2d() = default;

    CUDA_CALLABLE_MEMBER Vec2d(const Vec2d& arg)
    {
        elem[0] = arg.elem[0];
        elem[1] = arg.elem[1];
    }

    CUDA_CALLABLE_MEMBER Vec2d& operator=(const Vec2d& arg)
    {
        if (&arg == this) return (*this);
        elem[0] = arg.elem[0];
        elem[1] = arg.elem[1];
        return (*this);
    }

    ~Vec2d() = default;

    CUDA_CALLABLE_MEMBER T& operator[](int n)
    {
        return elem[n];
    }

    CUDA_CALLABLE_MEMBER const T& operator[](int n) const
    {
        return elem[n];
    }

    CUDA_CALLABLE_MEMBER [[maybe_unused]] [[maybe_unused]] T sum() const
    {
        return elem[0] + elem[1];
    }

    CUDA_CALLABLE_MEMBER T norm() const
    {
        return sqrtf(elem[0] * elem[0] + elem[1] * elem[1]);
    }

    template<typename U>
    CUDA_CALLABLE_MEMBER Vec2d<U> cast() const
    {
        return Vec2d<U>{ static_cast<U>(elem[0]), static_cast<U>(elem[1]) };
    }

    CUDA_CALLABLE_MEMBER Mat2d<T> outer(const Vec2d<T>& arg) const
    {
        Mat2d<T> res;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                res(i, j) = elem[i] * arg.elem[j];
        return res;
    }

    CUDA_CALLABLE_MEMBER T cross(const Vec2d<T>& arg) const
    {
        return elem[0] * arg.elem[1] - elem[1] * arg.elem[0];
    }

    CUDA_CALLABLE_MEMBER T dot(const Vec2d<T>& arg) const
    {
        return elem[0] * arg.elem[0] + elem[1] * arg.elem[1];
    }

//------------------------------------------------------------------------------

    CUDA_CALLABLE_MEMBER Vec2d operator+(T arg) const
    {
        Vec2d res{ (*this) };
        res[0] += arg;
        res[1] += arg;
        return res;
    }

    CUDA_CALLABLE_MEMBER Vec2d operator+(const Vec2d& arg) const
    {
        Vec2d res{ (*this) };
        res[0] += arg.elem[0];
        res[1] += arg.elem[1];
        return res;
    }

    CUDA_CALLABLE_MEMBER Vec2d& operator+=(const Vec2d& arg)
    {
        elem[0] += arg.elem[0];
        elem[1] += arg.elem[1];
        return (*this);
    }

    CUDA_CALLABLE_MEMBER Vec2d operator-(T arg) const
    {
        Vec2d res{ (*this) };
        res[0] -= arg;
        res[1] -= arg;
        return res;
    }


    CUDA_CALLABLE_MEMBER Vec2d operator-(const Vec2d& arg) const
    {
        Vec2d res{ (*this) };
        res[0] -= arg.elem[0];
        res[1] -= arg.elem[1];
        return res;
    }

    CUDA_CALLABLE_MEMBER Vec2d operator*(T arg) const
    {
        Vec2d res{ (*this) };
        res[0] *= arg;
        res[1] *= arg;
        return res;
    }

    CUDA_CALLABLE_MEMBER Vec2d operator*(const Vec2d& arg) const
    {
        Vec2d res{ (*this) };
        res[0] *= arg.elem[0];
        res[1] *= arg.elem[1];
        return res;
    }
};

//------------------------------------------------------------------------------

template<typename T>
CUDA_CALLABLE_MEMBER inline
std::ostream& operator<<(std::ostream& os, const Vec2d<T>& v)
{
    os << '[' << v[0] << ' ' << v[1] << ']';
    return os;
}

//------------------------------------------------------------------------------

template<typename T>
class Mat2d
{
    T elem[2][2];
public:
    CUDA_CALLABLE_MEMBER [[maybe_unused]] [[maybe_unused]] Mat2d(T a, T b, T c,
            T d)
    {
        elem[0][0] = a;
        elem[0][1] = b;
        elem[1][0] = c;
        elem[1][1] = d;
    }

    CUDA_CALLABLE_MEMBER [[maybe_unused]] [[maybe_unused]] explicit Mat2d(T a)
    {
        elem[0][0] = a;
        elem[0][1] = 0;
        elem[1][0] = 0;
        elem[1][1] = a;
    }

    CUDA_CALLABLE_MEMBER Mat2d()
    {
        elem[0][0] = 0;
        elem[0][1] = 0;
        elem[1][0] = 0;
        elem[1][1] = 0;
    }

    CUDA_CALLABLE_MEMBER Mat2d(const Mat2d& arg)
    {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                elem[i][j] = arg.elem[i][j];
    }

    CUDA_CALLABLE_MEMBER Mat2d& operator=(const Mat2d& arg)
    {
        if (&arg == this) return (*this);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                elem[i][j] = arg.elem[i][j];
        return (*this);
    }

    CUDA_CALLABLE_MEMBER T& operator()(int n, int m)
    {
        return elem[n][m];
    }

    CUDA_CALLABLE_MEMBER const T& operator()(int n, int m) const
    {
        return elem[n][m];
    }

    CUDA_CALLABLE_MEMBER Mat2d tran()
    {
        Mat2d<T> res{ (*this) };
        swap(res.elem[0][1], res.elem[1][0]);
        return res;
    }

    CUDA_CALLABLE_MEMBER T det()
    {
        return elem[0][0] * elem[1][1] - elem[0][1] * elem[1][0];
    }

    CUDA_CALLABLE_MEMBER [[maybe_unused]] [[maybe_unused]] Mat2d inv()
    {
        Mat2d<T> res{ (*this) };
        res.elem[0][1] *= -1.0;
        res.elem[1][0] *= -1.0;
        swap(res.elem[0][0], res.elem[1][1]);
        res = res * (1.0 / res.det());
        return res;
    }

    CUDA_CALLABLE_MEMBER T tr()
    {
        return elem[0][0] + elem[1][1];
    }


    CUDA_CALLABLE_MEMBER std::tuple<Mat2d, Mat2d, Mat2d> svd_solve()
    {
        Mat2d U, sig, Vt;
        float E = (float(elem[0][0]) + elem[1][1]) / 2;
        float F = (float(elem[0][0]) - elem[1][1]) / 2;
        float g = (float(elem[1][0]) + elem[0][1]) / 2;
        float H = (float(elem[1][0]) - elem[0][1]) / 2;

        float Q = sqrt(E * E + H * H);
        float r = sqrt(F * F + g * g);

        float a1 = atan2(g, F);
        float a2 = atan2(H, E);

        float theta = (a2 - a1) / 2;
        float phi = (a2 + a1) / 2;

        U(0, 0) = cos(phi);
        U(0, 1) = -sin(phi);
        U(1, 0) = -U(0, 1);
        U(1, 1) = U(0, 0);

        sig(0, 0) = Q + r;
        sig(0, 1) = 0;
        sig(1, 0) = 0;
        sig(1, 1) = Q - r;

        Vt(0, 0) = cos(theta);
        Vt(0, 1) = -sin(theta);
        Vt(1, 0) = -Vt(0, 1);
        Vt(1, 1) = Vt(0, 0);

        return { U, sig, Vt };
    }

//------------------------------------------------------------------------------

    CUDA_CALLABLE_MEMBER Mat2d operator+(T rhs)
    {
        Mat2d res{ *this };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                res.elem[i][j] += rhs;
        return res;
    }

    CUDA_CALLABLE_MEMBER Mat2d operator+(const Mat2d& rhs)
    {
        Mat2d res{ *this };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                res.elem[i][j] += rhs.elem[i][j];
        return res;
    }

    CUDA_CALLABLE_MEMBER Mat2d operator-(const Mat2d& rhs)
    {
        Mat2d res{ *this };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                res.elem[i][j] -= rhs.elem[i][j];
        return res;
    }

    CUDA_CALLABLE_MEMBER Mat2d operator*(T rhs)
    {
        Mat2d res{ *this };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                res.elem[i][j] *= rhs;
        return res;
    }

    CUDA_CALLABLE_MEMBER Mat2d operator*(const Mat2d& rhs)
    {
        Mat2d res{ *this };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                res.elem[i][j] *= rhs.elem[i][j];
        return res;
    }

    CUDA_CALLABLE_MEMBER Mat2d operator%(const Mat2d& rhs)
    {
        Mat2d res;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                res.elem[i][j] = elem[i][0] * rhs.elem[0][j]
                                 + elem[i][1] * rhs.elem[1][j];
        return res;
    }

    CUDA_CALLABLE_MEMBER Vec2d<T> operator%(const Vec2d<T>& rhs)
    {
        Vec2d<T> res;
        for (int i = 0; i < 2; i++)
            res[i] = elem[i][0] * rhs[0] + elem[i][1] * rhs[1];
        return res;
    }
};

//------------------------------------------------------------------------------

template<typename T>
CUDA_CALLABLE_MEMBER inline
std::ostream& operator<<(std::ostream& os, const Mat2d<T>& m)
{
    os << '[' << m(0, 0) << ' ' << m(0, 1) << "]\n["
       << m(1, 0) << ' ' << m(1, 1) << ']';
    return os;
}

template<typename T>
class Mat3d;

template<typename T>
class Vec3d
{
    T elem[3];
public:
    CUDA_CALLABLE_MEMBER Vec3d()
    {
        elem[0] = elem[1] = elem[2] = 0;
    }

    CUDA_CALLABLE_MEMBER Vec3d(Vec2d<T> v, T a)
    {
        elem[0] = v[0];
        elem[1] = v[1];
        elem[2] = a;
    }

    CUDA_CALLABLE_MEMBER Mat3d<T> outer(const Vec3d& rhs)
    const
    {
        Mat3d<T> res;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res(i, j) = elem[i] * rhs.elem[j];
        return res;
    }

    CUDA_CALLABLE_MEMBER T& operator[](int n)
    {
        return elem[n];
    }

    CUDA_CALLABLE_MEMBER const T& operator[](int n) const
    {
        return elem[n];
    }

    CUDA_CALLABLE_MEMBER [[maybe_unused]] [[maybe_unused]] T length2() const
    {
        T res = 0;
        for (auto& i: elem) res += i * i;
        return res;
    }

//------------------------------------------------------------------------------

    CUDA_CALLABLE_MEMBER Vec3d& operator+=(const Vec3d& rhs)
    {
        for (int i = 0; i < 3; i++) elem[i] += rhs.elem[i];
        return (*this);
    }

    CUDA_CALLABLE_MEMBER Vec3d operator*(T rhs)
    {
        Vec3d res(*this);
        for (int i = 0; i < 3; i++) res.elem[i] *= rhs;
        return res;
    }

};

//------------------------------------------------------------------------------

template<typename T>
class Mat3d
{
    T elem[3][3];
public:
    CUDA_CALLABLE_MEMBER Mat3d()
    {
        for (auto& i: elem)
            for (auto& j: i)
                j = 0;
    }

    CUDA_CALLABLE_MEMBER Mat3d(const Mat3d& arg)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                elem[i][j] = arg.elem[i][j];
    }

    CUDA_CALLABLE_MEMBER T& operator()(int i, int j)
    {
        return elem[i][j];
    }

    CUDA_CALLABLE_MEMBER const T& operator()(int i, int j)
    const
    {
        return elem[i][j];
    }

    CUDA_CALLABLE_MEMBER T det()
    {
        return elem[0][0] *
               (elem[1][1] * elem[2][2] - elem[2][1] * elem[1][2]) -
               elem[0][1] *
               (elem[1][0] * elem[2][2] - elem[1][2] * elem[2][0]) +
               elem[0][2] * (elem[1][0] * elem[2][1] - elem[1][1] * elem[2][0]);
    }

    CUDA_CALLABLE_MEMBER Mat3d inv()
    {
        Mat3d res;
        T inv_det = 1.0 / det();
        res(0, 0) =
                (elem[1][1] * elem[2][2] - elem[2][1] * elem[1][2]) * inv_det;
        res(0, 1) =
                (elem[0][2] * elem[2][1] - elem[0][1] * elem[2][2]) * inv_det;
        res(0, 2) =
                (elem[0][1] * elem[1][2] - elem[0][2] * elem[1][1]) * inv_det;
        res(1, 0) =
                (elem[1][2] * elem[2][0] - elem[1][0] * elem[2][2]) * inv_det;
        res(1, 1) =
                (elem[0][0] * elem[2][2] - elem[0][2] * elem[2][0]) * inv_det;
        res(1, 2) =
                (elem[1][0] * elem[0][2] - elem[0][0] * elem[1][2]) * inv_det;
        res(2, 0) =
                (elem[1][0] * elem[2][1] - elem[2][0] * elem[1][1]) * inv_det;
        res(2, 1) =
                (elem[2][0] * elem[0][1] - elem[0][0] * elem[2][1]) * inv_det;
        res(2, 2) =
                (elem[0][0] * elem[1][1] - elem[1][0] * elem[0][1]) * inv_det;
        return res;
    }

//------------------------------------------------------------------------------

    CUDA_CALLABLE_MEMBER Mat3d& operator+=(const Mat3d& rhs)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                elem[i][j] += rhs.elem[i][j];
        return (*this);
    }

    CUDA_CALLABLE_MEMBER Mat3d operator*(T rhs)
    {
        Mat3d res{ *this };
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res.elem[i][j] *= rhs;
        return res;
    }

    CUDA_CALLABLE_MEMBER Mat3d operator%(const Mat3d& rhs)
    {
        Mat3d res;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    res.elem[i][j] += elem[i][k] * rhs.elem[k][j];
        return res;
    }

    CUDA_CALLABLE_MEMBER Vec3d<T> operator%(const Vec3d<T>& rhs)
    {
        Vec3d<T> res;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res[i] += elem[i][j] * rhs[j];
        return res;
    }

};


#endif //MPM_MATRIX_H
