//
// Created by Acacia on 7/22/2021.
// Based on std_lib_facilities.h
//

#ifndef CPP__STD_LIB_FACILITIES_H_
#define CPP__STD_LIB_FACILITIES_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <list>
#include <forward_list>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <array>
#include <regex>
#include <random>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <fmt/core.h>

#ifdef UTILS_DEBUG
#define DEBUG(x) do { x } while(0)
#else
#define DEBUG(x) do {   } while(0)
#endif

#define ASSERT(x) if (!(x)) error("Assertion failed: ", #x)

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


//------------------------------------------------------------------------------


//using Unicode [[maybe_unused]] = long;

//------------------------------------------------------------------------------

namespace utils
{
    template<class T>
    std::string to_string(const T& t)
    {
        std::ostringstream os;
        os << t;
        return os.str();
    }

    struct Range_error : std::out_of_range
    {    // enhanced vector range error reporting
        [[maybe_unused]] unsigned int index;

        explicit Range_error(unsigned int i)
                : out_of_range("Range error: " + to_string(i)), index(i)
        {
        }
    };

    // disgusting macro hack to get a range checked vector:
#ifdef UTILS_DEBUG
    // trivially range-checked vector (no iterator checking):
    template<class T>
    struct Vector : public std::vector<T>
    {
        using size_type [[maybe_unused]] = typename std::vector<T>::size_type;

        /* #ifdef _MSC_VER
            // microsoft doesn't yet support C++11 inheriting constructors
            Vector() { }
            explicit Vector(size_type n) :std::vector<T>(n) {}
            Vector(size_type n, const T& v) :std::vector<T>(n, v) {}
            template <class I>
            Vector(I first, I last) : std::vector<T>(first, last) {}
            Vector(initializer_list<T> list) : std::vector<T>(list) {}
        */
        using std::vector<T>::vector;    // inheriting constructor

        T& operator[](unsigned int i) // rather than return at(i);
        {
            if (i < 0 || this->size() <= i)
                throw Range_error(i);
            return std::vector<T>::operator[](i);
        }
        const T& operator[](unsigned int i) const
        {
            if (i < 0 || this->size() <= i)
                throw Range_error(i);
            return std::vector<T>::operator[](i);
        }
    };
#else
    template<class T> using Vector = std::vector<T>;
#endif

// trivially range-checked string (no iterator checking):
    struct String : std::string
    {
        using size_type [[maybe_unused]] = std::string::size_type;
        //	using string::string;

        char& operator[](unsigned int i) // rather than return at(i);
        {
            if (/*i < 0 || */size() <= i)
                throw utils::Range_error(i);
            return std::string::operator[](i);
        }

        const char& operator[](unsigned int i) const
        {
            if (/*i < 0 || */size() <= i)
                throw utils::Range_error(i);
            return std::string::operator[](i);
        }
    };

    struct [[maybe_unused]] Exit : std::runtime_error
    {
        Exit()
                : runtime_error("Exit")
        {
        }
    };

// error() simply disguises throws:
    [[maybe_unused]] inline void error(const std::string& s)
    {
        throw std::runtime_error(s);
    }

    [[maybe_unused]] inline void error(const std::string& s,
            const std::string& s2)
    {
        error(s + s2);
    }

    [[maybe_unused]] inline void error(const std::string& s, int i)
    {
        std::ostringstream os;
        os << s << ": " << i;
        error(os.str());
    }

    template<class T>
    [[maybe_unused]] char* as_bytes(T& i)    // needed for binary I/O
    {
        void* addr = &i;    // get the address of the first byte
        // of memory used to store the object
        return static_cast<char*>(addr); // treat that memory as bytes
    }

    template<class T>
    [[maybe_unused]] const char*
    as_bytes_const(T& i)    // needed for binary I/O
    {
        const void* addr = &i;    // get the address of the first byte
        // of memory used to store the object
        return static_cast<const char*>(addr); // treat that memory as bytes
    }

    inline void keep_window_open()
    {
        std::cin.clear();
        std::cout << "Please enter a character to exit\n";
        char ch;
        std::cin >> ch;
    }

    [[maybe_unused]] inline void keep_window_open(const std::string& s)
    {
        if (s.empty()) return;
        std::cin.clear();
        std::cin.ignore(120, '\n');
        for (;;)
        {
            std::cout << "Please enter " << s << " to exit\n";
            std::string ss;
            while (std::cin >> ss && ss != s)
                std::cout << "Please enter " << s << " to exit\n";
            return;
        }
    }

    // error function to be used (only) until error() is introduced in Chapter 5:
    [[maybe_unused]] inline void simple_error(
            const std::string& s)    // write ``error: s and exit program
    {
        std::cerr << "error: " << s << '\n';
        keep_window_open();        // for some Windows environments
        exit(1);
    }

    // make std::min() and std::max() accessible on systems with antisocial macros:
#undef min
#undef max

// run-time checked narrowing cast (type conversion). See ???.
    template<class R, class A>
    CUDA_CALLABLE_MEMBER [[maybe_unused]] R narrow_cast(const A& a)
    {
        R r = R(a);
#ifdef __CUDACC__
        if (A(r) != a) printf("cuda warning: info loss\n");
#else
        if (A(r) != a) error(std::string("info loss"));
#endif
        return r;
    }

    inline int randint(int min, int max)
    {
        static std::random_device rd;
        static std::default_random_engine gen(rd());

        return std::uniform_int_distribution<>{ min, max }(gen);
    }

    [[maybe_unused]] inline int randint(int max)
    {
        return randint(0, max);
    }

    inline double rand_real(double min, double max)
    {
        static std::random_device rd;
        static std::default_random_engine gen(rd());

        return std::uniform_real_distribution<>{ min, max }(gen);
    }

    [[maybe_unused]] inline double rand_real()
    {
        return rand_real(0.0, 1.0);
    }

//inline double sqrt(int x) { return sqrt(double(x)); }	// to match C++0x

// container algorithms. See 21.9.   // C++ has better versions of this:

    template<typename C> using Value_type [[maybe_unused]] = typename C::value_type;

    template<typename C> using Iterator = typename C::iterator;

    template<typename C>
// requires Container<C>()
    [[maybe_unused]] void sort(C& c)
    {
        std::sort(c.begin(), c.end());
    }

    template<typename C, typename Pred>
// requires Container<C>() && Binary_Predicate<Value_type<C>>()
    [[maybe_unused]] void sort(C& c, Pred p)
    {
        std::sort(c.begin(), c.end(), p);
    }

    template<typename C, typename Val>
// requires Container<C>() && Equality_comparable<C,Val>()
    [[maybe_unused]] Iterator<C> find(C& c, Val v)
    {
        return std::find(c.begin(), c.end(), v);
    }

    template<typename C, typename Pred>
// requires Container<C>() && Predicate<Pred,Value_type<C>>()
    [[maybe_unused]] Iterator<C> find_if(C& c, Pred p)
    {
        return std::find_if(c.begin(), c.end(), p);
    }

    using Bytes = Vector<uint8_t>;

    template<class T, class ...Args>
    [[maybe_unused]] inline T open_file(const std::string& file_name,
            Args&& ... arg)
    {
        T fs{ file_name, std::forward<decltype(arg)>(arg)... };
        if (!fs) error(fmt::format("Cannot open file \"{}\"", file_name));
        return fs;
    }

    [[maybe_unused]]
    inline Bytes read_bytes(const std::string& file_name)
    {
        auto ifs = open_file<std::ifstream>(file_name, std::ios_base::binary);
        Bytes bytes;
        uint8_t byte;
        while (ifs.read(as_bytes(byte), sizeof(uint8_t)))
        {
            bytes.push_back(byte);
        }
        return bytes;
    }

    [[maybe_unused]] inline void write_bytes(const std::string& file_name,
            const Bytes& bytes)
    {
        auto ofs = open_file<std::ofstream>(file_name, std::ios_base::binary);
        for (const auto& byte: bytes)
        {
            ofs.write(as_bytes_const(byte), sizeof(uint8_t));
        }
    }

    [[maybe_unused]] inline std::string read_text(const std::string& file_name)
    {
        auto ifs = open_file<std::ifstream>(file_name);
        return std::string{ std::istreambuf_iterator<char>(ifs),
                            std::istreambuf_iterator<char>() };
    }

    [[maybe_unused]] inline void write_text(const std::string& file_name,
            const std::string& text)
    {
        auto ofs = open_file<std::ofstream>(file_name);
        ofs << text;
    }

    [[maybe_unused]] inline void print_byte(std::ostream& os, uint8_t b,
            int width = -1)
    {
        auto b_ = b;
        auto w = 0;
        while (b_)
        {
            w++;
            b_ >>= 1;
        }
        if (w == 0) w = 1;
        if (width > 0) w = width;
        for (auto i = w - 1; i >= 0; i--)
        {
            if (b & (1 << i)) os << '1';
            else os << '0';
        }
    }

    [[maybe_unused]] inline Vector<std::string> split(const std::string& s,
            const std::regex& delimiter)
    {
        return Vector<std::string>{
                std::sregex_token_iterator{ s.begin(), s.end(), delimiter, -1 },
                std::sregex_token_iterator() };
    }

    [[maybe_unused]] inline Vector<std::string> split(const std::string& s,
            const std::string& delimiter)
    {
        return split(s, std::regex{ delimiter });
    }

    template<class T, class ...Args>
    [[maybe_unused]] inline decltype(auto) func_timer(T&& func, Args&& ...
    args)
    {
        auto begin_time = std::chrono::high_resolution_clock::now();
        std::forward<decltype(func)>(func)(
                std::forward<decltype(args)>(args)...);
        auto end_time = std::chrono::high_resolution_clock::now();
        return static_cast<double>((end_time - begin_time)
                                   / std::chrono::milliseconds(1)) / 1000;
    }

    template<class T, class ...Args>
    [[maybe_unused]] inline decltype(auto) func_timer_res(T&& func,
            Args&& ... args)
    {
        auto begin_time = std::chrono::high_resolution_clock::now();
        auto res = std::forward<decltype(func)>(func)(
                std::forward<decltype(args)>(args)...);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = static_cast<double>((end_time - begin_time)
                                                  / std::chrono::milliseconds(
                1)) / 1000;
        return std::make_tuple(execution_time, res);
    }

    template<class T> using Vec2d = Vector<Vector<T>>;
    template<class T> using Vec3d = Vector<Vec2d<T>>;
    template<class T, std::size_t m, std::size_t n> using Arr2d = std::array<std::array<
            T,
            n>, m>;
    template<class T, std::size_t m, std::size_t n, std::size_t l> using Arr3d [[maybe_unused]] = Arr2d<
            std::array<T, l>,
            m,
            n>;

    template<class T>
    [[maybe_unused]] inline decltype(auto) make_2d(std::size_t m, std::size_t n,
            const T& val = T{})
    {
        return Vec2d<T>(m, Vector<T>(n, val));
    }

    template<class T>
    [[maybe_unused]] inline decltype(auto) idx_2d(const T& vec)
    {
        return std::make_tuple(vec.size(), vec[0].size());
    }

    template<class T>
    [[maybe_unused]] inline decltype(auto) make_3d(std::size_t m, std::size_t n,
            std::size_t l, const T& val = T{})
    {
        return Vec3d<T>(m, Vec2d<T>(n, Vector<T>(l, val)));
    }

    template<class T>
    [[maybe_unused]] inline decltype(auto) idx_3d(const T& vec)
    {
        return std::make_tuple(vec.size(), vec[0].size(), vec[0][0].size());
    }

    [[maybe_unused]] inline void utils_do_nothing()
    {
        ASSERT(true);
    }
}

namespace std
{
    template<>
    struct [[maybe_unused]] hash<utils::String>
    {
        size_t operator()(const utils::String& s) const
        {
            return hash<std::string>()(s);
        }
    };
}

#endif //CPP__STD_LIB_FACILITIES_H_
