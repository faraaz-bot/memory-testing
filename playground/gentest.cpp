
#include <iostream>
#include <string>
#include <variant>
#include <vector>

class Variable;
class Literal;

class Add;
class Subtract;
class Multiply;
class Divide;

template <typename T>
std::string vrender(const T& x)
{
    return std::visit([](const auto a) { return a.render(); }, x);
}

using arithmetic_operand = std::variant<Variable, Literal, Add, Subtract, Multiply, Divide>;

class Literal
{
    std::string value;

public:
    template <typename T>
    Literal(T l)
    {
        value = std::to_string(l);
    }

    std::string render() const
    {
        return value;
    }
};

class Variable
{
    std::string name;

public:
    Variable(std::string name)
        : name(name)
    {
    }

    std::string render() const
    {
        return name;
    }
};

#define MAKE_ARITH(NAME, SEP, PRECEDENCE)                        \
    class NAME                                                   \
    {                                                            \
        int const                       precedence = PRECEDENCE; \
        std::string                     separator{SEP};          \
        std::vector<arithmetic_operand> args;                    \
                                                                 \
    public:                                                      \
        NAME(std::initializer_list<arithmetic_operand> il)       \
            : args(il){};                                        \
        std::string render() const;                              \
    };

#define MAKE_RENDER(NAME)                      \
    std::string NAME::render() const           \
    {                                          \
        std::string s = vrender(args[0]);      \
        for(int i = 1; i < args.size(); ++i)   \
            s += separator + vrender(args[i]); \
        return s;                              \
    }

MAKE_ARITH(Add, " + ", 10);
MAKE_ARITH(Multiply, " * ", 20);
MAKE_ARITH(Subtract, " - ", 15);
MAKE_ARITH(Divide, " / ", 25);

MAKE_RENDER(Add);
MAKE_RENDER(Multiply);
MAKE_RENDER(Subtract);
MAKE_RENDER(Divide);

Add operator+(const arithmetic_operand& a, const arithmetic_operand& b)
{
    return Add{a, b};
}

Subtract operator-(const arithmetic_operand& a, const arithmetic_operand& b)
{
    return Subtract{a, b};
}

Multiply operator*(const arithmetic_operand& a, const arithmetic_operand& b)
{
    return Multiply{a, b};
}

Divide operator/(const arithmetic_operand& a, const arithmetic_operand& b)
{
    return Divide{a, b};
}

void test()
{
    Variable x("x"), y("y");
    auto     z = x * y;
    auto     w = z + y;

    std::cout << w.render() << std::endl;
}

int main(int argc, char* argv[])
{
    test();
}
