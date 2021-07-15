
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

using Expression = std::variant<Variable, Literal, Add, Subtract, Multiply, Divide>;

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

#define MAKE_ARITH(NAME, SEP, PRECEDENCE)                \
    class NAME                                           \
    {                                                    \
        int const               precedence = PRECEDENCE; \
        std::string             separator{SEP};          \
        std::vector<Expression> args;                    \
                                                         \
    public:                                              \
        NAME(std::initializer_list<Expression> il)       \
            : args(il){};                                \
        std::string render() const;                      \
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

class Assign
{
    Variable   lhs;
    Expression rhs;

public:
    Assign(Variable lhs, Expression rhs)
        : lhs(lhs)
        , rhs(rhs){};
    std::string render() const
    {
        return lhs.render() + " = " + vrender(rhs) + ";";
    }
};

using Statement = std::variant<Assign>;

class StatementList
{
    std::vector<Statement> statments;

public:
    StatementList(){};
    std::string render() const
    {
        std::string r;
        for(auto s : statments)
            r += vrender(s);
        return r;
    }
    void operator+=(Statement s)
    {
        statments.push_back(s);
    }
};

Add operator+(const Expression& a, const Expression& b)
{
    return Add{a, b};
}

Subtract operator-(const Expression& a, const Expression& b)
{
    return Subtract{a, b};
}

Multiply operator*(const Expression& a, const Expression& b)
{
    return Multiply{a, b};
}

Divide operator/(const Expression& a, const Expression& b)
{
    return Divide{a, b};
}

void test()
{
    Variable x("x"), y("y");
    auto     z = x * y + y;
    auto     w = z + y;

    auto stmts = StatementList();
    stmts += Assign(x, w);

    std::cout << stmts.render() << std::endl;
}

int main(int argc, char* argv[])
{
    test();
}
