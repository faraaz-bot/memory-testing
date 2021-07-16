
#include <iostream>
#include <string>
#include <variant>
#include <vector>

//
// Helpers
//

template <typename T>
std::string vrender(const T& x)
{
    return std::visit([](const auto a) { return a.render(); }, x);
}

//
// Expressions
//

class Variable;
class Literal;

class Add;
class Subtract;
class Multiply;
class Divide;

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
public:
    std::string name;

    Variable(std::string name)
        : name(name)
    {
    }

    std::string render() const
    {
        return name;
    }
};

#define MAKE_ARITH(NAME, SEP, PRECEDENCE)          \
    class NAME                                     \
    {                                              \
        int const   precedence = PRECEDENCE;       \
        std::string separator{SEP};                \
                                                   \
    public:                                        \
        std::vector<Expression> args;              \
        NAME(std::initializer_list<Expression> il) \
            : args(il){};                          \
        NAME(std::vector<Expression> il)           \
            : args(il){};                          \
        std::string render() const;                \
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

//
// Statements
//

class Assign
{
public:
    Variable   lhs;
    Expression rhs;

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
public:
    std::vector<Statement> statements;
    StatementList(){};
    std::string render() const
    {
        std::string r;
        for(auto s : statements)
            r += vrender(s);
        return r;
    }
    void operator+=(Statement s)
    {
        statements.push_back(s);
    }

    std::vector<Statement> get_args() const
    {
        return statements;
    }
};

//
// Example of AST transform
//

//
// make_planar
//

#define MAKE_ARITH_VISITOR(NAME)                  \
    Expression operator()(const NAME& v)          \
    {                                             \
        std::vector<Expression> args;             \
        for(auto a : v.args)                      \
        {                                         \
            args.push_back(std::visit(*this, a)); \
        }                                         \
        return Expression{NAME{args}};            \
    }

struct MakePlanarVisitor
{

    Expression operator()(const Variable& v)
    {
        if(v.name == "x")
            return Expression{Variable{"X"}};
        return Expression{v};
    }

    Expression operator()(const Literal& v)
    {
        return Expression{v};
    }

    MAKE_ARITH_VISITOR(Add)
    MAKE_ARITH_VISITOR(Subtract)
    MAKE_ARITH_VISITOR(Multiply)
    MAKE_ARITH_VISITOR(Divide)

    Statement operator()(const Assign& a)
    {
        auto lhs = std::get<Variable>((*this)(a.lhs));
        auto rhs = std::visit(*this, a.rhs);
        return Statement{Assign(lhs, rhs)};
    }
};

StatementList make_planar(const StatementList& stmts)
{
    auto visitor = MakePlanarVisitor();
    auto nstmts  = StatementList();
    for(auto s : stmts.statements)
    {
        nstmts += std::visit(visitor, s);
    }
    return nstmts;
}

//
// Test!
//

void test()
{
    Variable x("x"), y("y");
    auto     z = x * y + y;
    auto     w = z + y;

    auto stmts = StatementList();
    stmts += Assign(x, y + 1);

    auto r = make_planar(stmts);

    std::cout << r.render() << std::endl;
}

int main(int argc, char* argv[])
{
    test();
}
