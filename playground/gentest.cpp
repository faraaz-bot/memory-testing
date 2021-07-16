
#include <any>
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
    // Expression index;
    std::any index;

    Variable(std::string name)
        : name(name){};

    Variable operator[](const Expression& index) const;

    std::string render() const;
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

std::string Variable::render() const
{
    if(index.has_value())
    {
        auto expr = std::any_cast<Expression>(index);
        return name + "[" + vrender(expr) + "]";
    }
    return name;
}

Variable Variable::operator[](const Expression& index) const
{
    auto v  = Variable(name);
    v.index = index;
    return v;
}

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
            r += vrender(s) + "\n";
        return r;
    }
    void operator+=(Statement s)
    {
        statements.push_back(s);
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
    std::string old_name{"x"};
    std::string new_name{"X"};

    Expression operator()(const Variable& x)
    {
        auto y = Variable{x};
        if(x.name == old_name)
        {
            y.name = new_name;
        }
        if(y.index.has_value())
        {
            y.index = std::visit(*this, std::any_cast<Expression>(y.index));
        }
        return Expression{y};
    }

    Expression operator()(const Literal& x)
    {
        return Expression{x};
    }

    MAKE_ARITH_VISITOR(Add)
    MAKE_ARITH_VISITOR(Subtract)
    MAKE_ARITH_VISITOR(Multiply)
    MAKE_ARITH_VISITOR(Divide)

    Statement operator()(const Assign& x)
    {
        auto lhs = std::get<Variable>((*this)(x.lhs));
        auto rhs = std::visit(*this, x.rhs);
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
    auto     z = x * y;

    auto stmts = StatementList();
    for(int w = 0; w < 4; ++w)
    {
        stmts += Assign(x, y[z + w]);
    }
    stmts += Assign(x, y + 1);

    // copying is trivial
    auto o = StatementList{stmts};
    o += Assign(x, 22);

    auto r = make_planar(stmts);

    // original, with extra assign
    std::cout << stmts.render() << std::endl;

    // original, with extra assign
    std::cout << o.render() << std::endl;

    // transformed
    std::cout << r.render() << std::endl;
}

int main(int argc, char* argv[])
{
    test();
}
