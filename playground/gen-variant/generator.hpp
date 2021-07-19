
#include <algorithm>
#include <any>
#include <iostream>
#include <numeric>
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
// Declarations
//

class Declaration
{
public:
    std::string name;
    std::string type;
    Declaration(std::string name, std::string type)
        : name(name)
        , type(type){};
    std::string render() const;
};

std::string Declaration::render() const
{
    return type + " " + name + ";";
}

//
// Expressions
//

struct ScalarVariable;
class Variable;
class Literal;

class Add;
class Subtract;
class Multiply;
class Divide;
class Modulus;

using Expression
    = std::variant<ScalarVariable, Variable, Literal, Add, Subtract, Multiply, Divide, Modulus>;

class OptionalExpression
{
    std::any expr;

public:
    OptionalExpression(){};
    OptionalExpression(const Expression& expr);
    Expression operator*() const;
               operator bool() const;
};

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

struct ScalarVariable
{
    std::string name;
    ScalarVariable(std::string name)
        : name(name){};
    std::string render() const;
};

class Variable
{
public:
    std::string        name;
    ScalarVariable     x, y;
    OptionalExpression index;

    Variable(std::string _name)
        : name(_name)
        , x(_name + ".x")
        , y(_name + ".y"){};

    Variable(ScalarVariable v)
        : name(v.name)
        , x(v.name + ".x")
        , y(v.name + ".y"){};

    Variable       operator[](const Expression& index) const;
    Declaration    declaration() const;
    ScalarVariable address() const;

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
        for(uint i = 1; i < args.size(); ++i)  \
            s += separator + vrender(args[i]); \
        return s;                              \
    }

MAKE_ARITH(Add, " + ", 10);
MAKE_ARITH(Multiply, " * ", 20);
MAKE_ARITH(Subtract, " - ", 15);
MAKE_ARITH(Divide, " / ", 25);
MAKE_ARITH(Modulus, " % ", 25);

MAKE_RENDER(Add);
MAKE_RENDER(Multiply);
MAKE_RENDER(Subtract);
MAKE_RENDER(Divide);
MAKE_RENDER(Modulus);

std::string ScalarVariable::render() const
{
    return name;
}

Declaration Variable::declaration() const
{
    return Declaration(name, "int");
}

ScalarVariable Variable::address() const
{
    if(index)
    {
        return ScalarVariable("&" + name + "[" + vrender(*index) + "]");
    }
    return ScalarVariable("&" + name);
}

std::string Variable::render() const
{
    if(index)
    {
        return name + "[" + vrender(*index) + "]";
    }
    return name;
}

Variable Variable::operator[](const Expression& index) const
{
    auto v = Variable(name);
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

Modulus operator%(const Expression& a, const Expression& b)
{
    return Modulus{a, b};
}

OptionalExpression::operator bool() const
{
    return expr.has_value();
}

Expression OptionalExpression::operator*() const
{
    return std::any_cast<Expression>(expr);
}

OptionalExpression::OptionalExpression(const Expression& expr)
{
    this->expr = expr;
}

//
// Statements
//

class Assign;
class Call;
class For;
class StatementList;

using Statement = std::variant<Declaration, Assign, Call, For>;
//using Statement = std::variant<Assign>;

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

class ArgumentList
{
public:
    ArgumentList(){};
    ArgumentList(std::vector<Variable> arguments)
        : arguments(arguments){};
    std::vector<Variable> arguments;
    std::string           render() const;
};

std::string ArgumentList::render() const
{
    std::string f;
    if(!arguments.empty())
    {
        f = arguments[0].render();
        for(uint i = 1; i < arguments.size(); ++i)
        {
            f += ",";
            f += arguments[i].render();
        }
    }
    return f;
}

class Call
{
public:
    std::string  name;
    ArgumentList arguments;

    Call(std::string name, ArgumentList arguments)
        : name(name)
        , arguments(arguments){};

    std::string render() const;
};

std::string Call::render() const
{
    std::string f;
    f += name + "(" + arguments.render() + ");";
    return f;
}

class StatementList
{
public:
    std::vector<Statement> statements;
    StatementList(){};
    std::string render() const;
};

class For
{
public:
    Variable      initial;
    Expression    condition;
    Expression    iteration;
    StatementList body;
    For(Variable initial, Expression condition, Expression iteration)
        : initial(initial)
        , condition(condition)
        , iteration(iteration){};
    std::string render() const;
};

std::string StatementList::render() const
{
    std::string r;
    for(auto s : statements)
        r += vrender(s) + "\n";
    return r;
}

void operator+=(StatementList& stmts, const Statement& s)
{
    stmts.statements.push_back(s);
}

std::string For::render() const
{
    std::string s;
    s += "for(";
    s += initial.render() + "; ";
    s += vrender(condition) + "; ";
    s += vrender(iteration) + ") {\n ";
    s += body.render();
    s += "\n}\n";
    return s;
}

//
// Functions
//

class Function
{
public:
    std::string   name;
    StatementList body;
    ArgumentList  arguments;

    Function(std::string name)
        : name(name){};

    std::string render() const;
};

std::string Function::render() const
{
    std::string f;
    f = "void " + name + "(" + arguments.render() + ") {\n";
    f += body.render();
    f += "}\n";
    return f;
}

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

    Expression operator()(const ScalarVariable& x)
    {
        return Expression{x};
    }

    Expression operator()(const Variable& x)
    {
        auto y = Variable{x};
        if(x.name == old_name)
        {
            y.name = new_name;
        }
        if(y.index)
        {
            y.index = std::visit(*this, *y.index);
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
    MAKE_ARITH_VISITOR(Modulus)

    Statement operator()(const Assign& x)
    {
        auto lhs = std::get<Variable>((*this)(x.lhs));
        auto rhs = std::visit(*this, x.rhs);
        return Statement{Assign(lhs, rhs)};
    }

    Statement operator()(const For& x)
    {
        auto initial   = std::get<Variable>((*this)(x.initial));
        auto condition = std::visit(*this, x.condition);
        auto iteration = std::visit(*this, x.iteration);
        return Statement{For(initial, condition, iteration)};
    }

    Statement operator()(const Declaration& x)
    {
        return Statement{x};
    }

    Statement operator()(const Call& x)
    {
        return Statement{x};
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
