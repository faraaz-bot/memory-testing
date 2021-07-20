
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

template <typename T>
int get_precedence(const T& x)
{
    return std::visit([](const auto a) { return a.precedence; }, x);
}

//
// Declarations
//

class Declaration
{
public:
    std::string name, type;
    bool        pointer;
    std::string size;
    Declaration(std::string name, std::string type, bool pointer = false, std::string size = "")
        : name(name)
        , type(type)
        , pointer(pointer)
        , size(size){};
    std::string render() const;
};

std::string Declaration::render() const
{
    std::string s;
    s = type;
    if(pointer)
        s += "*";
    s += " " + name;
    if(!size.empty())
        s += "[" + size + "]";
    s += ";";
    return s;
}

// class InlineDeclaration
// {
// public:
//     std::string name;
//     std::string type;
//     InlineDeclaration(std::string name, std::string type)
//         : name(name)
//         , type(type){};
//     std::string render() const;
// };

// std::string InlineDeclaration::render() const
// {
//     return type + " " + name;
// }

//
// Expressions
//

struct ScalarVariable;
class Variable;
class Literal;
class ComplexLiteral;

class Add;
class Subtract;
class Multiply;
class Divide;
class Modulus;

class And;
class Less;

using Expression = std::variant<ScalarVariable,
                                Variable,
                                Literal,
                                ComplexLiteral,
                                Add,
                                Subtract,
                                Multiply,
                                Divide,
                                Modulus,
                                And,
                                Less>;

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
    int const precedence = 100;

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

class ComplexLiteral
{
    std::string xvalue, yvalue;

public:
    int const precedence = 100;

    template <typename T>
    ComplexLiteral(T l, T r)
    {
        xvalue = std::to_string(l);
        yvalue = std::to_string(r);
    }

    ComplexLiteral(std::string l, std::string r)
    {
        xvalue = l;
        yvalue = r;
    }

    std::string render() const
    {
        return "{" + xvalue + ", " + yvalue + "}";
    }
};

struct ScalarVariable
{
    int const   precedence = 100;
    std::string name, type;
    ScalarVariable(std::string name, std::string type)
        : name(name)
        , type(type){};
    std::string render() const;
};

class Variable
{
public:
    int const          precedence = 100;
    std::string        name, type;
    bool               pointer, restrict;
    ScalarVariable     x, y;
    OptionalExpression index;
    OptionalExpression size;

    Variable(std::string _name,
             std::string _type,
             bool        pointer  = false,
             bool        restrict = false,
             int         size     = 0);

    Variable(const ScalarVariable& v)
        : name(v.name)
        , type(v.type)
        , x(v.name + ".x", v.type)
        , y(v.name + ".y", v.type){};

    Variable(const Variable& v);

    Variable       operator[](const Expression& index) const;
    Declaration    declaration() const;
    ScalarVariable address() const;

    std::string render() const;
};

#define MAKE_ARITH(NAME, SEP, PRECEDENCE)                \
    class NAME                                           \
    {                                                    \
        std::string separator{SEP};                      \
                                                         \
    public:                                              \
        int const               precedence = PRECEDENCE; \
        std::vector<Expression> args;                    \
        NAME(std::initializer_list<Expression> il)       \
            : args(il){};                                \
        NAME(std::vector<Expression> il)                 \
            : args(il){};                                \
        std::string render() const;                      \
    };

#define MAKE_ARITH_METHODS(NAME)                 \
    std::string NAME::render() const             \
    {                                            \
        std::string s;                           \
        if(get_precedence(args[0]) < precedence) \
            s += "(" + vrender(args[0]) + ")";   \
        else                                     \
            s += vrender(args[0]);               \
        s += separator;                          \
        if(get_precedence(args[1]) < precedence) \
            s += "(" + vrender(args[1]) + ")";   \
        else                                     \
            s += vrender(args[1]);               \
        return s;                                \
    }

MAKE_ARITH(Add, " + ", 50);
MAKE_ARITH(Multiply, " * ", 100);
MAKE_ARITH(Subtract, " - ", 50);
MAKE_ARITH(Divide, " / ", 100);
MAKE_ARITH(Modulus, " % ", 100);

MAKE_ARITH(And, " && ", 100);
MAKE_ARITH(Less, " < ", 100);

MAKE_ARITH_METHODS(Add);
MAKE_ARITH_METHODS(Multiply);
MAKE_ARITH_METHODS(Subtract);
MAKE_ARITH_METHODS(Divide);
MAKE_ARITH_METHODS(Modulus);

MAKE_ARITH_METHODS(And);
MAKE_ARITH_METHODS(Less);

std::string ScalarVariable::render() const
{
    return name;
}

Variable::Variable(std::string _name, std::string _type, bool pointer, bool restrict, int size)
    : name(_name)
    , type(_type)
    , pointer(pointer)
    , restrict(restrict)
    , x(_name + ".x", _type)
    , y(_name + ".y", _type)

{
    if(size > 0)
        this->size = Expression{size};
}

Variable::Variable(const Variable& v)
    : name(v.name)
    , type(v.type)
    , x(v.name + ".x", v.type)
    , y(v.name + ".y", v.type)
{
    index = v.index;
    size  = v.size;

    if(index)
    {
        x.name = v.name + "[" + vrender(*index) + "].x";
        y.name = v.name + "[" + vrender(*index) + "].y";
    }
}

Declaration Variable::declaration() const
{
    if(size)
        return Declaration(name, type, pointer, vrender(*size));
    return Declaration(name, type, pointer);
}

ScalarVariable Variable::address() const
{
    if(index)
    {
        return ScalarVariable("&" + name + "[" + vrender(*index) + "]", type + "*");
    }
    return ScalarVariable("&" + name, type + "*");
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
    auto v  = Variable(name, type);
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

Less operator<(const Expression& a, const Expression& b)
{
    return Less{a, b};
}

And operator&&(const Expression& a, const Expression& b)
{
    return And{a, b};
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
class If;
class StatementList;

using Statement = std::variant<StatementList, Declaration, Assign, Call, For, If>;

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
    std::string           render_decl() const;
                          operator bool() const;
    void                  append(Variable);
};

using TemplateList = ArgumentList;

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

std::string ArgumentList::render_decl() const
{
    std::string f;
    if(!arguments.empty())
    {
        f = arguments[0].type;
        if(arguments[0].pointer)
            f += "*";
        f += " " + arguments[0].name;
        if(arguments[0].size)
            f += "[]";
        for(uint i = 1; i < arguments.size(); ++i)
        {
            f += ",";
            f += arguments[i].type;
            if(arguments[i].pointer)
                f += "*";
            f += " " + arguments[i].name;
            if(arguments[i].size)
                f += "[]";
        }
    }
    return f;
}

ArgumentList::operator bool() const
{
    return !arguments.empty();
}

void ArgumentList::append(Variable v)
{
    arguments.push_back(v);
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
    For(Variable initial, Expression condition, Expression iteration, StatementList body)
        : initial(initial)
        , condition(condition)
        , iteration(iteration)
        , body(body){};
    std::string render() const;
};

class If
{
public:
    Expression    condition;
    StatementList body;
    If(Expression condition, StatementList body)
        : condition(condition)
        , body(body){};
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

void operator+=(StatementList& stmts, const StatementList& s)
{
    //    stmts.statements.insert(stmts.statements.end(), s.statements.cbegin(), s.statements.cend());
    for(auto x : s.statements)
    {
        stmts += x;
    }
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

std::string If::render() const
{
    std::string s;
    s += "if(";
    s += vrender(condition);
    s += ") {\n";
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
    TemplateList  templates;

    Function(std::string name)
        : name(name){};

    std::string render() const;
};

std::string Function::render() const
{
    std::string f;
    if(templates)
    {
        f += "template<" + templates.render_decl() + ">";
    }
    f += "void " + name;
    f += "(" + arguments.render_decl() + ") {\n";
    f += body.render();
    f += "}\n";
    return f;
}

//
// Re-write helpers
//

#define MAKE_ARITH_VISITOR(NAME)                  \
    Expression operator()(const NAME& v)  \
    {                                             \
        std::vector<Expression> args;             \
        for(auto a : v.args)                      \
        {                                         \
            args.push_back(std::visit(*this, a)); \
        }                                         \
        return Expression{NAME{args}};            \
    }

#define MAKE_VISITOR(RET, CLS)       \
    RET operator()(const CLS& x)     \
    {                                \
        return RET(visit(*this, x)); \
    }

template <class Visitor>
Expression visit(Visitor&& vis, const ScalarVariable& x)
{
    return x;
}

template <class Visitor>
Expression visit(Visitor&& vis, const Variable& x) {
    // XXX
    return x;
}

template <class Visitor>
Expression visit(Visitor&& vis, const Literal& x)
{
    return x;
}

template <class Visitor>
Expression visit(Visitor&& vis, const ComplexLiteral& x)
{
    return x;
}

template <class Visitor>
Statement visit(Visitor&& vis, const For& x)
{
    auto initial   = std::get<Variable>(vis(x.initial));
    auto condition = std::visit(vis, x.condition);
    auto iteration = std::visit(vis, x.iteration);
    auto body      = StatementList();
    for(auto s : x.body.statements)
    {
        body += std::visit(vis, s);
    }
    return For(initial, condition, iteration, body);
}

template <class Visitor>
Statement visit(Visitor&& vis, const If& x)
{
    auto condition = std::visit(vis, x.condition);
    auto body      = StatementList();
    for(auto s : x.body.statements)
    {
        body += std::visit(vis, s);
    }
    return If(condition, body);
}

template <class Visitor>
Statement visit(Visitor&& vis, const Declaration& x)
{
    return x;
}

template <class Visitor>
Statement visit(Visitor&& vis, const Call& x)
{
    auto y = Call(x);
    // XXX arguments
    return y;
}

template <class Visitor>
Statement visit(Visitor&& vis, const StatementList& x)
{
    // XXX
    return x;
}

struct MakePlanarVisitor
{
    std::string varname, rename, imname;

    MakePlanarVisitor(std::string varname)
        : varname(varname)
    {
        rename = varname + "re";
        imname = varname + "im";
    }

    MAKE_VISITOR(Expression, ScalarVariable)
    MAKE_VISITOR(Expression, Variable)
    MAKE_VISITOR(Expression, Literal)
    MAKE_VISITOR(Expression, ComplexLiteral)

    MAKE_VISITOR(Statement, For)
    MAKE_VISITOR(Statement, If)
    MAKE_VISITOR(Statement, Declaration)
    MAKE_VISITOR(Statement, Call)
    MAKE_VISITOR(Statement, StatementList)

    MAKE_ARITH_VISITOR(Add)
    MAKE_ARITH_VISITOR(Subtract)
    MAKE_ARITH_VISITOR(Multiply)
    MAKE_ARITH_VISITOR(Divide)
    MAKE_ARITH_VISITOR(Modulus)

    MAKE_ARITH_VISITOR(And)
    MAKE_ARITH_VISITOR(Less)

    ArgumentList operator()(const ArgumentList& args)
    {
        ArgumentList nargs;
        for(auto x : args.arguments)
        {
            if(x.name == varname)
            {
                auto re = Variable{x};
                re.name = rename;
                re.type = "real_type_t<" + x.type + ">";
                auto im = Variable{x};
                im.name = imname;
                im.type = "real_type_t<" + x.type + ">";
                nargs.append(re);
                nargs.append(im);
            }
            else
            {
                nargs.append(x);
            }
        }
        return nargs;
    }

    Statement operator()(const Assign& x)
    {
        if(x.lhs.name == varname && std::holds_alternative<Variable>(x.rhs))
        {
            // on lhs, lhs needs to be split; use .x and .y on rhs

            auto rhs   = std::get<Variable>(x.rhs);
            auto stmts = StatementList();

            auto re = Variable(x.lhs);
            re.name = rename;
            auto im = Variable(x.lhs);
            im.name = imname;

            stmts += Assign(re, rhs.x);
            stmts += Assign(im, rhs.y);
            return Statement(stmts);
        }
        else if(std::holds_alternative<Variable>(x.rhs)
                && std::get<Variable>(x.rhs).name == varname)
        {
            // on rhs, rhs needs to be joined as a complex literal

            auto rhs = std::get<Variable>(x.rhs);
            auto re  = Variable(rhs);
            re.name  = rename;
            auto im  = Variable(rhs);
            im.name  = imname;
            return Statement(Assign(x.lhs, ComplexLiteral(re.render(), im.render())));
        }

        return Statement(x);
    }

};

Function make_planar(const Function& f)
{

    auto visitor = MakePlanarVisitor("R");
    auto g       = Function(f.name);
    g.arguments  = visitor(f.arguments);
    for(auto s : f.body.statements)
    {
        g.body += std::visit(visitor, s);
    }
    return g;
}
