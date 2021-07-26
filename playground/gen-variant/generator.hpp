
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
    int const precedence = 0;

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
    int const precedence = 0;

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
    int const   precedence = 0;
    std::string name, type;
    ScalarVariable(std::string name, std::string type)
        : name(name)
        , type(type){};
    std::string render() const;
};

class Variable
{
public:
    int const          precedence = 0;
    std::string        name, type;
    bool               pointer, restrict;
    ScalarVariable     x, y;
    OptionalExpression index;
    OptionalExpression size;

    Variable(std::string _name,
             std::string _type,
             bool        pointer = false,
             bool restrict       = false,
             int size            = 0);

    Variable(const ScalarVariable& v)
        : name(v.name)
        , type(v.type)
        , x(v.name + ".x", v.type)
        , y(v.name + ".y", v.type){};

    Variable(const Variable& v);
    Variable(const Variable& v, const Expression& index);

    Variable       operator[](const Expression& index) const;
    ScalarVariable address() const;

    std::string render() const;
};

#define MAKE_BINARY(NAME, SEP, PRECEDENCE)               \
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

#define MAKE_BINARY_METHODS(NAME)                \
    std::string NAME::render() const             \
    {                                            \
        std::string s;                           \
        if(get_precedence(args[0]) > precedence) \
            s += "(" + vrender(args[0]) + ")";   \
        else                                     \
            s += vrender(args[0]);               \
        s += separator;                          \
        if(get_precedence(args[1]) > precedence) \
            s += "(" + vrender(args[1]) + ")";   \
        else                                     \
            s += vrender(args[1]);               \
        return s;                                \
    }

MAKE_BINARY(Add, " + ", 6);
MAKE_BINARY(Subtract, " - ", 6);
MAKE_BINARY(Multiply, " * ", 5);
MAKE_BINARY(Divide, " / ", 5);
MAKE_BINARY(Modulus, " % ", 5);

MAKE_BINARY(Less, " < ", 9);
MAKE_BINARY(And, " && ", 14);

MAKE_BINARY_METHODS(Add);
MAKE_BINARY_METHODS(Subtract);
MAKE_BINARY_METHODS(Multiply);
MAKE_BINARY_METHODS(Divide);
MAKE_BINARY_METHODS(Modulus);

MAKE_BINARY_METHODS(Less);
MAKE_BINARY_METHODS(And);

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
    , pointer(v.pointer)
    , restrict(v.restrict)
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

Variable::Variable(const Variable& v, const Expression& _index)
    : name(v.name)
    , type(v.type)
    , pointer(v.pointer)
    , restrict(v.restrict)
    , x(v.name, v.type)
    , y(v.name, v.type)
    , index(_index)
{
    size   = v.size;
    x.name = v.name + "[" + vrender(*index) + "].x";
    y.name = v.name + "[" + vrender(*index) + "].y";
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
    return Variable(*this, index);
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
class Declaration;
class For;
class If;
class StatementList;

struct LineBreak
{
    std::string render() const
    {
        return "\n\n";
    }
};

struct SyncThreads
{
    std::string render() const
    {
        return "__syncthreads();\n";
    }
};

struct Return
{
    std::string render() const
    {
        return "return;\n";
    }
};

struct CommentLines
{
    std::vector<std::string> comments;
    std::string              render() const
    {
        std::string s;
        for(auto c : comments)
        {
            s += "// " + c + "\n";
        }
        return s;
    }
    CommentLines(std::initializer_list<std::string> il)
        : comments(il){};
};

using Statement = std::variant<Assign,
                               Call,
                               CommentLines,
                               Declaration,
                               For,
                               If,
                               LineBreak,
                               Return,
                               SyncThreads,
                               StatementList>;

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

//
// Declarations
//

class Declaration
{
public:
    Variable                  var;
    std::optional<Expression> value;
    Declaration(Variable v)
        : var(v){};
    Declaration(Variable v, Expression val)
        : var(v)
        , value(val){};
    std::string render() const;
};

std::string Declaration::render() const
{
    std::string s;
    s = var.type;
    if(var.pointer)
        s += "*";
    s += " " + var.name;
    if(var.size)
        s += "[" + vrender(*var.size) + "]";
    if(value)
        s += " = " + vrender(*value);
    s += ";";
    return s;
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
    Variable      var;
    Expression    initial;
    Expression    condition;
    Expression    increment;
    StatementList body;
    For(Variable      var,
        Expression    initial,
        Expression    condition,
        Expression    increment,
        StatementList body = {})
        : var(var)
        , initial(initial)
        , condition(condition)
        , increment(increment)
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
    s += var.type + " " + var.name + " = ";
    s += vrender(initial) + "; ";
    s += vrender(condition) + "; ";

    s += var.name + " += " + vrender(increment);
    s += ") {\n ";
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

#define MAKE_VISITOR(RET, CLS)       \
    RET operator()(const CLS& x)     \
    {                                \
        return RET(visit(*this, x)); \
    }

#define MAKE_TRIVIAL_VISIT(RET, CLS)       \
    template <class Visitor>               \
    RET visit(Visitor&& vis, const CLS& x) \
    {                                      \
        return x;                          \
    }

MAKE_TRIVIAL_VISIT(Expression, Add)
MAKE_TRIVIAL_VISIT(Expression, And)
MAKE_TRIVIAL_VISIT(Expression, ComplexLiteral)
MAKE_TRIVIAL_VISIT(Expression, Divide)
MAKE_TRIVIAL_VISIT(Expression, Less)
MAKE_TRIVIAL_VISIT(Expression, Literal)
MAKE_TRIVIAL_VISIT(Expression, Modulus)
MAKE_TRIVIAL_VISIT(Expression, Multiply)
MAKE_TRIVIAL_VISIT(Expression, ScalarVariable)
MAKE_TRIVIAL_VISIT(Expression, Subtract)

MAKE_TRIVIAL_VISIT(Statement, CommentLines)
MAKE_TRIVIAL_VISIT(Statement, Declaration)
MAKE_TRIVIAL_VISIT(Statement, LineBreak)
MAKE_TRIVIAL_VISIT(Statement, Return)
MAKE_TRIVIAL_VISIT(Statement, SyncThreads)

template <class Visitor>
Expression visit(Visitor&& vis, const Variable& x)
{
    auto y = Variable(x);
    // y.x = std::get<ScalarVariable>(vis(y.x));
    // y.y = std::get<ScalarVariable>(vis(y.y));
    // if (y.index) y.index = vis(*y.index);
    return y;
}

template <class Visitor>
Statement visit(Visitor&& vis, const StatementList& x)
{
    auto y = StatementList();
    for(auto s : x.statements)
    {
        y += std::visit(vis, s);
    }
    return y;
}

template <class Visitor>
ArgumentList visit(Visitor&& vis, const ArgumentList& x)
{
    auto y = ArgumentList();
    for(auto s : x.arguments)
    {
        y.append(std::get<Variable>(vis(s)));
    }
    return y;
}

template <class Visitor>
Statement visit(Visitor&& vis, const Call& x)
{
    auto y      = Call(x);
    y.arguments = visit(vis, x.arguments);
    return y;
}

template <class Visitor>
Statement visit(Visitor&& vis, const For& x)
{
    auto var       = std::get<Variable>(vis(x.var));
    auto initial   = std::visit(vis, x.initial);
    auto condition = std::visit(vis, x.condition);
    auto increment = std::visit(vis, x.increment);
    auto body      = std::get<StatementList>(visit(vis, x.body));
    return For(var, initial, condition, increment, body);
}

template <class Visitor>
Statement visit(Visitor&& vis, const If& x)
{
    auto condition = std::visit(vis, x.condition);
    auto body      = std::get<StatementList>(visit(vis, x.body));
    return If(condition, body);
}

template <class Visitor>
Function visit(Visitor&& vis, const Function& x)
{
    auto y      = Function(x.name);
    y.arguments = visit(vis, x.arguments);
    y.body      = std::get<StatementList>(visit(vis, x.body));
    return y;
}

//
// Make planar
//

struct MakePlanarVisitor
{
    std::string varname, rename, imname;

    MakePlanarVisitor(std::string varname)
        : varname(varname)
    {
        rename = varname + "re";
        imname = varname + "im";
    }

    MAKE_VISITOR(Expression, Add)
    MAKE_VISITOR(Expression, And)
    MAKE_VISITOR(Expression, ComplexLiteral)
    MAKE_VISITOR(Expression, Divide)
    MAKE_VISITOR(Expression, Less)
    MAKE_VISITOR(Expression, Literal)
    MAKE_VISITOR(Expression, Modulus)
    MAKE_VISITOR(Expression, Multiply)
    MAKE_VISITOR(Expression, ScalarVariable)
    MAKE_VISITOR(Expression, Subtract)
    MAKE_VISITOR(Expression, Variable)

    MAKE_VISITOR(Statement, Call)
    MAKE_VISITOR(Statement, CommentLines)
    MAKE_VISITOR(Statement, Declaration)
    MAKE_VISITOR(Statement, For)
    MAKE_VISITOR(Statement, If)
    MAKE_VISITOR(Statement, LineBreak)
    MAKE_VISITOR(Statement, Return)
    MAKE_VISITOR(Statement, StatementList)
    MAKE_VISITOR(Statement, SyncThreads)

    ArgumentList operator()(const ArgumentList& x)
    {
        ArgumentList y;
        for(auto a : x.arguments)
        {
            if(a.name == varname)
            {
                auto re = Variable(a);
                re.name = rename;
                re.type = "real_type_t<" + a.type + ">";
                auto im = Variable(a);
                im.name = imname;
                im.type = "real_type_t<" + a.type + ">";
                y.append(re);
                y.append(im);
            }
            else
            {
                y.append(a);
            }
        }
        return y;
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
            return stmts;
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
            return Assign(x.lhs, ComplexLiteral(re.render(), im.render()));
        }

        return x;
    }

    Function operator()(const Function& x)
    {
        auto y      = Function(x.name);
        y.arguments = (*this)(x.arguments);
        y.body      = std::get<StatementList>(visit(*this, x.body));
        return y;
    }
};

Function make_planar(const Function& f, std::string varname)
{
    auto visitor = MakePlanarVisitor(varname);
    return visitor(f);
}
