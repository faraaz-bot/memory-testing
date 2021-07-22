//
// Simple AST based code generator.
//

#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace gen
{

    //
    // Base classes
    //

    struct Node
    {
        int precedence = 100;

        virtual std::string render() const = 0;
        virtual ~Node()                    = default;
    };

    //
    //  Simple keywords etc
    //

    struct LineBreak : Node
    {
        std::string render() const override
        {
            return "\n\n";
        }
    };

    std::shared_ptr<LineBreak> line_break();

    struct Keyword : Node
    {
        std::string keyword;
        Keyword(std::string keyword)
            : keyword(keyword)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Keyword> return_statement();
    std::shared_ptr<Keyword> sync_threads();

    void format_and_write(std::string fname, std::string code);

    struct ArgumentList : Node
    {
        std::vector<std::shared_ptr<Node>> arguments;
        ArgumentList(){};
        std::string render() const;
        void        append(std::shared_ptr<Node> a);
        bool        empty() const;
    };

    using TemplateList = ArgumentList;

    struct StatementList : Node
    {
        std::vector<std::shared_ptr<Node>> statements;
        StatementList(){};
        std::string render() const;
    };

    inline void operator+=(StatementList& stmts, const std::shared_ptr<Node>& s)
    {
        stmts.statements.push_back(s);
    }

    inline void operator+=(StatementList& stmts, const StatementList& s)
    {
        for(auto x : s.statements)
            stmts.statements.push_back(x);
    }

    //
    // Helpers
    //

    std::string join(std::string seperator, std::vector<std::shared_ptr<Node>> const& x);

    //
    // Arithmetic
    //

#define MAKE_BINARY(NAME, SEP, PRECEDENCE)                         \
    struct NAME : Node                                             \
    {                                                              \
        std::string separator{SEP};                                \
                                                                   \
        std::shared_ptr<Node> lhs, rhs;                            \
        NAME(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs) \
            : lhs(lhs)                                             \
            , rhs(rhs)                                             \
        {                                                          \
            precedence = PRECEDENCE;                               \
        };                                                         \
        std::string render() const;                                \
    };

    MAKE_BINARY(Add, " + ", 50);
    MAKE_BINARY(Multiply, " * ", 100);
    MAKE_BINARY(Subtract, " - ", 50);
    MAKE_BINARY(Divide, " / ", 100);
    MAKE_BINARY(Modulus, " % ", 100);

    MAKE_BINARY(And, " && ", 100);
    MAKE_BINARY(Less, " < ", 100);

    //
    // Variables
    //

    struct Literal : Node
    {
        std::string literal;
        Literal(std::string l)
        {
            literal = l;
        }
        template <typename T>
        Literal(T l)
        {
            literal = std::to_string(l);
        }
        std::string render() const override;
    };

    template <typename T>
    std::shared_ptr<Literal> literal(T l)
    {
        return std::make_shared<Literal>(l);
    }

    template <typename T>
    std::shared_ptr<Node> literal(std::shared_ptr<T> l)
    {
        return l;
    }

    std::shared_ptr<Literal> literal_true();
    std::shared_ptr<Literal> literal_false();

    struct VariableDeclaration : Node
    {
        std::string           name, type;
        std::shared_ptr<Node> size;
        VariableDeclaration(std::string name, std::string type, std::shared_ptr<Node> size)
            : name(name)
            , type(type)
            , size(size)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<VariableDeclaration>
        variable_declaration(std::string name, std::string type, std::shared_ptr<Node> size);

    struct VariableArgument : Node
    {
        std::string name, type;
        VariableArgument(std::string name, std::string type)
            : name(name)
            , type(type)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<VariableArgument> variable_argument(std::string name, std::string type);

    struct Variable : Node
    {
        std::string           name, type;
        std::shared_ptr<Node> size;
        Variable() {}
        Variable(std::string name)
            : name(name)
        {
        }
        Variable(std::string name, std::string type)
            : name(name)
            , type(type)
        {
        }
        Variable(std::string name, std::string type, std::shared_ptr<Node> size)
            : name(name)
            , type(type)
            , size(size)
        {
        }
        std::string                          render() const override;
        std::shared_ptr<VariableDeclaration> declaration() const;
        std::shared_ptr<VariableArgument>    argument() const;
        std::shared_ptr<Variable>            address() const;
    };

    std::shared_ptr<Variable> variable(std::string name, std::string type);

    struct ScalarVariable : Variable
    {
        std::shared_ptr<ScalarVariable> x, y;
        ScalarVariable()
            : Variable()
        {
        }
        ScalarVariable(std::string name)
            : Variable(name)
        {
            make_xy();
        }
        ScalarVariable(std::string name, std::string type)
            : Variable(name, type)
        {
            make_xy();
        }
        void make_xy()
        {
            x       = std::make_shared<ScalarVariable>();
            y       = std::make_shared<ScalarVariable>();
            x->type = "real_type_t<" + type + ">";
            x->name = name + ".x";
            y->type = "real_type_t<" + type + ">";
            y->name = name + ".y";
        }
    };

    std::shared_ptr<ScalarVariable> scalar(std::string name);
    std::shared_ptr<ScalarVariable> scalar(std::string name, std::string type);

    struct ArrayVariable : Variable
    {
        using Variable::Variable;

        template <typename T>
        std::shared_ptr<ScalarVariable> at(T i)
        {
            return scalar(name + "[" + i->render() + "]");
        }

        template <typename T>
        std::shared_ptr<ScalarVariable> operator[](T i)
        {
            return scalar(name + "[" + i->render() + "]");
        }

        std::shared_ptr<ScalarVariable> operator[](int i)
        {
            return scalar(name + "[" + std::to_string(i) + "]");
        }

        std::shared_ptr<ScalarVariable> operator[](uint i)
        {
            return scalar(name + "[" + std::to_string(i) + "]");
        }
    };

    std::shared_ptr<ArrayVariable> array(std::string name);
    std::shared_ptr<ArrayVariable> array(std::string name, std::string type);
    std::shared_ptr<ArrayVariable>
        array(std::string name, std::string type, std::shared_ptr<Node> size);

    struct Assign : Node
    {
        std::shared_ptr<Node> lhs, rhs;
        Assign(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs)
            : lhs(lhs)
            , rhs(rhs)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Assign> assign(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs);

    //
    // Blocks
    //

    struct IfBlock : Node
    {
        std::shared_ptr<Node> condition;
        StatementList         body;
        IfBlock(std::shared_ptr<Node> condition, StatementList body)
            : condition(condition)
            , body(body)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Node> if_block(std::shared_ptr<Node> condition, StatementList body);

    //
    // Functions
    //

    enum FunctionTypeQualifier
    {
        NONE,
        DEVICE,
        GLOBAL,
        HOST
    };

    struct Function : Node
    {
        std::string                name;
        TemplateList               templates;
        ArgumentList               arguments;
        ArgumentList               kernel_arguments;
        StatementList              body;
        enum FunctionTypeQualifier type_qualifier;
        Function(std::string name)
            : name(name)
            , type_qualifier(NONE)
        {
        }
        Function(std::string name, ArgumentList args)
            : name(name)
            , arguments(args)
            , type_qualifier(NONE)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Function> function(std::string name);
    struct FunctionCall : Function
    {
        using Function::Function;
        std::string render() const override;
    };

    std::shared_ptr<FunctionCall> call(std::string name);
    std::shared_ptr<FunctionCall> call(std::string name, ArgumentList args);

    //
    // Operators
    //

#define MAKE_OVERLOAD(NAME, OP)                                                           \
    inline std::shared_ptr<NAME> OP(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs) \
    {                                                                                     \
        return std::make_shared<NAME>(lhs, rhs);                                          \
    }                                                                                     \
    template <typename T>                                                                 \
    inline std::shared_ptr<NAME> OP(std::shared_ptr<Node> lhs, T rhs)                     \
    {                                                                                     \
        return std::make_shared<NAME>(lhs, literal(rhs));                                 \
    }                                                                                     \
    inline std::shared_ptr<NAME> OP(std::shared_ptr<ScalarVariable> lhs,                  \
                                    std::shared_ptr<Node>           rhs)                  \
    {                                                                                     \
        return std::make_shared<NAME>(lhs, rhs);                                          \
    }                                                                                     \
    template <typename T>                                                                 \
    inline std::shared_ptr<NAME> OP(std::shared_ptr<ScalarVariable> lhs, T rhs)           \
    {                                                                                     \
        return std::make_shared<NAME>(lhs, literal(rhs));                                 \
    }

    MAKE_OVERLOAD(Add, operator+)
    MAKE_OVERLOAD(Subtract, operator-)
    MAKE_OVERLOAD(Multiply, operator*)
    MAKE_OVERLOAD(Divide, operator/)
    MAKE_OVERLOAD(Modulus, operator%)
    MAKE_OVERLOAD(Less, operator<)
    MAKE_OVERLOAD(And, operator&&)



    std::shared_ptr<Function> make_planar(std::shared_ptr<Function> f, std::string varname);

}
