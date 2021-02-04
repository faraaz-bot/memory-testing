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
        virtual std::string render() const = 0;
        virtual ~Node()                    = default;
    };

    //
    // Helpers
    //

    std::string join(std::string seperator, std::vector<std::shared_ptr<Node>> const& x);

    //
    // Arithmetic
    //

    struct Add : Node
    {
        std::vector<std::shared_ptr<Node>> operands;
        Add(std::vector<std::shared_ptr<Node>> o)
            : operands{o}
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Add> add(std::vector<std::shared_ptr<Node>> operands);

    struct Subtract : Node
    {
        std::vector<std::shared_ptr<Node>> operands;
        Subtract(std::vector<std::shared_ptr<Node>> o)
            : operands{o}
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Subtract> sub(std::vector<std::shared_ptr<Node>> operands);

    struct Multiply : Node
    {
        std::vector<std::shared_ptr<Node>> operands;
        Multiply(std::vector<std::shared_ptr<Node>> o)
            : operands{o}
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Multiply> multiply(std::vector<std::shared_ptr<Node>> operands);

    struct Divide : Node
    {
        std::vector<std::shared_ptr<Node>> operands;
        Divide(std::vector<std::shared_ptr<Node>> o)
            : operands{o}
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Divide> divide(std::vector<std::shared_ptr<Node>> operands);

    struct Mod : Node
    {
        std::vector<std::shared_ptr<Node>> operands;
        Mod(std::vector<std::shared_ptr<Node>> o)
            : operands{o}
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Mod> mod(std::vector<std::shared_ptr<Node>> operands);

    struct Group : Node
    {
        std::shared_ptr<Node> group;
        Group(std::shared_ptr<Node> group)
            : group(group)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<Group> group(std::shared_ptr<Node> group);

    //
    // Variables
    //

    struct Literal : Node
    {
        std::string literal;
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

    struct VariableDeclaration : Node
    {
        std::string name, type;
        VariableDeclaration(std::string name, std::string type)
            : name(name)
            , type(type)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<VariableDeclaration> variable_declaration(std::string name, std::string type);
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
        std::string name, type;
        Variable(std::string name)
            : name(name)
        {
        }
        Variable(std::string name, std::string type)
            : name(name)
            , type(type)
        {
        }
        std::string                          render() const override;
        std::shared_ptr<VariableDeclaration> declaration() const;
        std::shared_ptr<VariableArgument>    argument() const;
    };

    std::shared_ptr<Variable> variable(std::string name, std::string type);

    struct ScalarVariable : Variable
    {
        using Variable::Variable;
    };

    std::shared_ptr<ScalarVariable> scalar(std::string name);

    struct ArrayVariable : Variable
    {
        using Variable::Variable;

        template <typename T>
        std::shared_ptr<ScalarVariable> operator[](int i)
        {
            return scalar(name + "[" + std::to_string(i) + "]");
        }

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
    };

    std::shared_ptr<ArrayVariable> array(std::string name);
    std::shared_ptr<ArrayVariable> array(std::string name, std::string type);
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
        std::shared_ptr<Node>              condition;
        std::vector<std::shared_ptr<Node>> body;
        IfBlock(std::shared_ptr<Node> condition)
            : condition(condition)
        {
        }
        std::string render() const override;
    };

    std::shared_ptr<IfBlock> if_block(std::shared_ptr<Node> condition);

    //
    // Functions
    //

    struct Function : Node
    {
        std::string                        name;
        std::vector<std::shared_ptr<Node>> templates;
        std::vector<std::shared_ptr<Node>> arguments;
        std::vector<std::shared_ptr<Node>> body;
        Function(std::string name)
            : name(name)
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

}
