//
// Simple AST based code generator.
//
// This generator
// - builds a syntax tree and renders HIP code;
// - uses runtime polymorphism to support different node types;
// - doesn't enforce type consistency (all nodes inherit from a single
//   base node class).
//
// It's not very sophisticated, but it should be easy to pick up and
// get started.
//

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "generator.hpp"

namespace gen
{

    //
    // Keywords etc
    //

    std::shared_ptr<LineBreak> line_break()
    {
        return std::make_shared<LineBreak>();
    }

    std::string Keyword::render() const
    {
        return keyword + ";";
    }

    std::shared_ptr<Keyword> return_statement()
    {
        return std::make_shared<Keyword>("return");
    }

    //
    // Helpers
    //

    std::string join(std::string seperator, std::vector<std::shared_ptr<Node>> const& x)
    {
        if(x.empty())
            return "";
        std::string s = x[0]->render();
        for(int i = 1; i < x.size(); ++i)
            s += seperator + x[i]->render();
        return s;
    }

    //
    // Arithmetic
    //

    std::string Add::render() const
    {
        return join(" + ", operands);
    }

    std::shared_ptr<Add> add(std::vector<std::shared_ptr<Node>> operands)
    {
        return std::make_shared<Add>(operands);
    }

    std::string Subtract::render() const
    {
        return join(" - ", operands);
    }

    std::shared_ptr<Subtract> sub(std::vector<std::shared_ptr<Node>> operands)
    {
        return std::make_shared<Subtract>(operands);
    }

    std::string Multiply::render() const
    {
        return join(" * ", operands);
    }

    std::shared_ptr<Multiply> multiply(std::vector<std::shared_ptr<Node>> operands)
    {
        return std::make_shared<Multiply>(operands);
    }

    std::string Divide::render() const
    {
        return join(" / ", operands);
    }

    std::shared_ptr<Divide> divide(std::vector<std::shared_ptr<Node>> operands)
    {
        return std::make_shared<Divide>(operands);
    }

    std::string Mod::render() const
    {
        return join(" % ", operands);
    }

    std::shared_ptr<Mod> mod(std::vector<std::shared_ptr<Node>> operands)
    {
        return std::make_shared<Mod>(operands);
    }

    std::string Group::render() const
    {
        return "( " + group->render() + " )";
    }

    std::shared_ptr<Group> group(std::shared_ptr<Node> group)
    {
        return std::make_shared<Group>(group);
    }

    //
    // Operators
    //

    std::string BinaryOperator::render() const
    {
        return lhs->render() + " " + op + " " + rhs->render();
    }

    std::shared_ptr<BinaryOperator> greater_than(std::shared_ptr<Node> lhs,
                                                 std::shared_ptr<Node> rhs)
    {
        return std::make_shared<BinaryOperator>(">", lhs, rhs);
    }

    //
    // Variables
    //

    std::string Literal::render() const
    {
        return literal;
    }

    std::string VariableDeclaration::render() const
    {
        if(!size)
            return type + " " + name + ";";
        return type + " " + name + "[" + size->render() + "];";
    }

    std::shared_ptr<VariableDeclaration>
        variable_declaration(std::string name, std::string type, std::shared_ptr<Node> size)
    {
        return std::make_shared<VariableDeclaration>(name, type, size);
    }

    std::string VariableArgument::render() const
    {
        return type + " " + name;
    }

    std::shared_ptr<VariableArgument> variable_argument(std::string name, std::string type)
    {
        return std::make_shared<VariableArgument>(name, type);
    }

    std::string Variable::render() const
    {
        return name;
    }

    std::shared_ptr<VariableDeclaration> Variable::declaration() const
    {
        return variable_declaration(name, type, size);
    }

    std::shared_ptr<VariableArgument> Variable::argument() const
    {
        return variable_argument(name, type);
    }

    std::shared_ptr<Variable> Variable::address() const
    {
        return variable("&" + name, type + " *");
    }

    std::shared_ptr<Variable> variable(std::string name, std::string type)
    {
        return std::make_shared<Variable>(name, type);
    }

    std::shared_ptr<ScalarVariable> scalar(std::string name)
    {
        return std::make_shared<ScalarVariable>(name);
    }

    std::shared_ptr<ScalarVariable> scalar(std::string name, std::string type)
    {
        return std::make_shared<ScalarVariable>(name, type);
    }

    std::shared_ptr<ArrayVariable> array(std::string name)
    {
        return std::make_shared<ArrayVariable>(name);
    }

    std::shared_ptr<ArrayVariable> array(std::string name, std::string type)
    {
        return std::make_shared<ArrayVariable>(name, type);
    }

    std::shared_ptr<ArrayVariable>
        array(std::string name, std::string type, std::shared_ptr<Node> size)
    {
        return std::make_shared<ArrayVariable>(name, type, size);
    }

    std::string Assign::render() const
    {
        return lhs->render() + " = " + rhs->render() + ";";
    }

    std::shared_ptr<Assign> assign(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs)
    {
        return std::make_shared<Assign>(lhs, rhs);
    }

    //
    // Blocks
    //

    std::string IfBlock::render() const
    {
        return "if(" + condition->render() + ") {" + join("", body) + "}";
    }

    std::shared_ptr<IfBlock> if_block(std::shared_ptr<Node> condition)
    {
        return std::make_shared<IfBlock>(condition);
    }

    //
    // Functions
    //

    std::string Function::render() const
    {
        std::string s;
        if(!templates.empty())
            s += "template <" + join(", ", templates) + "> ";
        if(type_qualifier == DEVICE)
            s += "__device__ ";
        if(type_qualifier == GLOBAL)
            s += "__global__ ";
        if(type_qualifier == HOST)
            s += "__host__ ";
        s += "void ";
        s += name + "(" + join(", ", arguments) + ") {";
        s += join("\n", body);
        s += "}";
        return s;
    }

    std::shared_ptr<Function> function(std::string name)
    {
        return std::make_shared<Function>(name);
    }

    std::string FunctionCall::render() const
    {
        std::string s = name;
        if(!templates.empty())
            s += "<" + join(", ", templates) + ">";
        if(!kernel_arguments.empty())
            s += "<<<" + join(", ", kernel_arguments) + ">>>";
        s += "(" + join(", ", arguments) + ");";
        return s;
    }

    std::shared_ptr<FunctionCall> call(std::string name)
    {
        return std::make_shared<FunctionCall>(name);
    }

}
