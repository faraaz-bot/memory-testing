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

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

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

    std::shared_ptr<Keyword> sync_threads()
    {
        return std::make_shared<Keyword>("__syncthreads()");
    }

    std::shared_ptr<Literal> literal_true()
    {
        return literal(std::string("true"));
    }

    std::shared_ptr<Literal> literal_false()
    {
        return literal(std::string("false"));
    }

    //
    // Helpers
    //

    std::string join(std::string seperator, std::vector<std::shared_ptr<Node>> const& x)
    {
        if(x.empty())
            return "";
        std::string s = x[0]->render();
        for(uint i = 1; i < x.size(); ++i)
            s += seperator + x[i]->render();
        return s;
    }

    // std::string join(std::string seperator, StatementList x)
    // {
    //     return join(seperator, x.statements);
    // }

    void ArgumentList::append(std::shared_ptr<Node> a)
    {
        arguments.push_back(a);
    }

    bool ArgumentList::empty() const
    {
        return arguments.empty();
    }

    std::string ArgumentList::render() const
    {
        return join(", ", arguments);
    }

    std::string StatementList::render() const
    {
        std::string s;
        for(auto x: statements)
            s += x->render();
        return s;
    }

    //
    // Arithmetic
    //

#define MAKE_BINARY_RENDER(NAME)            \
    std::string NAME::render() const        \
    {                                       \
        std::string s;                      \
        if(lhs->precedence < precedence)    \
            s += "(" + lhs->render() + ")"; \
        else                                \
            s += lhs->render();             \
        s += separator;                     \
        if(rhs->precedence < precedence)    \
            s += "(" + rhs->render() + ")"; \
        else                                \
            s += rhs->render();             \
        return s;                           \
    }

    MAKE_BINARY_RENDER(Add);
    MAKE_BINARY_RENDER(Multiply);
    MAKE_BINARY_RENDER(Subtract);
    MAKE_BINARY_RENDER(Divide);
    MAKE_BINARY_RENDER(Modulus);

    MAKE_BINARY_RENDER(And);
    MAKE_BINARY_RENDER(Less);

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
        return "if(" + condition->render() + ") {" + join("", body.statements) + "}";
    }

    std::shared_ptr<Node> if_block(std::shared_ptr<Node> condition, StatementList body)
    {
        return std::make_shared<IfBlock>(condition, body);
    }

    //
    // Functions
    //

    std::string Function::render() const
    {
        std::string s;
        if(!templates.empty())
            s += "template <" + join(", ", templates.arguments) + "> ";
        if(type_qualifier == DEVICE)
            s += "__device__ ";
        if(type_qualifier == GLOBAL)
            s += "__global__ ";
        if(type_qualifier == HOST)
            s += "__host__ ";
        s += "void ";
        s += name + "(" + join(", ", arguments.arguments) + ") {";
        s += join("\n", body.statements);
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
            s += "<" + join(", ", templates.arguments) + ">";
        if(!kernel_arguments.empty())
            s += "<<<" + join(", ", kernel_arguments.arguments) + ">>>";
        s += "(" + join(", ", arguments.arguments) + ");";
        return s;
    }

    std::shared_ptr<FunctionCall> call(std::string name)
    {
        return std::make_shared<FunctionCall>(name);
    }

    std::shared_ptr<FunctionCall> call(std::string name, ArgumentList args)
    {
        return std::make_shared<FunctionCall>(name, args);
    }

    //
    // Misc
    //

    void format_and_write(std::string fname, std::string code)
    {
        std::ofstream     ofile;
        std::ifstream     ifile;
        std::stringstream formatted, existing;

        auto tname = fname + ".tmp";

        ofile.open(tname);
        ofile << code;
        ofile.close();

        std::string cmd = "clang-format-10 -i -style=file " + tname;
        std::system(cmd.c_str());
        ifile.open(tname);
        formatted << ifile.rdbuf();
        ifile.close();

        unlink(tname.c_str());

        bool exists = static_cast<bool>(std::ifstream(fname));
        if(exists)
        {
            ifile.open(fname);
            existing << ifile.rdbuf();
            ifile.close();

            if(formatted.str().compare(existing.str()) == 0)
                return;
        }

        ofile.open(fname);
        ofile << formatted.str();
        ofile.close();
    }

}
