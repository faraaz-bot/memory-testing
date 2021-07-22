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
        for(auto x : statements)
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

        std::string cmd = "/opt/rocm/llvm/bin/clang-format -i -style=file " + tname;
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

    //
    // AST transformations
    //

    //
    // Planar
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

        // MAKE_VISITOR(Expression, Add)
        // MAKE_VISITOR(Expression, And)
        // MAKE_VISITOR(Expression, ComplexLiteral)
        // MAKE_VISITOR(Expression, Divide)
        // MAKE_VISITOR(Expression, Less)
        // MAKE_VISITOR(Expression, Literal)
        // MAKE_VISITOR(Expression, Modulus)
        // MAKE_VISITOR(Expression, Multiply)
        // MAKE_VISITOR(Expression, ScalarVariable)
        // MAKE_VISITOR(Expression, Subtract)
        // MAKE_VISITOR(Expression, Variable)

        // MAKE_VISITOR(Statement, Call)
        // MAKE_VISITOR(Statement, CommentLines)
        // MAKE_VISITOR(Statement, Declaration)
        // MAKE_VISITOR(Statement, For)
        // MAKE_VISITOR(Statement, If)
        // MAKE_VISITOR(Statement, LineBreak)
        // MAKE_VISITOR(Statement, Return)
        // MAKE_VISITOR(Statement, StatementList)
        // MAKE_VISITOR(Statement, SyncThreads)

        ArgumentList operator()(const ArgumentList& x)
        {
            ArgumentList y;
            for(auto a : x.arguments)
            {
                auto var = std::dynamic_pointer_cast<Variable>(a);
                if(var)
                {
                    auto aname = var->name;
                    if(aname == varname)
                    {
                        auto re  = std::make_shared<Variable>(var->name, var->type, var->size);
                        re->name = rename;
                        re->type = "real_type_t<" + var->type + ">";
                        auto im  = std::make_shared<Variable>(var->name, var->type, var->size);
                        im->name = imname;
                        im->type = "real_type_t<" + var->type + ">";
                        y.append(re);
                        y.append(im);
                    }
                    else
                    {
                        y.append(a);
                    }
                }
                else
                {
                    y.append(a);
                }
            }
            return y;
        }

        StatementList operator()(const StatementList& x)
        {
            StatementList y;
            for(auto a : x.statements) {
                y += a;
            }
            return y;
        }


        // Statement operator()(const Assign& x)
        //     {
        //         if(x.lhs.name == varname && std::holds_alternative<Variable>(x.rhs))
        //         {
        //             // on lhs, lhs needs to be split; use .x and .y on rhs

        //             auto rhs   = std::get<Variable>(x.rhs);
        //             auto stmts = StatementList();

        //             auto re = Variable(x.lhs);
        //             re.name = rename;
        //             auto im = Variable(x.lhs);
        //             im.name = imname;

        //             stmts += Assign(re, rhs.x);
        //             stmts += Assign(im, rhs.y);
        //             return Statement(stmts);
        //         }
        //         else if(std::holds_alternative<Variable>(x.rhs)
        //                 && std::get<Variable>(x.rhs).name == varname)
        //         {
        //             // on rhs, rhs needs to be joined as a complex literal

        //             auto rhs = std::get<Variable>(x.rhs);
        //             auto re  = Variable(rhs);
        //             re.name  = rename;
        //             auto im  = Variable(rhs);
        //             im.name  = imname;
        //             return Statement(Assign(x.lhs, ComplexLiteral(re.render(), im.render())));
        //         }

        //         return Statement(x);
        //     }

        std::shared_ptr<Function> operator()(const std::shared_ptr<Function>& x)
        {
            auto nargs   = (*this)(x->arguments);
            auto y       = std::make_shared<Function>(x->name, nargs);
            y->arguments = (*this)(x->arguments);
            y->body      = (*this)(x->body);
            return y;
        }
    };

    std::shared_ptr<Function> make_planar(std::shared_ptr<Function> x, std::string varname)
    {
        auto visitor = MakePlanarVisitor(varname);
        return visitor(x);
    }

    /*
        auto args = std::dynamic_pointer_cast<ArgumentList>(x);
        if (args) {
            auto nargs = std::make_shared<ArgumentList>();
            for(auto arg : args->arguments) {
                auto var = std::dynamic_pointer_cast<Variable>(arg);
                if (var) {
                    if (var->name == varname) {
                        auto nvar = std::make_shared<Variable>(var->name, var->type, var->size);
                        nvar->type = "real_type_t<" + var->type + ">";
                        nargs->append(nvar);
                    } else {
                        nargs->append(var);
                    }
                } else {
                    nargs->append(arg);
                }
            }
            return nargs;
        }

        auto func = std::dynamic_pointer_cast<Function>(x);
        if (func) {
            auto alist = make_planar(func->arguments);
            auto nfunc = std::make_shared<Function>(func->name, alist);
        }

        return x;
*/

}
