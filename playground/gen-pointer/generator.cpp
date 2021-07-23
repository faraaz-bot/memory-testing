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
    MAKE_BINARY_RENDER(Greater);

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
        std::string s;
        s += name;
        if(index)
        {
            s += "[" + index->render() + "]";
        }
        s += component;
        return s;
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

#define MAKE_BINARY_VISITOR(CLS)                                            \
    virtual std::shared_ptr<Node> operator()(const std::shared_ptr<CLS>& x) \
    {                                                                       \
        auto lhs = (*this)(x->lhs);                                         \
        auto rhs = (*this)(x->rhs);                                         \
        return std::make_shared<CLS>(lhs, rhs);                             \
    }

    class CopyVisitor
    {
    public:
        virtual ArgumentList operator()(const ArgumentList& x)
        {
            ArgumentList y;
            for(auto a : x.arguments)
            {
                y.append((*this)(a));
            }
            return y;
        }

        virtual StatementList operator()(const StatementList& x)
        {
            StatementList y;
            for(auto a : x.statements)
            {
                y += (*this)(a);
            }
            return y;
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<Function>& x)
        {
            auto nargs   = (*this)(x->arguments);
            auto y       = std::make_shared<Function>(x->name, nargs);
            y->arguments = (*this)(x->arguments);
            y->body      = (*this)(x->body);
            return y;
        }

        MAKE_BINARY_VISITOR(Add)
        MAKE_BINARY_VISITOR(And)
        MAKE_BINARY_VISITOR(Assign)
        MAKE_BINARY_VISITOR(Divide)
        MAKE_BINARY_VISITOR(Less)
        MAKE_BINARY_VISITOR(Modulus)
        MAKE_BINARY_VISITOR(Multiply)
        MAKE_BINARY_VISITOR(Subtract)

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<VariableDeclaration>& x)
        {
            if(x->size)
            {
                auto size = (*this)(x->size);
                return std::make_shared<VariableDeclaration>(x->name, x->type, size);
            }
            return std::make_shared<VariableDeclaration>(x->name, x->type, nullptr);
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<ScalarVariable>& x)
        {
            auto y = std::make_shared<ScalarVariable>(x->name, x->type, x->index);
            y->component = x->component;
            // y->x = (*this)(x->x);
            // y->y = (*this)(x->y);
            return y;
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<Variable>& x)
        {
            std::shared_ptr<Node> size, index;
            if(x->size)
                size = (*this)(x->size);
            if(x->index)
                index = (*this)(x->index);
            auto y = std::make_shared<Variable>(x->name, x->type, size, index);
            y->component = x->component;
            return y;
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<ComplexLiteral>& x)
        {
            auto re = (*this)(x->re);
            auto im = (*this)(x->im);
            return std::make_shared<ComplexLiteral>(re, im);
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<Literal>& x)
        {
            return x;
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<Keyword>& x)
        {
            return x;
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<LineBreak>& x)
        {
            return x;
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<IfBlock>& x)
        {
            auto condition = (*this)(x->condition);
            auto body      = (*this)(x->body);
            return std::make_shared<IfBlock>(condition, body);
        }

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<FunctionCall>& x)
        {
            auto f       = std::make_shared<FunctionCall>(x->name);
            f->arguments = (*this)(x->arguments);
            f->body      = (*this)(x->body);
            return f;
        }

#define MAKE_DISPATCH(CLS)                \
    if(std::dynamic_pointer_cast<CLS>(x)) \
        return (*this)(std::dynamic_pointer_cast<CLS>(x));

        virtual std::shared_ptr<Node> operator()(const std::shared_ptr<Node>& x)
        {
            MAKE_DISPATCH(Add)
            MAKE_DISPATCH(And)
            MAKE_DISPATCH(Assign)
            MAKE_DISPATCH(ComplexLiteral)
            MAKE_DISPATCH(Divide)
            MAKE_DISPATCH(Less)
            MAKE_DISPATCH(Literal)
            MAKE_DISPATCH(Modulus)
            MAKE_DISPATCH(Multiply)
            MAKE_DISPATCH(Subtract)
            MAKE_DISPATCH(Variable)

            MAKE_DISPATCH(ScalarVariable)

            MAKE_DISPATCH(FunctionCall)
            // MAKE_DISPATCH(CommentLines)
            MAKE_DISPATCH(VariableDeclaration)
            // MAKE_DISPATCH(For)
            MAKE_DISPATCH(IfBlock)
            MAKE_DISPATCH(Keyword)
            MAKE_DISPATCH(LineBreak)

            std::cout << "UNHANDLED " << x->render() << std::endl;

            return x;
        }
    };

    //
    // Planar
    //
    class MakePlanarVisitor : public CopyVisitor
    {
    public:
        using CopyVisitor::operator();

        std::string varname, rename, imname;

        MakePlanarVisitor(std::string varname)
            : varname(varname)
        {
            rename = varname + "re";
            imname = varname + "im";
        }

        ArgumentList operator()(const ArgumentList& x) override
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

        std::shared_ptr<Node> operator()(const std::shared_ptr<Assign>& x) override
        {
            auto lhs = std::dynamic_pointer_cast<ScalarVariable>(x->lhs);
            auto rhs = std::dynamic_pointer_cast<ScalarVariable>(x->rhs);

            if(lhs == nullptr || rhs == nullptr)
                return CopyVisitor::operator()(x);

            if(lhs->name == varname)
            {
                // on lhs, lhs needs to be split; use .x and .y on rhs

                auto stmts = std::make_shared<StatementList>();

                auto re  = std::make_shared<ScalarVariable>(lhs->name, lhs->type, lhs->index);
                re->name = rename;
                auto im  = std::make_shared<ScalarVariable>(lhs->name, lhs->type, lhs->index);
                im->name = imname;

                *stmts += assign(re, rhs->x);
                *stmts += assign(im, rhs->y);

                return std::dynamic_pointer_cast<Node>(stmts);
            }
            else if(rhs->name == varname)
            {
                // on rhs, rhs needs to be joined as a complex literal

                auto rhs = std::dynamic_pointer_cast<ScalarVariable>(x->rhs);
                auto re  = std::make_shared<ScalarVariable>(rhs->name, rhs->type, rhs->index);
                re->name = rename;
                auto im  = std::make_shared<ScalarVariable>(rhs->name, rhs->type, rhs->index);
                im->name = imname;

                return std::make_shared<Assign>(lhs, std::make_shared<ComplexLiteral>(re, im));
            }

            return CopyVisitor::operator()(x);
        }
    };

    std::shared_ptr<Function> make_planar(const std::shared_ptr<Function>& x, std::string varname)
    {
        auto visitor = MakePlanarVisitor(varname);
        return std::dynamic_pointer_cast<Function>(visitor(x));
    }

}
