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

struct Add : Node
{
    std::vector<std::shared_ptr<Node>> operands;
    Add(std::vector<std::shared_ptr<Node>> o)
        : operands{o}
    {
    }
    std::string render() const override
    {
        return join(" + ", operands);
    }
};

std::shared_ptr<Add> add(std::vector<std::shared_ptr<Node>> operands)
{
    return std::make_shared<Add>(operands);
}

struct Subtract : Node
{
    std::vector<std::shared_ptr<Node>> operands;
    Subtract(std::vector<std::shared_ptr<Node>> o)
        : operands{o}
    {
    }
    std::string render() const override
    {
        return join(" - ", operands);
    }
};

std::shared_ptr<Subtract> sub(std::vector<std::shared_ptr<Node>> operands)
{
    return std::make_shared<Subtract>(operands);
}

struct Multiply : Node
{
    std::vector<std::shared_ptr<Node>> operands;
    Multiply(std::vector<std::shared_ptr<Node>> o)
        : operands{o}
    {
    }
    std::string render() const override
    {
        return join(" * ", operands);
    }
};

std::shared_ptr<Multiply> multiply(std::vector<std::shared_ptr<Node>> operands)
{
    return std::make_shared<Multiply>(operands);
}

struct Divide : Node
{
    std::vector<std::shared_ptr<Node>> operands;
    Divide(std::vector<std::shared_ptr<Node>> o)
        : operands{o}
    {
    }
    std::string render() const override
    {
        return join(" / ", operands);
    }
};

std::shared_ptr<Divide> divide(std::vector<std::shared_ptr<Node>> operands)
{
    return std::make_shared<Divide>(operands);
}

struct Mod : Node
{
    std::vector<std::shared_ptr<Node>> operands;
    Mod(std::vector<std::shared_ptr<Node>> o)
        : operands{o}
    {
    }
    std::string render() const override
    {
        return join(" % ", operands);
    }
};

std::shared_ptr<Mod> mod(std::vector<std::shared_ptr<Node>> operands)
{
    return std::make_shared<Mod>(operands);
}

struct Group : Node
{
    std::shared_ptr<Node> group;
    Group(std::shared_ptr<Node> group)
        : group(group)
    {
    }
    std::string render() const override
    {
        return "( " + group->render() + " )";
    }
};

std::shared_ptr<Group> group(std::shared_ptr<Node> group)
{
    return std::make_shared<Group>(group);
}

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
    std::string render() const override
    {
        return literal;
    }
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
    std::string render() const override
    {
        return type + " " + name + ";";
    }
};

std::shared_ptr<VariableDeclaration> variable_declaration(std::string name, std::string type)
{
    return std::make_shared<VariableDeclaration>(name, type);
}

struct VariableArgument : Node
{
    std::string name, type;
    VariableArgument(std::string name, std::string type)
        : name(name)
        , type(type)
    {
    }
    std::string render() const override
    {
        return type + " " + name;
    }
};

std::shared_ptr<VariableArgument> variable_argument(std::string name, std::string type)
{
    return std::make_shared<VariableArgument>(name, type);
}

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
    std::string render() const override
    {
        return name;
    }
    std::shared_ptr<VariableDeclaration> declaration() const
    {
        return variable_declaration(name, type);
    }
    std::shared_ptr<VariableArgument> argument() const
    {
        return variable_argument(name, type);
    }
};

std::shared_ptr<Variable> variable(std::string name, std::string type)
{
    return std::make_shared<Variable>(name, type);
}

struct ScalarVariable : Variable
{
    using Variable::Variable;
};

std::shared_ptr<ScalarVariable> scalar(std::string name)
{
    return std::make_shared<ScalarVariable>(name);
}

struct ArrayVariable : Variable
{
    using Variable::Variable;
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

std::shared_ptr<ArrayVariable> array(std::string name)
{
    return std::make_shared<ArrayVariable>(name);
}

std::shared_ptr<ArrayVariable> array(std::string name, std::string type)
{
    return std::make_shared<ArrayVariable>(name, type);
}

struct Assign : Node
{
    std::shared_ptr<Node> lhs, rhs;
    Assign(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs)
        : lhs(lhs)
        , rhs(rhs)
    {
    }
    std::string render() const override
    {
        return lhs->render() + " = " + rhs->render() + ";";
    }
};

std::shared_ptr<Assign> assign(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs)
{
    return std::make_shared<Assign>(lhs, rhs);
}

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
    std::string render() const override
    {
        return "if(" + condition->render() + ") {" + join("", body) + "}";
    }
};

std::shared_ptr<IfBlock> if_block(std::shared_ptr<Node> condition)
{
    return std::make_shared<IfBlock>(condition);
}

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
    std::string render() const override
    {
        std::string s;
        if(!templates.empty())
            s += "template <" + join(", ", templates) + ">";
        s += "void " + name + "(" + join(", ", arguments) + ") {";
        s += join("\n", body);
        s += "}";
        return s;
    }
};

std::shared_ptr<Function> function(std::string name)
{
    return std::make_shared<Function>(name);
}

struct FunctionCall : Function
{
    using Function::Function;
    std::string render() const override
    {
        std::string s = name;
        if(!templates.empty())
            s += "<" + join(", ", templates) + ">";
        s += "(" + join(", ", arguments) + ");";
        return s;
    }
};

std::shared_ptr<FunctionCall> call(std::string name)
{
    return std::make_shared<FunctionCall>(name);
}

//
// FFTs!!!
//

std::vector<int> unique_factors(std::vector<int> const& factors)
{
    auto result(factors);
    std::sort(result.begin(), result.end());
    auto end = std::unique(result.begin(), result.end());
    result.resize(std::distance(result.begin(), end));
    return result;
}

template <typename T>
T product(std::vector<T> x, int last = -1)
{
    if(last == 0)
        return 1;
    if(last > 0)
        return std::accumulate(x.cbegin(), x.cbegin() + last, T(1), std::multiplies<T>());
    return std::accumulate(x.cbegin(), x.cend(), T(1), std::multiplies<T>());
}

//
// Stockham pass
//
std::shared_ptr<Function> make_device_fft_pass(int pass, std::vector<int> factors)
{
    //
    // function and argument definitions
    //
    auto length = product(factors);
    auto fft = function("forward_length" + std::to_string(length) + "_pass" + std::to_string(pass));
    auto scalar_type = variable("scalar_type", "typename");
    auto input       = array("input", "scalar_type *");
    auto output      = array("output", "scalar_type *");
    auto rw          = variable("rw", "unsigned int");
    auto thread      = variable("thread", "unsigned int");
    auto twiddles    = array("twiddles", "const scalar_type *");
    auto stride_in   = variable("stride_in", "const size_t");
    auto stride_out  = variable("stride_out", "const size_t");
    auto offset_in   = variable("offset_in", "unsigned int");
    auto offset_out  = variable("offset_out", "unsigned int");

    fft->templates.push_back(scalar_type->argument());
    fft->arguments.push_back(input->argument());
    fft->arguments.push_back(output->argument());
    fft->arguments.push_back(rw->argument());
    fft->arguments.push_back(thread->argument());
    fft->arguments.push_back(twiddles->argument());
    fft->arguments.push_back(stride_in->argument());
    fft->arguments.push_back(stride_out->argument());
    fft->arguments.push_back(offset_in->argument());
    fft->arguments.push_back(offset_out->argument());

    //
    // register definitions
    //
    auto unique  = unique_factors(factors);
    auto width   = factors[pass];
    auto height  = product(unique) / width;
    auto nheight = product(factors, pass);

    for(int r = 0; r < width * height; ++r)
    {
        auto R = variable("R" + std::to_string(r), "scalar_type *");
        fft->arguments.push_back(R->argument());
    }

    std::vector<std::shared_ptr<ScalarVariable>> R(width * height);
    for(int r = 0; r < width * height; ++r)
        R[r] = scalar("(*R" + std::to_string(r) + ")");

    auto x = *input;
    auto z = *output;
    auto T = *twiddles;

    //
    // load
    //
    if(pass == 0)
    {
        auto load = if_block(rw);
        for(int w = 0; w < width; ++w)
        {
            for(int h = 0; h < height; ++h)
            {
                // clang-format off
                auto idx = add({
                    offset_in,
                    multiply({
                        group(add({
                              multiply({literal(height), thread}),
                              literal((length / width) * w + h)})),
                        stride_in
                      })
                  });
                // clang-format on
                load->body.push_back(assign(R[width * h + w], x[idx]));
            }
        }
        fft->body.push_back(load);
    }

    //
    // twiddle
    //
    if(pass > 0)
    {
        auto W  = variable("W", "scalar_type");
        auto Wx = variable("W.x", "scalar_type");
        auto Wy = variable("W.y", "scalar_type");
        auto t  = variable("t", "scalar_type");
        auto tx = variable("t.x", "scalar_type");
        auto ty = variable("t.y", "scalar_type");
        fft->body.push_back(W->declaration());
        fft->body.push_back(t->declaration());

        for(int h = 0; h < height; ++h)
        {
            for(int w = 1; w < width; ++w)
            {
                // clang-format off
                auto idx =
                  add({
                      literal(nheight - 1 + w - 1),
                      multiply({
                          literal(width - 1),
                          group(
                                mod({
                                    group(
                                          add({
                                              multiply({literal(height), thread}),
                                              literal(h)})),
                                    literal(nheight)}))})});
                // clang-format on
                auto ridx = h * width + w;
                auto Rx   = variable(R[ridx]->name + ".x", "");
                auto Ry   = variable(R[ridx]->name + ".y", "");
                fft->body.push_back(assign(W, T[idx]));
                fft->body.push_back(assign(tx, sub({multiply({Wx, Rx}), multiply({Wy, Ry})})));
                fft->body.push_back(assign(ty, add({multiply({Wy, Rx}), multiply({Wx, Ry})})));
                fft->body.push_back(assign(R[ridx], t));
            }
        }
    }

    //
    // butterflies
    //
    for(int h = 0; h < height; ++h)
    {
        auto fwd = call("FwdRad" + std::to_string(width) + "B1");
        for(int w = 0; w < width; ++w)
            fwd->arguments.push_back(scalar("R" + std::to_string(h * width + w)));
        fft->body.push_back(fwd);
    }

    //
    // store
    //
    auto store = if_block(rw);
    if(pass < factors.size() - 1)
    {
        for(int h = 0; h < height; ++h)
        {
            for(int w = 0; w < width; ++w)
            {
                // clang-format off
                auto base = group(add({multiply({literal(height), thread}), literal(h)}));
                auto idx = add({
                    offset_out,
                    multiply({
                        group(add({
                              multiply({group(divide({base, literal(nheight)})), literal(width*nheight)}),
                              mod({base, literal(nheight)}),
                              literal(w*nheight)})),
                        stride_out
                      })
                  });
                // clang-format on
                store->body.push_back(assign(z[idx], R[h * width + w]));
            }
        }
    }
    else
    {
        height = factors[0];
        width  = product(unique) / height;

        for(int w = 0; w < width; ++w)
        {
            for(int h = 0; h < height; ++h)
            {
                // clang-format off
                auto idx = add({
                    offset_out,
                    multiply({
                        group(add({
                              multiply({literal(height), thread}),
                              literal((length / width) * w + h)})),
                        stride_out
                      })});
                // clang-format on
                store->body.push_back(assign(z[idx], R[width * h + w]));
            }
        }
    }
    fft->body.push_back(store);

    //
    // reload
    //
    // XXX something funky when square
    if(pass < factors.size() - 1)
    {
        height = factors[0];
        width  = product(unique) / height;
        //        std::swap(width, height);

        auto reload = if_block(rw);
        for(int w = 0; w < width; ++w)
        {
            for(int h = 0; h < height; ++h)
            {
                // clang-format off
                auto idx = add({
                    offset_out,
                    multiply({literal(height), thread}),
                    literal((length / width) * w + h)});
                // clang-format on
                reload->body.push_back(assign(R[h * width + w], z[idx]));
            }
        }
        fft->body.push_back(reload);
    }

    return fft;
}

std::shared_ptr<Function> make_device_fft(int length, std::vector<int> factors)
{

    auto fft = function("fwd_len" + std::to_string(length) + "_device");

    auto T          = variable("T", "typename");
    auto sb         = variable("sb", "StrideBin");
    auto sync       = variable("sync", "bool");
    auto twiddles   = variable("twiddles", "const T *");
    auto stride_in  = variable("stride_in", "const size_t");
    auto stride_out = variable("stride_out", "const size_t");
    auto rw         = variable("rw", "unsigned int");
    auto me         = variable("me", "unsigned int");
    auto ldsOffset  = variable("ldsOffset", "unsigned int");

    fft->templates.push_back(T->argument());
    fft->templates.push_back(sync->argument());
    fft->arguments.push_back(twiddles->argument());
    fft->arguments.push_back(stride_in->argument());
    fft->arguments.push_back(stride_out->argument());
    fft->arguments.push_back(rw->argument());
    fft->arguments.push_back(me->argument());
    fft->arguments.push_back(ldsOffset->argument());

    auto lwbIn  = variable("lwbIn", "T *");
    auto lwbOut = variable("lwbOut", "T *");
    auto lds    = variable("lds", "T *");

    fft->arguments.push_back(lwbIn->argument());
    fft->arguments.push_back(lwbOut->argument());
    fft->arguments.push_back(lds->argument());

    // XXX
    int width = 14;
    for(int i = 0; i < width; ++i)
    {
        auto R = variable("R" + std::to_string(i), "T");
        fft->body.push_back(R->declaration());
    }

    for(int pass = 0; pass < factors.size(); ++pass)
    {
        auto fwd = call("FwdPass" + std::to_string(pass) + "_len" + std::to_string(length));
        fwd->templates.push_back(T);
        fwd->templates.push_back(sb);
        fwd->templates.push_back(sync);
        fwd->arguments.push_back(twiddles);
        fwd->arguments.push_back(stride_in);
        fwd->arguments.push_back(stride_out);
        fwd->arguments.push_back(rw);
        fwd->arguments.push_back(me);

        std::shared_ptr<Node> offset_in  = ldsOffset;
        std::shared_ptr<Node> offset_out = ldsOffset;

        if(pass == 0)
            offset_in = literal(0);
        if(pass == factors.size() - 1)
            offset_out = literal(0);

        fwd->arguments.push_back(offset_in);
        fwd->arguments.push_back(offset_out);
        fwd->arguments.push_back(lds);
        fwd->arguments.push_back(lds);
        fwd->arguments.push_back(lds);
        fwd->arguments.push_back(lds);

        for(int i = 0; i < width; ++i)
        {
            fwd->arguments.push_back(scalar("&R" + std::to_string(i)));
        }

        fft->body.push_back(fwd);
    }

    return fft;
}

int main(int argc, char* argv[])
{
    //std::vector<int> factors = {4, 4, 4, 4};
    std::vector<int> factors = {7, 2, 2, 2};
    //std::vector<int> factors = {7, 8};
    //std::vector<int> factors = {5, 3};
    for(int pass = 0; pass < factors.size(); ++pass)
    {
        auto pass_kernel = make_device_fft_pass(pass, factors);
        std::cout << pass_kernel->render() << std::endl;
    }
}
