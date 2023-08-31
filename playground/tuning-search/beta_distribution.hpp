#include <iostream>
#include <sstream>
#include <string>
#include <random>

template <typename RealType = double>
class beta_distribution
{
private:
    typedef std::gamma_distribution<RealType> gamma_dist_type;

    gamma_dist_type a_gamma, b_gamma;

    template <typename URNG>
    RealType generate(URNG& engine, gamma_dist_type& x_gamma, gamma_dist_type& y_gamma)
    {
        RealType x = x_gamma(engine);
        return x / (x + y_gamma(engine));
    }

public:
    explicit beta_distribution(RealType a = 1.0, RealType b = 1.0)
        : a_gamma(a)
        , b_gamma(b)
    {
    }

    void param(RealType a, RealType b)
    {
        a_gamma = gamma_dist_type(a);
        b_gamma = gamma_dist_type(b);
    }

    template <typename URNG>
    RealType operator()(URNG& engine)
    {
        return generate(engine, a_gamma, b_gamma);
    }

    template <typename URNG>
    RealType operator()(URNG& engine, RealType a, RealType b)
    {
        gamma_dist_type a_param_gamma(a), b_param_gamma(b);
        return generate(engine, a_param_gamma, b_param_gamma);
    }

    RealType min() const
    {
        return 0.0;
    }
    RealType max() const
    {
        return 1.0;
    }

    RealType a() const
    {
        return a_gamma.alpha();
    }
    RealType b() const
    {
        return b_gamma.alpha();
    }

    bool operator==(const beta_distribution<RealType>& other) const
    {
        return (a_gamma == other.a_gamma && b_gamma == other.b_gamma);
    }

    bool operator!=(const beta_distribution<RealType>& other) const
    {
        return !(*this == other);
    }
};