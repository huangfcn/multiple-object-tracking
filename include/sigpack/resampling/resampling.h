// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SP_RESAMPLING_H
#define SP_RESAMPLING_H
namespace sp
{
///
/// @defgroup resampling Resampling
/// \brief Resampling functions.
/// @{

////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Upsampling without anti alias filtering.
/// @returns A vector with p-1 zeros inserted in the input vector [x0,0,0,..,x1,0,0,..-..,xN,0,0,..]
/// @param x Input vector
/// @param p Upsampling factor
////////////////////////////////////////////////////////////////////////////////////////////
template <class T1>
arma::Col<T1> upsample(const arma::Col<T1>& x, const int p )
{
    long int N = x.size();
    arma::Col<T1> y;
    y.set_size(p*N);
    y.zeros();
    for(long int n=0; n<N; n++)
        y[p*n] = x[n];
    return y;
}

////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Downsampling without anti alias filtering.
/// @returns A vector with every q:th value from the input vector
/// @param x Input vector
/// @param q Downsampling factor
////////////////////////////////////////////////////////////////////////////////////////////
template <class T1>
arma::Col<T1> downsample(const arma::Col<T1>& x, const int q )
{
    arma::Col<T1> y;
    int N = int(floor(1.0*x.size()/q));
    y.set_size(N);
    for(long int n=0; n<N; n++)
        y[n] = x[n*q];
    return y;
}

///
/// \brief A resampling class.
///
/// Implements up/downsampling functions
///
template <class T1>
class resampling
{
private:
    FIR_filt<T1,double,T1> aa_filt;
    arma::vec H;         ///< Filter coefficients
    arma::vec K;         ///< Number of filter coefficients
    arma::uword P;       ///< Upsampling rate
    arma::uword Q;       ///< Downsampling rate

public:
    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Constructor.
    /// @param _P Upsampling rate
    /// @param _Q Downsampling rate
    /// @param _H FIR filter coefficients
    ////////////////////////////////////////////////////////////////////////////////////////////
    resampling(const arma::uword _P,const arma::uword _Q,const arma::vec _H)
    {
        P = _P;
        Q = _Q;
        H = _H;
        K = H.n_elem;
        aa_filt.set_coeffs(H);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Constructor using a #fir1 filter with 8*M+1 taps and cutoff 1/M where M=max(P,Q)
    /// @param _P Upsampling rate
    /// @param _Q Downsampling rate
    ////////////////////////////////////////////////////////////////////////////////////////////
    resampling(const arma::uword _P,const arma::uword _Q)
    {
        P = _P;
        Q = _Q;
        arma::uword M=(P>Q)?P:Q;
        H = fir1(8*M,1.0f/M);
        K = H.n_elem;
        aa_filt.set_coeffs(H);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Destructor.
    ////////////////////////////////////////////////////////////////////////////////////////////
    ~resampling() {}


    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Downsampling with anti alias filter.
    ///
    /// @param in  Input vector
    /// @param out Output vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    void downfir(const arma::Col<T1>& in, arma::Col<T1>& out)
    {
        arma::uword sz = in.n_elem;
        for( arma::uword n=0; n<sz; n++)
        {
            T1 tmp = aa_filt(in[n]);
            if(n%Q==0)
                out[n/Q] = tmp;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Upsampling with anti alias filter.
    ///
    /// @param in  Input vector
    /// @param out Output vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    void upfir(const arma::Col<T1>& in, arma::Col<T1>& out)
    {
        arma::uword sz = P*in.n_elem;
        for( arma::uword n=0; n<sz; n++)
        {
            if(n%P==0)
                out[n] = P*aa_filt(in[n/P]);
            else
                out[n] = P*aa_filt(0.0);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Resampling by a rational P/Q with anti alias filtering.
    ///
    /// The caller needs to allocate the input and output vector so that length(out)==length(in)*P/Q
    /// @param in  Input vector
    /// @param out Output vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    void upfirdown(const arma::Col<T1>& in, arma::Col<T1>& out)
    {
        arma::uword sz = P*in.n_elem;
        T1 tmp;
        for( arma::uword n=0; n<sz; n++)
        {
            if(n%P==0)
                tmp = aa_filt(in[n/P]);
            else
                tmp = aa_filt(0.0);
            if(n%Q==0)
                out[n/Q] = P*tmp;
        }
    }
};
/// @}
} // end namespace
#endif
