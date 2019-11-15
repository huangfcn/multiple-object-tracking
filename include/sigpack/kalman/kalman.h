// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef KALMAN_H
#define KALMAN_H

#include <functional>
namespace sp
{
    #define FCN_XUW [=](arma::mat x,arma::mat u,arma::mat w)     // Lambda function f(x,u,w) ([capture] by copy)
    using fcn_t = std::function< double(arma::mat,arma::mat,arma::mat) >;
    using fcn_v = std::vector<fcn_t>;
    using fcn_m = std::vector<fcn_v>;

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Evaluate function f(x,u,w)
    /// @param  f Function pointer vector
    /// @param  x Input vector
    /// @param  u Input vector
    /// @param  w Input vector
    /// @return y Output column vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    arma_inline arma::mat eval_fcn( const fcn_v f, const arma::mat& x, const arma::mat& u, const arma::mat& w)
    {
        arma::mat y((arma::uword)(f.size()),1,arma::fill::zeros);
        for( arma::uword n=0; n<y.n_rows;n++)
            y(n,0)    = f[n](x,u,w);
        return y;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Evaluate function f(x,u,w=0)
    /// @param  f Function pointer vector
    /// @param  x Input vector
    /// @param  u Input vector
    /// @return y Output column vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    arma_inline arma::mat eval_fcn( const fcn_v f, const arma::mat& x, const arma::mat& u)
    {
        arma::mat w0(0,0,arma::fill::zeros);
        return eval_fcn(f,x,u,w0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Evaluate function f(x,u=0,w=0)
    /// @param  f Function pointer vector
    /// @param  x Input vector
    /// @return y Output column vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    arma_inline arma::mat eval_fcn( const fcn_v f, const arma::mat& x)
    {
        arma::mat w0(0,0,arma::fill::zeros);
        arma::mat u0(w0);
        return eval_fcn(f,x,u0,w0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Discretize to a state transition and state noise cov matrix from an LTI system.
    ///
    ///  Discretize a continious LTI system using van Loan method. The model is in the form
    ///
    ///     dx/dt = F x + Ww,  w ~ N(0,Qc)
    ///
    ///  Result of discretization is the model
    ///
    ///     x[k] = A x[k-1] + q, q ~ N(0,Q)
    ///
    ///  See http://becs.aalto.fi/en/research/bayes/ekfukf/
    /// @param F  LTI system model matrix
    /// @param W  LTI system noise model matrix
    /// @param Qc LTI power spectra density matrix
    /// @param dT Discretization delta time
    /// @param A  Output - discrete system model
    /// @param Q  Output - discrete state noise cov matrix
    ////////////////////////////////////////////////////////////////////////////////////////////
    arma_inline void lti2discr(const arma::mat& F,const arma::mat& W, const arma::mat& Qc, const double dT, arma::mat& A,arma::mat& Q)
    {
        arma::uword M = F.n_rows;

        // Solve A
        A = arma::expmat(F*dT);

        // Solve Q by using matrix fraction decomposition
        arma::mat AB = arma::zeros(2*M,M);
        arma::mat CD = arma::zeros(2*M,2*M);
        arma::mat EF = arma::zeros(2*M,M);
        EF.submat(M,0, 2*M-1,M-1)  = arma::eye(M,M);
        CD.submat(0,0, M-1,M-1)    = F;
        CD.submat(M,M,2*M-1,2*M-1) = -F.t();
        CD.submat(0,M,M-1,2*M-1)   = W*Qc*W.t();

        AB = arma::expmat(CD*dT)*EF;

        Q = AB.rows(0,M-1)*arma::inv(AB.rows(M,2*M-1));
    }

    ///
    /// @defgroup kalman Kalman
    /// \brief Kalman predictor/filter functions.
    /// @{

    ////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Kalman filter class.
    ///
    /// Implements Kalman functions for the discrete system
    /// \f[ x_k = Ax_{k-1}+Bu_{k-1} + w_{k-1}  \f]
    /// with measurements
    /// \f[ z_k = Hx_k + v_k  \f]
    /// The predicting stage is
    /// \f[ \hat{x}^-_k = A\hat{x}_{k-1}+Bu_{k-1} \f]
    /// \f[ P^-_k = AP_{k-1}A^T+Q \f]
    /// and the updates stage
    /// \f[ K_k = P^-_kH^T(HP^-_kH^T+R)^{-1} \f]
    /// \f[ \hat{x}_k = \hat{x}^-_k + K_k(z_k-H\hat{x}^-_k) \f]
    /// \f[ P_k = (I-K_kH)P^-_k \f]
    ///
    /// For detailed info see: http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    ////////////////////////////////////////////////////////////////////////////////////////////
    class KF
    {
        protected:
            arma::uword N;        ///< Number of states
            arma::uword M;        ///< Number of inputs
            arma::uword L;        ///< Number of measurements/observations
            bool lin_proc;        ///< Linearity flag for process
            bool lin_meas;        ///< Linearity flag for measurement
            arma::mat x;          ///< State vector
            arma::mat z_err;      ///< Prediction error
            arma::mat A;          ///< State transition matrix
            arma::mat B;          ///< Input matrix
            arma::mat H;          ///< Measurement matrix
            arma::mat P;          ///< Error covariance matrix (estimated accuracy)
            arma::mat Q;          ///< Process noise
            arma::mat R;          ///< Measurement noise
            arma::mat K;          ///< Kalman gain vector
            fcn_v f;              ///< Vector of Kalman state transition functions
            fcn_v h;              ///< Vector of Kalman measurement functions
        public:
            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Constructor.
            ////////////////////////////////////////////////////////////////////////////////////////////
            KF(arma::uword _N,arma::uword _M,arma::uword _L)
            {
                N = _N;   // Nr of states
                M = _M;   // Nr of measurements/observations
                L = _L;   // Nr of inputs
                lin_proc = true;
                lin_meas = true;
                x.set_size(N,1); x.zeros();
                z_err.set_size(M,1); z_err.zeros();
                A.set_size(N,N); A.eye();
                B.set_size(N,L); B.zeros();
                H.set_size(M,N); H.zeros();
                P.set_size(N,N); P.eye();
                Q.set_size(N,N); Q.eye();
                R.set_size(M,M); R.eye();
                K.set_size(N,M); K.zeros();
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Destructor.
            ////////////////////////////////////////////////////////////////////////////////////////////
            ~KF() {}

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Clear the internal states and pointer.
            ////////////////////////////////////////////////////////////////////////////////////////////
            void clear(void)
            {
                K.zeros();
                P.eye();
                x.zeros();
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            //  Set/get functions
            ////////////////////////////////////////////////////////////////////////////////////////////
            void set_state_vec(const arma::mat& _x)       { x = _x;   }    // Set state vector.[Nx1]
            void set_trans_mat(const arma::mat& _A)       { A = _A;   }    // Set state transition matrix.[NxN]
            void set_control_mat(const arma::mat& _B)     { B = _B;   }    // Set input matrix.[NxL]
            void set_meas_mat(const arma::mat& _H)        { H = _H;   }    // Set measurement matrix.[MxN]
            void set_err_cov(const arma::mat& _P)         { P = _P;   }    // Set error covariance matrix.[NxN]
            void set_proc_noise(const arma::mat& _Q)      { Q = _Q;   }    // Set process noise cov. matrix.
            void set_meas_noise(const arma::mat& _R)      { R = _R;   }    // Set meas noise cov. matrix.
            void set_kalman_gain(const arma::mat& _K)     { K = _K;   }    // Set Kalman gain matrix.[NxM]
            void set_trans_fcn(fcn_v _f)    // Set state transition functions
            {
                f = _f;
                lin_proc = false;
            }
            void set_meas_fcn(fcn_v _h)     // Set measurement functions
            {
                h = _h;
                lin_meas = false;
            }

            arma::mat get_state_vec(void)        { return x;       }   // Get states [Nx1]
            arma::mat get_err(void)              { return z_err;   }   // Get pred error [Mx1]
            arma::mat get_kalman_gain(void)      { return K;       }   // Get Kalman gain [NxM]
            arma::mat get_err_cov(void)          { return P;       }   // Get error cov matrix [NxN]

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Predict the internal states using a control input.
            /// @param u Input/control signal
            ////////////////////////////////////////////////////////////////////////////////////////////
            void predict(const arma::mat u )
            {
                x = A*x+B*u;      // New state
                P = A*P*A.t()+Q;  // New error covariance
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Predict the internal states, no control.
            ////////////////////////////////////////////////////////////////////////////////////////////
            void predict(void)
            {
                 arma::mat u0(L,1,arma::fill::zeros);
                 predict(u0);
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Correct and update the internal states.
            ////////////////////////////////////////////////////////////////////////////////////////////
            void update(const arma::mat z )
            {
                // Compute the Kalman gain
                K = P*H.t()*inv(H*P*H.t()+R);

                // Update estimate with measurement z
                z_err = z-H*x;
                x = x+K*z_err;

                // Joseph’s form covariance update
                arma::mat Jf = arma::eye<arma::mat>(N,N)-K*H;
                P = Jf*P*Jf.t() + K*R*K.t();
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Rauch-Tung-Striebel smoother.
            /// See http://www.lce.hut.fi/~ssarkka/course_k2011/pdf/course_booklet_2011.pdf
            ////////////////////////////////////////////////////////////////////////////////////////////
            void rts_smooth(const arma::mat& Xf, const arma::cube& Pf, arma::mat& Xs, arma::cube& Ps )
            {
                arma::uword Nf = Xf.n_cols;
                arma::mat X_pred(N,1);
                arma::mat P_pred(N,N);
                arma::mat C(N,N);

                // Copy forward data
                Xs = Xf;
                Ps = Pf;

                // Backward filter
                for(arma::uword n=Nf-2; n>0; n--)
                {
                    // Project state and error covariance
                    X_pred = A*Xf.col(n);
                    P_pred = A*Pf.slice(n)*A.t()+Q;

                    // Update
                    C = Pf.slice(n)*A.t()*inv(P_pred);
                    Xs.col(n)   += C*(Xs.col(n+1)-X_pred);
                    Ps.slice(n) += C*(Ps.slice(n+1)-P_pred)*C.t();
                }
            }
    }; // End class KF

    ////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief First order Extended Kalman filter class
    ///
    /// Implements Kalman functions for the discrete system with additive noise
    /// \f[ x_k = f(x_{k-1})+Bu_{k-1} + w_{k-1}  \f]
    /// and with measurements
    /// \f[ z_k = h(x_k) + v_k  \f]
    /// where f(x) and h(x) may be nonlinear functions.
    /// The predicting stage is
    /// \f[ \hat{x}^-_k = A\hat{x}_{k-1}+Bu_{k-1} \f]
    /// \f[ P^-_k = AP_{k-1}A^T+Q \f]
    /// and the updates stage
    /// \f[ K_k = P^-_kH^T(HP^-_kH^T+R)^{-1} \f]
    /// \f[ \hat{x}_k = \hat{x}^-_k + K_k(z_k-H\hat{x}^-_k) \f]
    /// \f[ P_k = (I-K_kH)P^-_k \f]
    ///
    /// Where A and H is the Jacobians of f(x) and h(x) functions.
    ///
    /// For detailed info see: http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    ////////////////////////////////////////////////////////////////////////////////////////////
    class EKF: public KF
    {
        protected:
            fcn_m f_jac;            ///< Matrix of Extended Kalman state transition jacobian
            fcn_m h_jac;            ///< Matrix of Extended Kalman measurement transition jacobian
            double dx;              ///< Finite difference approximation step size
        public:

            EKF(arma::uword _N,arma::uword _M,arma::uword _L): KF(_N,_M,_L)
            {
                dx = 1e-7;  // Default diff step size
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            //  Set/get functions
            ////////////////////////////////////////////////////////////////////////////////////////////
            void set_diff_step(double _dx)  { dx  = _dx;    }   // Set diff step size
            void set_state_jac(fcn_m _f)    { f_jac = _f;   }   // Set EKF state transition jacobian
            void set_meas_jac(fcn_m _h)     { h_jac = _h;   }   // Set EKF measurement transition jacobian

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Calculate and evaluate Jacobian matrix using finite difference approximation
            /// @param  _F Jacobian matrix d/dx evaluated at x
            /// @param  _f Function vector [ f0(x,u,w) ...  fN(x,u,w)]
            /// @param  _x State vector
            ///
            /// Alternative: Complex Step Diff: http://blogs.mathworks.com/cleve/2013/10/14/complex-step-differentiation/
            ////////////////////////////////////////////////////////////////////////////////////////////
            void jacobian_diff(arma::mat& _F, fcn_v _f, const arma::mat& _x)
            {
                arma::uword nC = _F.n_cols;
                arma::uword nR = static_cast<arma::uword>(_f.size());
                arma::mat z0(nC,1,arma::fill::zeros); // Zero matrix, assume dim u and w <= states

                if(nR==0 || nR!=_F.n_rows) err_handler("Function list is empty or wrong size");

                for(arma::uword c=0; c<nC; c++)
                {
                    arma::mat xp(_x);
                    arma::mat xm(_x);
                    xp(c,0) += dx;
                    xm(c,0) -= dx;

                    // Finite diff approx, evaluate at x,u=0,w=0
                    for(arma::uword r=0; r<nR; r++)
                        _F(r,c) = (_f[r](xp,z0,z0)-_f[r](xm,z0,z0))/(2*dx);
                }
            }
            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Calculate and evaluate Jacobian matrix using finite difference approximation
            /// @param  _F Jacobian matrix d/dx evaluated at x
            /// @param  _f Function vector [ f0(x,u,w) ...  fN(x,u,w)]
            ////////////////////////////////////////////////////////////////////////////////////////////
            void jacobian_diff(arma::mat& _F, fcn_v _f)
            {
               jacobian_diff(_F,_f,x); // Use current state from object
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Evaluate Jacobian matrix using analytical jacobian
            /// @param  _F Jacobian matrix d/dx evaluated at x
            /// @param  _f_m Jacobian function matrix
            /// @param  _x State vector
            ////////////////////////////////////////////////////////////////////////////////////////////
            void jacobian_analytical(arma::mat& _F, fcn_m _f_m, const arma::mat& _x)
            {
                arma::uword nC = _F.n_cols;
                arma::uword nR = static_cast<arma::uword>(_f_m.size());

                if(nR==0 || nR!=_F.n_rows) err_handler("Function list is empty or wrong size");

                arma::mat z0(nC,1,arma::fill::zeros); // Zero matrix, assume dim u and w <= states
                for(arma::uword c=0; c<nC; c++)
                {
                    for(arma::uword r=0; r<nR; r++)
                        _F(r,c) = _f_m[r][c](_x,z0,z0);
                }
            }
            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Evaluate Jacobian matrix using analytical jacobian
            /// @param  _F Jacobian matrix d/dx evaluated at x
            /// @param  _f_m Jacobian function matrix
            ////////////////////////////////////////////////////////////////////////////////////////////
            void jacobian_analytical(arma::mat& _F, fcn_m _f_m)
            {
               jacobian_analytical(_F,_f_m,x); // Use current state from object
            }


            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Predict the internal states using a control input.
            /// @param u Input/control signal
            ////////////////////////////////////////////////////////////////////////////////////////////
            void predict(const arma::mat u )
            {
                if( !lin_proc )
                {
                    // Update A with jacobian approx or analytical if set
                    if(f_jac.size()>0)
                        jacobian_analytical(A,f_jac);
                    else
                        jacobian_diff(A,f);

                    // Predict state   x+ = f(x,u,0)
                    x = eval_fcn(f,x,u);
                }
                else  // Linear process
                    x = A*x+B*u;

                // Project error covariance
                P = A*P*A.t()+Q;
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Predict the internal states, no control.
            ////////////////////////////////////////////////////////////////////////////////////////////
            void predict(void)
            {
                 arma::mat u0(L,1,arma::fill::zeros);
                 predict(u0);
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Correct and update the internal states. EKF
            ////////////////////////////////////////////////////////////////////////////////////////////
            void update(const arma::mat z )
            {
                arma::mat z_hat(M,1);

                if(!lin_meas) // Nonlinear measurement
                {
                    // Update H with jacobian approx or analytical if set
                    if( h_jac.size()>0)
                        jacobian_analytical(H,h_jac);
                    else
                        jacobian_diff(H,h);

                    // Update measurement
                    z_hat = eval_fcn(h,x);
                }
                else  // Linear meas
                    z_hat = H*x;

                // Calc residual
                z_err = z-z_hat;

                // Compute the Kalman gain
                K = P*H.t()*inv(H*P*H.t()+R);

                // Update estimate with measurement residual
                x = x+K*z_err;

                // Joseph’s form covariance update
                arma::mat Jf = arma::eye<arma::mat>(N,N)-K*H;
                P = Jf*P*Jf.t()+K*R*K.t();
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Rauch-Tung-Striebel smoother.
            /// See http://www.lce.hut.fi/~ssarkka/course_k2011/pdf/course_booklet_2011.pdf
            ////////////////////////////////////////////////////////////////////////////////////////////
            void rts_smooth(const arma::mat& Xf, const arma::cube& Pf, arma::mat& Xs, arma::cube& Ps )
            {
                arma::uword Nf = Xf.n_cols;
                arma::mat X_pred(N,1);
                arma::mat P_pred(N,N);
                arma::mat C(N,N);

                // Copy forward data
                Xs = Xf;
                Ps = Pf;

                // Backward filter
                for(arma::uword n=Nf-2; n>0; n--)
                {
                    if( !lin_proc )
                    {
                        // Update A with jacobian approx or analytical if set
                        if(f_jac.size()>0)
                            jacobian_analytical(A,f_jac,Xf.col(n));
                        else
                            jacobian_diff(A,f,Xf.col(n));

                        // Project state
                        X_pred = eval_fcn(f,Xf.col(n));
                    }
                    else  // Linear process
                    {
                        // Project state
                        X_pred = A*Xf.col(n);
                    }

                    // Project error covariance
                    P_pred = A*Pf.slice(n)*A.t()+Q;

                    // Update
                    C = Pf.slice(n)*A.t()*inv(P_pred);
                    Xs.col(n)   += C*(Xs.col(n+1)-X_pred);
                    Ps.slice(n) += C*(Ps.slice(n+1)-P_pred)*C.t();
                }

            }

    }; // End class EKF


    ////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Unscented Kalman filter class
    ///
    /// Implements Kalman functions for the discrete system with additive noise
    /// \f[ x_k = f(x_{k-1})+Bu_{k-1} + w_{k-1}  \f]
    /// and with measurements
    /// \f[ z_k = h(x_k) + v_k  \f]
    /// where f(x) and h(x) may be nonlinear functions.
    ///
    /// The predict and update stage is using the unscented transform of the sigma points
    /// of the input and/or the measurements. <br>
    /// For detailed info see: http://www.lce.hut.fi/~ssarkka/course_k2011/pdf/course_booklet_2011.pdf
    ////////////////////////////////////////////////////////////////////////////////////////////
    class UKF: public KF
    {
        protected:
            double    alpha;     ///< Spread factor of sigma points
            double    beta;      ///< x distr. prior knowledge factor
            double    kappa;     ///< Scaling par.
            double    lambda;

            arma::mat X;         ///< Sigma points
            arma::mat S;         ///< Output covariance
            arma::mat C;         ///< Cross covariance input-output
            arma::vec Wx;        ///< Weights states
            arma::vec Wp;        ///< Weights covariance
        public:

            UKF(arma::uword _N,arma::uword _M,arma::uword _L): KF(_N,_M,_L)
            {
                alpha  = 1e-3;
                beta   = 2.0;
                kappa  = 0;
                lambda = alpha*alpha*(_N+kappa)-_N;
                X.set_size(_N,2*_N+1);X.zeros();
                Wx.set_size(2*_N+1);Wx.zeros();
                Wp.set_size(2*_N+1);Wp.zeros();
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            //  Set/get functions
            ////////////////////////////////////////////////////////////////////////////////////////////
            void set_alpha(double _a)  { alpha   = _a;    }   // Set alpha
            void set_beta(double _b)   { beta    = _b;    }   // Set beta
            void set_kappa(double _k)  { kappa   = _k;    }   // Set kappa
            void set_lambda(double _l) { lambda  = _l;    }   // Set lambda

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Calculate sigma point weights
            ////////////////////////////////////////////////////////////////////////////////////////////
            void update_weights( void )
            {
                // Update lambda
                lambda = alpha*alpha*(N+kappa)-N;

                // Update weights
                Wx(0) = lambda/(N+lambda);
                Wp(0) = lambda/(N+lambda)+(1-alpha*alpha+beta);

                for(arma::uword n=1;n<=2*N;n++)
                {
                    Wx(n) = 1/(2*(N+lambda));
                    Wp(n) = Wx(n);
                }
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Calculate sigma points around a reference point
            /// @param _x State matrix
            /// @param _P Covariance matrix
            ////////////////////////////////////////////////////////////////////////////////////////////
            void update_sigma(const arma::mat& _x, const arma::mat& _P )
            {
                // Update sigma points using Cholesky decomposition
                arma::mat _A = sqrt(N + lambda)*arma::chol(_P,"lower");

                X = arma::repmat(_x,1,2*N+1);
                X.cols(1  ,  N) += _A;
                X.cols(N+1,2*N) -= _A;
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Calculate unscented transform
            ////////////////////////////////////////////////////////////////////////////////////////////
            arma::mat ut( const arma::mat& _x, const arma::mat& _P, const fcn_v _f )
            {
                arma::uword Ny = static_cast<arma::uword>(_f.size());
                arma::mat y(Ny,1);
                S.set_size(Ny,Ny);
                C.set_size(N,Ny);

                update_weights();
                update_sigma(_x,_P);

                // Propagate sigma points through nonlinear function
                arma::mat Xy(Ny,2*N+1);
                for(arma::uword n=0;n<2*N+1;n++)
                    Xy.col(n) = eval_fcn(_f,X.col(n));

                // New mean
                y = Xy*Wx;

                // New cov
                S = (Xy-arma::repmat(y,1,2*N+1))*arma::diagmat(Wp)*(Xy-arma::repmat(y,1,2*N+1)).t();

                // New crosscov
                C = (X-arma::repmat(_x,1,2*N+1))*arma::diagmat(Wp)*(Xy-arma::repmat(y,1,2*N+1)).t();

                return y;
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Predict the internal states using a control input.
            /// @param u Input/control signal
            ////////////////////////////////////////////////////////////////////////////////////////////
            void predict(const arma::mat u )
            {
                if(!lin_proc) // Nonlinear process
                {
                    // Do the Unscented Transform
                    x = ut(x,P,f)+B*u;

                    // Add process noise cov
                    P = S + Q;
                }
                else  // Linear process
                {
                    // Project state
                    x = A*x+B*u;

                    // Project error covariance
                    P = A*P*A.t()+Q;
                }
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Predict the internal states, no control. Convenient function
            ////////////////////////////////////////////////////////////////////////////////////////////
            void predict(void)
            {
                 arma::mat u0(L,1,arma::fill::zeros);
                 predict(u0);
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Correct and update the internal states. UKF
            ////////////////////////////////////////////////////////////////////////////////////////////
            void update(const arma::mat z )
            {
                arma::mat z_hat(M,1);
                if(!lin_meas) // Nonlinear measurement
                {
                    // Do the Unscented Transform
                    z_hat = ut(x,P,h);

                    // Add measurement noise cov
                    S = S + R;

                    // Compute the Kalman gain
                    K = C *arma::inv(S);

                    // Update estimate with measurement residual
                    z_err = z-z_hat;
                    x = x+K*z_err;

                    // Update covariance, TODO: improve numerical perf. Josephs form?
                    P = P-K*S*K.t();
                }
                else  // Linear measurement
                {
                    // Calc residual
                    z_err = z-H*x;

                    // Compute the Kalman gain
                    K = P*H.t()*inv(H*P*H.t()+R);

                    // Update estimate with measurement residual
                    x = x+K*z_err;

                    // Joseph’s form covariance update
                    arma::mat Jf = arma::eye<arma::mat>(N,N)-K*H;
                    P = Jf*P*Jf.t()+K*R*K.t();
                }
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Rauch-Tung-Striebel smoother.
            /// See http://www.lce.hut.fi/~ssarkka/course_k2011/pdf/course_booklet_2011.pdf
            ////////////////////////////////////////////////////////////////////////////////////////////
            void rts_smooth(const arma::mat& Xf, const arma::cube& Pf, arma::mat& Xs, arma::cube& Ps )
            {
                arma::uword Nf = Xf.n_cols;
                arma::mat X_pred(N,1,arma::fill::zeros);
                arma::mat P_pred(N,N,arma::fill::zeros);
                arma::mat D_pred(N,N,arma::fill::zeros);

                // Copy forward data
                Xs = Xf;
                Ps = Pf;

                // Backward filter
                for(arma::uword k=Nf-2; k>0; k--)
                {
                    if( !lin_proc )
                    {
                        // Do the unscented transform
                        X_pred = ut(Xf.col(k),Pf.slice(k),f);
                        P_pred = S+Q;

                        // Update
                        D_pred = C*inv(P_pred);
                        Xs.col(k)   += D_pred*(Xs.col(k+1)-X_pred);
                        Ps.slice(k) += D_pred*(Ps.slice(k+1)-P_pred)*D_pred.t();
                    }
                    else  // Linear process
                    {
                        // Project state
                        X_pred = A*Xf.col(k);

                        // Project error covariance
                        P_pred = A*Pf.slice(k)*A.t()+Q;

                        // Update
                        D_pred = Pf.slice(k)*A.t()*inv(P_pred);
                        Xs.col(k)   += D_pred*(Xs.col(k+1)-X_pred);
                        Ps.slice(k) += D_pred*(Ps.slice(k+1)-P_pred)*D_pred.t();
                    }
                }
            }
    }; // End class UKF
}
#endif // KALMAN_H
