#include <sigpack.h>
#include "../top/cnntype.h"

using namespace std;
using namespace sp;

//////////////////////////////////////////////////////////////////////////
// Kalman Parameters                                                    //
//////////////////////////////////////////////////////////////////////////
typedef struct _kalman_cb_s
{
    arma::uword N; 
    arma::uword M;
    arma::uword L; 

    KF * pkalman;
} kalman_tracker_t;

kalman_tracker_t * kalman_tracker_alloc()
{
    return ((kalman_tracker_t *)(malloc(sizeof(kalman_tracker_t))));
}

void kalman_tracker_free(kalman_tracker_t * ptracker)
{
    free(ptracker);
}

int kalman_tracker_initialize(kalman_tracker_t * ptracker, arma::colvec & x0)
{
    ///////////////////////////////////////////////////////////////////
    // BUILD THE MODEL                                               //
    ///////////////////////////////////////////////////////////////////
    arma::uword N = 6;  // Nr of states
    arma::uword M = 4;  // Nr of measurements
    arma::uword L = 0;  // Nr of inputs

    /* dimensions */
    ptracker->N = N;
    ptracker->M = M;
    ptracker->L = L;

    ptracker->pkalman = new KF(N, M, L);

    // Initialisation and setup of system
    double P0 = 1e+4;
    double Q0 = 1e-2;
    double R0 = 512.0;

    // Meas interval
    double dT = 1.0;

    ///////////////////////////////////////////////////////////////////
    // [X0,Y0,X1,Y1,Vx,Vy], 2-corner-positions, velocity
    arma::mat A =
    {
        {1, 0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 1},
        {0, 0, 1, 0, 1, 0},
        {0, 0, 0, 1, 0, 1},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1},
    };
    ptracker->pkalman->set_trans_mat(A);

    arma::mat H =
    {
        {1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0},
    };
    ptracker->pkalman->set_meas_mat(H);

    arma::mat Q = 
    {
        {0.25, 0.00, 0.00, 0.00, 0.50, 0.00},
        {0.00, 0.25, 0.00, 0.00, 0.00, 0.50},
        {0.00, 0.00, 0.25, 0.00, 0.50, 0.00},
        {0.00, 0.00, 0.00, 0.25, 0.00, 0.50},
        {0.50, 0.00, 0.50, 0.00, 1.00, 0.00},
        {0.00, 0.50, 0.00, 0.50, 0.00, 1.00},
    };
    Q = Q0 * Q;
    ptracker->pkalman->set_proc_noise(Q);

    arma::mat R = R0*arma::eye(M,M);
    ptracker->pkalman->set_meas_noise(R);

    arma::mat P = P0*arma::eye(N,N);
    ptracker->pkalman->set_err_cov(P);

    ptracker->pkalman->set_state_vec(x0);
    ///////////////////////////////////////////////////////////////////

    return (0);
}

void kalman_tracker_deinitialize(kalman_tracker_t * ptracker)
{
    delete ptracker->pkalman;
}

/* get tracker prediction */
void kalman_tracker_predict(kalman_tracker_t * ptracker, bbox_t * pbbox)
{
    ptracker->pkalman->predict();

    /* get l, t, b r */
    arma::colvec x = ptracker->pkalman->get_state_vec();

    pbbox->l = x[0];
    pbbox->t = x[1];
    pbbox->r = x[2];
    pbbox->b = x[3];
}

void kalman_tracker_update(kalman_tracker_t * ptracker, bbox_t * pbbox)
{
    arma::colvec x(4);

    x[0] = pbbox->l;
    x[1] = pbbox->t;
    x[2] = pbbox->r;
    x[3] = pbbox->b;

    ptracker->pkalman->update(x);
}


void tracker_predict(void * ptracker, float * rgb, bbox_t * pbox)
{
    kalman_tracker_predict((kalman_tracker_t *)ptracker, pbox);
}

void tracker_update(void * ptracker, float * rgb, bbox_t * pbox)
{
    kalman_tracker_update((kalman_tracker_t *)ptracker, pbox);
}

void tracker_delete(void * ptracker)
{
    kalman_tracker_deinitialize((kalman_tracker_t *)ptracker);
    kalman_tracker_free((kalman_tracker_t *)ptracker);

}

void * tracker_new(bbox_t * pbox)
{
    kalman_tracker_t * pkalman = kalman_tracker_alloc();

    arma::colvec x(6);
    x[0] = pbox->l;
    x[1] = pbox->t;
    x[2] = pbox->r;
    x[3] = pbox->b;
    x[4] = 0;
    x[5] = 0;

    kalman_tracker_initialize(pkalman, x);

    return ((void *)pkalman);
}