#include <math.h>
#include <fftw3.h>

#include <armadillo>
#include <sigpack.h>

#include "../top/cnntype.h"
#include "../libhog/fhog.h" 

using namespace arma;
using namespace std;
using namespace sp;

#ifndef max
#define max(x, y)   (((x) > (y)) ? (x) : (y))
#endif

#ifndef min
#define min(x, y)   (((x) < (y)) ? (x) : (y))
#endif

// typedef struct _bbox_pos_s
// {
//     int l, t, b, r;
// } bbox_t;

typedef struct kcf_cb_s
{
    fftwf_plan plan_xf[32];
    // fftwf_plan plan_zf[32];
    fftwf_plan plan_zb;

    int      cols;
    int      rows;
    
    int      f_cols;
    int      f_rows;
    int      f_chan;

    int      cell_size;

    /* features in time/frequency domain: (rows/4)*(cols/4)*31 */
    float         * xf_tm;
    fftwf_complex * xf_fq;
    fftwf_complex * xf_md;

    /* features in time/frequency domain: (rows/4)*(cols/4)*31 */
    float         * zf_tm;
    fftwf_complex * zf_fq;

    /* xf * xf' */
    fftwf_complex * yf;
    fftwf_complex * kf;
    fftwf_complex * zf;

    /* alphaf */
    float * alpha;
    float * response;

    /* */
    float feature_norm_ratio;

    /* labels & cos_win */
    arma::fmat labels;
    arma::fmat cos_win;

    /* */
    bbox_t pos;
    float  scale_vert, scale_horiz;

    /* first time run */
    int    first_update;
    float  factor;
    float  lamda;

} kcf_t;

template<class ty> void circshift(ty & out, const ty & in, int xshift, int yshift)
{
    int iOutputInd, iInputInd, ii, jj;
    int ydim = in.n_cols;
    int xdim = in.n_rows;
    for (int j = 0; j < ydim; j++)
    {
        jj = (j + yshift) % ydim;
        if (jj < 0) jj = jj + ydim;
        for (int i = 0; i < xdim; i++)
        {
            ii = (i + xshift) % xdim;
            if (ii < 0) ii = ii + xdim;
            out[jj * xdim + ii] = in[j * xdim + i];
        }
    }
}

static arma::fmat gaussian_shaped_labels(float sigma, int rows, int cols)
{
    arma::fcolvec _x(rows);
    arma::frowvec _y(cols);

    int range_y[2] = {-cols / 2, cols - cols / 2};
    int range_x[2] = {-rows / 2, rows - rows / 2};

    float sigma_s_inv = 1.0 / (sigma*sigma);
    for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i)
    {
        _x[i] = std::exp(-0.5 * x * x * sigma_s_inv);
    }

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j)
    {
        _y[j] = std::exp(-0.5 * y * y * sigma_s_inv);
    }

    arma::fmat labels = _x * _y;
    arma::fmat rot_labels(rows, cols);

    //rotate so that 1 is at top-left corner (see KCF paper for explanation)
    circshift(rot_labels, labels, range_x[0], range_y[0]);

    return rot_labels;
}

static arma::fmat cosine_window_function(int rows, int cols)
{
    arma::fcolvec y = sp::hann_f(rows);
    arma::fcolvec x = sp::hann_f(cols);

    return y * x.t();
}

static void kcf_fft2_label(kcf_t * pkcf, float * yd)
{
    fftwf_plan plan_yf = fftwf_plan_dft_r2c_2d(
        pkcf->f_cols, 
        pkcf->f_rows, 
        yd, 
        pkcf->yf, 
        FFTW_ESTIMATE
        );

    fftwf_execute     (plan_yf);
    fftwf_destroy_plan(plan_yf);
}

int kcf_initialize(kcf_t * pkcf, bbox_t * pbbox, int cell_size)
{
    int rows = pbbox->b - pbbox->t + 1;
    int cols = pbbox->r - pbbox->l + 1;

    pkcf->cols  = cols;
    pkcf->rows  = rows;

    pkcf->cell_size = cell_size;
    pkcf->f_rows    = rows / cell_size;
    pkcf->f_cols    = cols / cell_size;
    pkcf->f_chan    = (32 - 1);

    /* time domain & frequency domain feature space */
    pkcf->xf_tm   = (float         *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(float        ) * (pkcf->f_chan + 1), 16);
    pkcf->xf_fq   = (fftwf_complex *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(fftwf_complex) * (pkcf->f_chan + 1), 16);
    pkcf->xf_md   = (fftwf_complex *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(fftwf_complex) * (pkcf->f_chan + 1), 16);

    pkcf->yf      = (fftwf_complex *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(fftwf_complex) *  1, 16);

    pkcf->zf      = (fftwf_complex *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(fftwf_complex) *  1, 16);    
    pkcf->kf      = (fftwf_complex *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(fftwf_complex) *  1, 16);

    pkcf->alpha   = (float         *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(float        ) *  1, 16);
    pkcf->response= (float         *)_aligned_malloc((pkcf->f_cols) * (pkcf->f_rows) * sizeof(float        ) *  1, 16);

    /* initialize kf --> 0 */
    memset(pkcf->kf,    0, (pkcf->f_cols) * (pkcf->f_rows) * sizeof(fftwf_complex) * 1           );
    memset(pkcf->alpha, 0, (pkcf->f_cols) * (pkcf->f_rows) * sizeof(float        ) * 1           );
    memset(pkcf->xf_md, 0, (pkcf->f_cols) * (pkcf->f_rows) * sizeof(fftwf_complex) * pkcf->f_chan);

    /* create # of feature fft2 plans */
    for (int i = 0; i < pkcf->f_chan; i++)
    {
        pkcf->plan_xf[i] = fftwf_plan_dft_r2c_2d(
            pkcf->f_cols, 
            pkcf->f_rows, 
            pkcf->xf_tm + ((pkcf->f_cols) * (pkcf->f_rows        )) * i, 
            pkcf->xf_fq + ((pkcf->f_cols) * (pkcf->f_rows / 2 + 1)) * i, 
            FFTW_ESTIMATE
            );
    }

    pkcf->plan_zb = fftwf_plan_dft_c2r_2d(
            pkcf->f_cols, 
            pkcf->f_rows, 
            pkcf->zf, 
            pkcf->response, 
            FFTW_ESTIMATE
            );

    pkcf->feature_norm_ratio = 1.0 / ((float)(pkcf->f_cols * pkcf->f_rows * pkcf->f_chan));

    /* initialize position */
    pkcf->scale_vert  = 1.0;
    pkcf->scale_horiz = 1.0;
    pkcf->pos = *pbbox;
    
    /* prepare labels & cos win */
    pkcf->labels  = gaussian_shaped_labels(0.7289, pkcf->f_rows, pkcf->f_cols);
    pkcf->cos_win = cosine_window_function(pkcf->f_rows, pkcf->f_cols);
    kcf_fft2_label(pkcf, pkcf->labels.memptr());

    /* next update is first update */
    pkcf->first_update = 1;
    pkcf->factor       = 0.05;
    pkcf->lamda        = 0.0001;
    return (0);
};

void kcf_deinitialize(kcf_t * pkcf)
{
    _aligned_free(pkcf->xf_tm);
    _aligned_free(pkcf->xf_fq);
    _aligned_free(pkcf->xf_md);

    _aligned_free(pkcf->yf      );
    _aligned_free(pkcf->zf      );
    _aligned_free(pkcf->kf      );
    _aligned_free(pkcf->alpha   );
    _aligned_free(pkcf->response);

    for (int i = 0; i < pkcf->f_chan; i++)
    {
        fftwf_destroy_plan(pkcf->plan_xf[i]);
    }
    fftwf_destroy_plan(pkcf->plan_zb);
};

kcf_t * kcf_alloc()
{
    return (new kcf_t);
}

void kcf_free(kcf_t * pkcf)
{
    delete pkcf;
}

static void kcf_get_features(kcf_t * pkcf, arma::fmat & img, arma::fmat & win)
{
    FHoG::extract(img.memptr(), img.n_rows, img.n_cols, pkcf->xf_tm);

    float * phoga = pkcf->xf_tm;
    float * phogb = phoga;
    for (int l = 0; l < pkcf->f_chan; l++)
    {
        float * pwin = win.memptr();
        for (int i = 0; i < win.n_rows * win.n_cols; i++)
        {
            *phogb++ = (*phoga++) * (*pwin++);
        }
    }
}

static void kcf_fft2_features(kcf_t * pkcf)
{
    for (int i = 0; i < pkcf->f_chan; i++)
    {
        fftwf_execute(pkcf->plan_xf[i]);
    }
}

static void kcf_linear_correlation_kf(kcf_t * pkcf)
{
    fftwf_complex * pxffq = pkcf->xf_fq;
    {
        fftwf_complex * xf = pkcf->kf;
        for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
        {
            (*xf)[0] = ((*pxffq)[0] * (*pxffq)[0]) + 
                       ((*pxffq)[1] * (*pxffq)[1]) ;

            ++pxffq; ++xf;
        } 
    }

    for (int l = 1; l < pkcf->f_chan; l++)
    {
        fftwf_complex * xf = pkcf->kf;
        for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
        {
            (*xf)[0] = ((*pxffq)[0] * (*pxffq)[0]) + 
                       ((*pxffq)[1] * (*pxffq)[1]) + 
                       (((*xf)[0]));

            ++pxffq; ++xf;
        }
    }

    {
        fftwf_complex * xf = pkcf->kf;
        for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
        {
            (*xf)[0] = ((*xf)[0]) * pkcf->feature_norm_ratio;
            ++xf;
        }
    }
}

static void kcf_linear_correlation_zf(kcf_t * pkcf)
{
    fftwf_complex * pxffq = pkcf->xf_fq;
    fftwf_complex * pzffq = pkcf->xf_md;

    {
        fftwf_complex * xf = pkcf->zf;
        for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
        {
            float ia = (*pxffq)[0];
            float qa = (*pxffq)[1];
            float ib = (*pzffq)[0];
            float qb = (*pzffq)[1];

            float ic = ia * ib + qa * qb;
            float qc = qa * ib - ia * qb;

            (*xf)[0] = ic;
            (*xf)[1] = qc;

            ++pxffq; ++pzffq; ++xf;
        } 
    }

    for (int l = 1; l < pkcf->f_chan; l++)
    {
        fftwf_complex * xf = pkcf->zf;
        for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
        {
            float ia = (*pxffq)[0];
            float qa = (*pxffq)[1];
            float ib = (*pzffq)[0];
            float qb = (*pzffq)[1];

            float ic = ia * ib + qa * qb;
            float qc = qa * ib - ia * qb;

            (*xf)[0] += ic;
            (*xf)[1] += qc;
            
            ++pxffq; ++pzffq; ++xf;
        } 
    }

    {
        fftwf_complex * xf = pkcf->zf;
        float         * af = pkcf->alpha;

        for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
        {
            (*xf)[0] = ((*xf)[0]) * (af[0]) * pkcf->feature_norm_ratio;
            (*xf)[1] = ((*xf)[1]) * (af[0]) * pkcf->feature_norm_ratio;

            ++xf; ++af;
        }
    }
}

static void kcf_update_alpha(kcf_t * pkcf, float factor, float lamada)
{
    fftwf_complex * yf = pkcf->yf;
    fftwf_complex * xf = pkcf->kf;
    float         * af = pkcf->alpha;
    float         * bf = af;

    for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
    {
        float a = (*yf)[0] / (((*xf)[0]) + lamada);
        (*af++) = (1 - factor) * (*bf++) + factor * a;

        ++xf; ++yf;
    }
};

static void kcf_update_xf(kcf_t * pkcf, float factor)
{
    fftwf_complex * xm = pkcf->xf_md;
    fftwf_complex * xf = pkcf->xf_fq;

    for (int l = 0; l < pkcf->f_chan; l++)
    {
        for (int i = 0; i < (pkcf->f_rows/2+1) * (pkcf->f_cols); i++)
        {
            (*xm)[0] = (1 - factor) * ((*xm)[0]) + factor * ((*xf)[0]);
            (*xm)[1] = (1 - factor) * ((*xm)[1]) + factor * ((*xf)[1]);

            ++xf; ++xm;
        }
    }
};

static void kcf_predict_ifft2(kcf_t * pkcf)
{
    fftwf_execute(pkcf->plan_zb);

    /* code translated from matlab, index from 1 */
    float max_val = -99999.0;

    int vert_delta, horiz_delta;
    float * presp = pkcf->response;
    for (int j = 1; j <= pkcf->f_cols; j++)
    {
        for (int i = 1; i <= pkcf->f_rows; i++)
        {
            if (presp[0] > max_val)
            {
                max_val     = presp[0];
                vert_delta  = i;
                horiz_delta = j; 
            }
            ++presp;
        }
    }

    if (vert_delta  > (pkcf->f_rows / 2)) vert_delta  -= pkcf->f_rows;
    if (horiz_delta > (pkcf->f_cols / 2)) horiz_delta -= pkcf->f_cols;

    /* predicted position */
    pkcf->pos.t = pkcf->pos.t + pkcf->cell_size * (vert_delta  - 1) * pkcf->scale_vert;
    pkcf->pos.b = pkcf->pos.b + pkcf->cell_size * (vert_delta  - 1) * pkcf->scale_vert;
    pkcf->pos.l = pkcf->pos.l + pkcf->cell_size * (horiz_delta - 1) * pkcf->scale_horiz;
    pkcf->pos.r = pkcf->pos.r + pkcf->cell_size * (horiz_delta - 1) * pkcf->scale_horiz;
};

void kcf_predict(kcf_t * pkcf, arma::fmat & patch, bbox_t * pbox)
{
    kcf_get_features         (pkcf, patch, pkcf->cos_win);
    kcf_fft2_features        (pkcf                      );
    kcf_linear_correlation_zf(pkcf                      );
    kcf_predict_ifft2        (pkcf                      );

    /* return predicted pos */
    *pbox =  pkcf->pos;
}

void kcf_update(kcf_t * pkcf, arma::fmat & patch)
{
    float factor = (pkcf->first_update) ? 1.0 : (pkcf->factor);
    float lamda  = pkcf->lamda;

    pkcf->first_update = 0;

    kcf_get_features         (pkcf, patch, pkcf->cos_win);
    kcf_fft2_features        (pkcf                      );
    kcf_linear_correlation_kf(pkcf                      );
    kcf_update_alpha         (pkcf, factor, lamda       );
    kcf_update_xf            (pkcf, factor              );
}

void tracker_predict(void * ptracker, float * rgb, bbox_t * pbox)
{
    kcf_t * pkcf = (kcf_t *)ptracker;
    arma::fmat patch(rgb, pkcf->rows, pkcf->cols, false);
    kcf_predict((kcf_t *)ptracker, patch, pbox);
}

void tracker_update(void * ptracker, float * rgb, bbox_t * pbox)
{
    kcf_t * pkcf = (kcf_t *)ptracker;

    arma::fmat patch(rgb, pkcf->rows, pkcf->cols, false);

    /////////////////////////////////////////////////////////////////////////
    /* update position & scaling information */
    pkcf->pos         = *pbox;
    pkcf->scale_horiz = ((float)(pbox->r - pbox->l + 1)) / ((float)pkcf->cols);
    pkcf->scale_vert  = ((float)(pbox->b - pbox->t + 1)) / ((float)pkcf->rows);
    /////////////////////////////////////////////////////////////////////////

    kcf_update((kcf_t *)ptracker, patch);
}

void tracker_delete(void * ptracker)
{
    kcf_deinitialize((kcf_t *)ptracker);
    kcf_free((kcf_t *)ptracker);
}

void * tracker_new(bbox_t * pbox)
{
    kcf_t * pkcf = kcf_alloc();

    kcf_initialize(pkcf, pbox, 4);

    return ((void *)pkcf);
}

#if (0)
void feature_print(const float * yf, int rows, int cols)
{
    for (int i = 0; i < rows; i++ )
    {
        for (int j = 0; j < cols; j++ )
        {
            printf ("%12.8f  ", yf[j * rows + i]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void fftw3_complex_print(const fftwf_complex * yf, int rows, int cols)
{
    for (int i = 0; i < rows; i++ )
    {
        for (int j = 0; j < cols; j++ )
        {
            printf ("%6.4f  ", yf[j * rows + i][0]);
        }
        printf("\n");
    }
    printf("\n\n");
}
#endif