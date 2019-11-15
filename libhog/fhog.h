/*
    - c++ wrapper for the piotr toolbox
    Created by Tomas Vojir, 2014
*/


#ifndef FHOG_HEADER_7813784354687
#define FHOG_HEADER_7813784354687

#include <armadillo>
#include "gradientMex.h"

class FHoG
{
public:
    static float * extract(float * I, int h, int w, float * H, int bin_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2)
    {
        float * M = (float *)_aligned_malloc(h * w * 4 * sizeof(float), 16);
        float * O = M + h * w * 2;

        gradMag(
            I, M, O, 
            h, w, 
            1, true
            );

        int n_chns = n_orients * 3 + 5;
        int hb = h / bin_size, 
            wb = w / bin_size;

        memset(H, 0, hb * wb * n_chns * sizeof(float));

        fhog(M, O, H, h, w, bin_size, n_orients, soft_bin, clip);

        _aligned_free(M);

        return (H);
    }
};

#endif //FHOG_HEADER_7813784354687