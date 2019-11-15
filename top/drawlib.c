#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <stdlib.h>
#include <malloc.h>

/* 1280 * 720 */
#define PIXEL_AT(y, x)   ( ((y) << 12) - ((y) << 8) + ((x) << 2) - (x) )
#define SCREEN_WIDTH     ( 3840 )

#define abs(x)  (((x) < 0) ? (-(x)) : (x))
#define sgn(x)  ( ((x) < 0) ? (-1) : (((x) > 0) ? (+1) : (0)) )

/**************************************************************************
 *  line_fast                                                             *
 *    draws a line using Bresenham's line-drawing algorithm, which uses   *
 *    no multiplication or division.                                      *
 **************************************************************************/

void drawLine(
   uint8_t * fbuf,
   int32_t   x1,
   int32_t   y1,
   int32_t   x2,
   int32_t   y2,
   uint8_t   R,
   uint8_t   G,
   uint8_t   B
)
{
   int i,
      dx, dy,
      sdx, sdy,
      dxabs, dyabs,
      x, y,
      px, py;

   dx = x2 - x1;      /* the horizontal distance of the line */
   dy = y2 - y1;      /* the vertical distance of the line */

   dxabs = abs(dx);
   dyabs = abs(dy);
   sdx = sgn(dx);
   sdy = sgn(dy);
   x = dyabs >> 1;
   y = dxabs >> 1;

   px = x1;
   py = y1;

   fbuf[PIXEL_AT(py, px) + 0] = R;
   fbuf[PIXEL_AT(py, px) + 1] = G;
   fbuf[PIXEL_AT(py, px) + 2] = B;

   if (dxabs >= dyabs) /* the line is more horizontal than vertical */
   {
      for (i = 0; i < dxabs; i++)
      {
         y += dyabs;
         if (y >= dxabs)
         {
            y -= dxabs;
            py += sdy;
         }
         px += sdx;

         fbuf[PIXEL_AT(py, px) + 0] = R;
         fbuf[PIXEL_AT(py, px) + 1] = G;
         fbuf[PIXEL_AT(py, px) + 2] = B;
      }
   }
   else /* the line is more vertical than horizontal */
   {
      for (i = 0; i < dyabs; i++)
      {
         x += dxabs;
         if (x >= dyabs)
         {
            x -= dyabs;
            px += sdx;
         }
         py += sdy;

         fbuf[PIXEL_AT(py, px) + 0] = R;
         fbuf[PIXEL_AT(py, px) + 1] = G;
         fbuf[PIXEL_AT(py, px) + 2] = B;
      }
   }
}

/**************************************************************************
 *  rect_fast                                                             *
 *    Draws a rectangle by drawing all lines by itself.                   *
 **************************************************************************/

void drawRect(
   uint8_t * fbuf,
   int32_t   left,
   int32_t   top,
   int32_t   right,
   int32_t   bottom,
   uint32_t  RGB
)
{
   uint8_t   R = (RGB >> 16) & 0xff;
   uint8_t   G = (RGB >> 8) & 0xff;
   uint8_t   B = (RGB >> 0) & 0xff;

   int32_t temp;

   if (top > bottom)
   {
      temp = top;
      top = bottom;
      bottom = temp;
   }

   if (left > right)
   {
      temp = left;
      left = right;
      right = temp;
   }

   int32_t top_offset = PIXEL_AT(top, left);
   int32_t bottom_offset = PIXEL_AT(bottom, left);

   uint8_t * ptop = fbuf + top_offset;
   uint8_t * pbot = fbuf + bottom_offset;

   for (int i = left; i <= right; i++)
   {
      *ptop++ = R; *ptop++ = G; *ptop++ = B;
      *pbot++ = R; *pbot++ = G; *pbot++ = B;
   }

   int32_t left_offset = PIXEL_AT(top, left); // (top << 8) + (top << 6);
   int32_t right_offset = PIXEL_AT(top, right); // (bottom << 8) + (bottom << 6);

   uint8_t * pleft = fbuf + left_offset;
   uint8_t * prght = fbuf + right_offset;

   for (int i = top; i <= bottom; i++)
   {
      pleft[0] = R; pleft[1] = G; pleft[2] = B;
      prght[0] = R; prght[1] = G; prght[2] = B;

      pleft += SCREEN_WIDTH; prght += SCREEN_WIDTH;
   }
}

void copyImage(
   uint8_t * pdst,
   uint8_t * pbuf,
   int32_t   left,
   int32_t   top,
   int32_t   right,
   int32_t   bottom
)
{
   int32_t temp;

   if (top > bottom)
   {
      temp = top;
      top = bottom;
      bottom = temp;
   }

   if (left > right)
   {
      temp = left;
      left = right;
      right = temp;
   }

   int32_t cols = right - left + 1;
   int32_t top_offset = PIXEL_AT(top, left);

   uint8_t * ptop = pbuf + top_offset;

   for (int r = top; r <= bottom; r++)
   {
      memcpy(pdst, ptop, 3 * cols);

      pdst += 3 * cols;
      ptop += SCREEN_WIDTH;
   }
};

void rgb2Gray(
   float   * pgra,
   uint8_t * prgb,
   int32_t   left,
   int32_t   top,
   int32_t   right,
   int32_t   bottom
)
{
   int32_t temp;

   if (top > bottom)
   {
      temp = top;
      top = bottom;
      bottom = temp;
   }

   if (left > right)
   {
      temp = left;
      left = right;
      right = temp;
   }

   int32_t cols = right - left + 1;
   int32_t rows = bottom - top + 1;

   int32_t top_offset = PIXEL_AT(top, left);

   uint8_t * ptop = prgb + top_offset;

   for (int r = 0; r < rows; r++)
   {
      uint8_t * psrc = ptop;
      float   * pdst = pgra;
      for (int c = 0; c < cols; c++)
      {
         uint8_t B = (*psrc++);
         uint8_t G = (*psrc++);
         uint8_t R = (*psrc++);

         pdst[r] = 0.144 * B + 0.587 * G + 0.299 * R;
         pdst += rows;
      }

      ptop += SCREEN_WIDTH;
   }
};

#if (0)
void bilinearInterpolationGray_(
   float * pdst,
   const float * psrc,
   int     rows_s,
   int     cols_s,
   int     rows_d,
   int     cols_d
)
{
   int in_rows = rows_s;
   int in_cols = cols_s;

   int out_rows = rows_d;
   int out_cols = cols_d;

   // Let S_R = R / R'        
   float S_R = ((float)in_rows) / ((float)out_rows);

   // Let S_C = C / C'
   float S_C = ((float)in_cols) / ((float)out_cols);

   // Define grid of co-ordinates in our image
   // Generate (x,y) pairs for each point in our image
   float cf[1920];
   float rf[1080];

   for (int i = 0; i < out_rows; i++)
   {
      rf[i] = i * S_R;
   }

   for (int i = 0; i < out_cols; i++)
   {
      cf[i] = i * S_C;
   }

   // Let r = floor(rf) and c = floor(cf)
   int r[1080];
   int c[1920];

   for (int i = 0; i < out_rows; i++)
   {
      r[i] = (int)(rf[i]);
   }

   for (int i = 0; i < out_cols; i++)
   {
      c[i] = (int)(cf[i]);
   }

   /* Any values out of range, cap */
   for (int i = out_rows - 1; i >= 0; i--)
   {
      if (r[i] < (out_rows - 1)) break;

      r[i] = (out_rows - 2);
   }

   for (int i = out_cols - 1; i >= 0; i--)
   {
      if (c[i] < (out_cols - 1)) break;

      c[i] = (out_cols - 2);
   }

   // Let delta_R = rf - r and delta_C = cf - c
   float delta_R[1080];
   float delta_C[1920];

   for (int i = 0; i < out_rows; i++)
   {
      delta_R[i] = rf[i] - r[i];
   }

   for (int i = 0; i < out_cols; i++)
   {
      delta_C[i] = cf[i] - c[i];
   }

   /* interpolation */
   for (int j = 0; j < out_cols; j++)
   {
      for (int i = 0; i < out_rows; i++)
      {
         int base = c[j] * in_rows + r[i];
         float xa = psrc[base];
         float xb = psrc[base + 1];
         float xc = psrc[base + in_rows];
         float xd = psrc[base + 1 + in_rows];

         float xf = xa * (1 - delta_R[i]) * (1 - delta_C[j]) +
            xb * (delta_R[i]) * (1 - delta_C[j]) +
            xc * (1 - delta_R[i]) * (delta_C[j]) +
            xd * (delta_R[i]) * (delta_C[j]);

         *pdst++ = xf;
      }
   }
}
#endif

typedef struct _bilinearInterpCB
{
   int   * ppos1,
         * ppos2,
         * ppos3,
         * ppos4;

   float * fracx, 
         * fracy, 
         * ifracx, 
         * ifracy;

   int     heightSource,
           widthSource,
           height,
           width;

} bilinearinterp_t;

void * bilinearInterpolationRGBSetup(
   int  heightSource,
   int  widthSource,
   int  height,
   int  width
)
{
   bilinearinterp_t * pcb = (bilinearinterp_t *)malloc(sizeof(bilinearinterp_t) + 8);

   // unsigned int start = clock();
   pcb->ppos1  = (int   *)_aligned_malloc(sizeof(int  ) * height * width + 32, 16);
   pcb->ppos2  = (int   *)_aligned_malloc(sizeof(int  ) * height * width + 32, 16);
   pcb->ppos3  = (int   *)_aligned_malloc(sizeof(int  ) * height * width + 32, 16);
   pcb->ppos4  = (int   *)_aligned_malloc(sizeof(int  ) * height * width + 32, 16);

   pcb->fracx  = (float *)_aligned_malloc(sizeof(float) * height * width + 32, 16);
   pcb->fracy  = (float *)_aligned_malloc(sizeof(float) * height * width + 32, 16);
   pcb->ifracx = (float *)_aligned_malloc(sizeof(float) * height * width + 32, 16);
   pcb->ifracy = (float *)_aligned_malloc(sizeof(float) * height * width + 32, 16);

   pcb->heightSource = heightSource;
   pcb->height       = height;
   pcb->widthSource  = widthSource;
   pcb->width        = width;

   float xs = ((float)widthSource) / ((float)width);
   float ys = ((float)heightSource) / ((float)height);

   float   fracx, fracy, ifracx, ifracy, sx, sy, l0, l1, rf, gf, bf;

   int     c, x0, x1, y0, y1;
   uint8_t c1a, c1r, c1g, c1b,
           c2a, c2r, c2g, c2b,
           c3a, c3r, c3g, c3b,
           c4a, c4r, c4g, c4b;

   uint8_t a, r, g, b;

   // Bilinear
   int dstIdx = 0;

   for (int y = 0; y < height; y++)
   {
      sy = y * ys;
      y0 = (int)sy;

      fracy = sy - y0;
      ifracy = 1.0f - fracy;

      y1 = y0 + 1;
      if (y1 >= heightSource)
      {
         y1 = y0;
      }

      int Y0_base = y0 * widthSource * 3;
      int Y1_base = y1 * widthSource * 3;

      for (int x = 0; x < width; x++)
      {
         sx = x * xs;
         x0 = (int)sx;

         // Calculate coordinates of the 4 interpolation points
         fracx = sx - x0;
         ifracx = 1.0f - fracx;

         x1 = x0 + 1;
         if (x1 >= widthSource)
         {
            x1 = x0;
         }

         pcb->ppos1[dstIdx] = Y0_base + x0 * 3;
         pcb->ppos2[dstIdx] = Y0_base + x1 * 3;
         pcb->ppos3[dstIdx] = Y1_base + x0 * 3;
         pcb->ppos4[dstIdx] = Y1_base + x1 * 3;

         pcb->fracx[dstIdx]  = fracx;
         pcb->fracy[dstIdx]  = fracy;

         pcb->ifracx[dstIdx] = ifracx;
         pcb->ifracy[dstIdx] = ifracy;

         ++dstIdx;
      }
   }

   return (void *)(pcb);
}

void bilinearInterpolationRGBExecute(
   void          * pcb_,
   uint8_t       * pdst,
   const uint8_t * psrc
)
{
   bilinearinterp_t * pcb = (bilinearinterp_t *)(pcb_);

   int srcIdx = 0;

   for (int y = 0; y < pcb->height; y++)
   {
      for (int x = 0; x < pcb->width; x++)
      {
         int B1 = pcb->ppos1[srcIdx];
         int B2 = pcb->ppos2[srcIdx];
         int B3 = pcb->ppos3[srcIdx];
         int B4 = pcb->ppos4[srcIdx];

         float fracx  = pcb->fracx [srcIdx];
         float fracy  = pcb->fracy [srcIdx];
         float ifracx = pcb->ifracx[srcIdx];
         float ifracy = pcb->ifracy[srcIdx];

         ++srcIdx;

         // Read source color
         float c1r = (float)psrc[B1 + 0];
         float c1g = (float)psrc[B1 + 1];
         float c1b = (float)psrc[B1 + 2];

         float c2r = (float)psrc[B2 + 0];
         float c2g = (float)psrc[B2 + 1];
         float c2b = (float)psrc[B2 + 2];

         float c3r = (float)psrc[B3 + 0];
         float c3g = (float)psrc[B3 + 1];
         float c3b = (float)psrc[B3 + 2];

         float c4r = (float)psrc[B4 + 0];
         float c4g = (float)psrc[B4 + 1];
         float c4b = (float)psrc[B4 + 2];

         float l0, l1, rf, gf, bf;

         {
            // Red
            l0 = ifracx * c1r + fracx * c2r;
            l1 = ifracx * c3r + fracx * c4r;
            rf = ifracy * l0  + fracy * l1;

            // Green
            l0 = ifracx * c1g + fracx * c2g;
            l1 = ifracx * c3g + fracx * c4g;
            gf = ifracy * l0  + fracy * l1;

            // Blue
            l0 = ifracx * c1b + fracx * c2b;
            l1 = ifracx * c3b + fracx * c4b;
            bf = ifracy * l0  + fracy * l1;

            *pdst++ = ((uint8_t)(rf));
            *pdst++ = ((uint8_t)(gf));
            *pdst++ = ((uint8_t)(bf));
         }
      }
   }
}

void bilinearInterpolationRGBDestroy(
   void * pcb_
)
{
   bilinearinterp_t * pcb = (bilinearinterp_t *)(pcb_);

   _aligned_free(pcb->ppos1);
   _aligned_free(pcb->ppos2);
   _aligned_free(pcb->ppos3);
   _aligned_free(pcb->ppos4);

   _aligned_free(pcb->fracx );
   _aligned_free(pcb->fracy );
   _aligned_free(pcb->ifracx);
   _aligned_free(pcb->ifracy);

   free(pcb);
}

void bilinearInterpolationGray(
         float * pdst,
   const float * psrc,
   int           heightSource,
   int           widthSource,
   int           height,
   int           width
)
{
   float xs = ((float)widthSource) / ((float)width);
   float ys = ((float)heightSource) / ((float)height);

   float   fracx, fracy, ifracx, ifracy, sx, sy, l0, l1, rf, gf, bf;

   int     c, x0, x1, y0, y1;
   float   c1a, c1r, c1g, c1b,
           c2a, c2r, c2g, c2b,
           c3a, c3r, c3g, c3b,
           c4a, c4r, c4g, c4b;

   float   a, r, g, b;

   ///////////////////////////////////////////////////////////
   int     x0_    [SCREEN_WIDTH / 3 + 16], 
           x1_    [SCREEN_WIDTH / 3 + 16];
   float   fracx_ [SCREEN_WIDTH / 3 + 16], 
           ifracx_[SCREEN_WIDTH / 3 + 16];

   for (int x = 0; x < width; x++)
   {
      sx = x * xs;
      x0 = (int)sx;

      // Calculate coordinates of the 4 interpolation points
      fracx  = sx - x0;
      ifracx = 1.0f - fracx;

      x1 = x0 + 1;
      if (x1 >= widthSource)
      {
         x1 = x0;
      }

      x0_[x] = x0;
      x1_[x] = x1;

      fracx_[x]  = fracx;
      ifracx_[x] = ifracx;
   }
   ///////////////////////////////////////////////////////////


   ///////////////////////////////////////////////////////////
   // Bilinear
   for (int y = 0; y < height; y++)
   {
      sy = y * ys;
      y0 = (int)sy;

      fracy = sy - y0;
      ifracy = 1.0f - fracy;

      y1 = y0 + 1;
      if (y1 >= heightSource)
      {
         y1 = y0;
      }

      int Y0_base = y0 * widthSource;
      int Y1_base = y1 * widthSource;
      for (int x = 0; x < width; x++)
      {

         x0     = x0_    [x];    
         x1     = x1_    [x];
         fracx  = fracx_ [x]; 
         ifracx = ifracx_[x];

         // Read source color
         c1r = psrc[Y0_base + x0 + 0];
         c2r = psrc[Y0_base + x1 + 0];
         c3r = psrc[Y1_base + x0 + 0];
         c4r = psrc[Y1_base + x1 + 0];

         {
            // Red
            l0 = ifracx * c1r + fracx * c2r;
            l1 = ifracx * c3r + fracx * c4r;
            rf = ifracy * l0  + fracy * l1;

            *pdst++ = rf;
         }
      }
   }
   ///////////////////////////////////////////////////////////
}

void bilinearInterpolationRGB48(
   uint8_t       * pdst,
   const uint8_t * psrc,
   int             heightSource,
   int             widthSource,
   int             strideSource,
   int             depthDest
)
{
   int   X0[48],     
         X1[48];

   float fracx_ [48], 
         ifracx_[48];

   float xs = ((float)widthSource ) / ((float)48.0);
   float ys = ((float)heightSource) / ((float)48.0);

   float   fracx, fracy, ifracx, ifracy, sx, sy, l0, l1, rf, gf, bf;

   int     c,   
           x0,  x1,  
           y0,  y1;

   uint8_t c1a, c1r, c1g, c1b,
           c2a, c2r, c2g, c2b,
           c3a, c3r, c3g, c3b,
           c4a, c4r, c4g, c4b;

   //////////////////////////////////////////////////////////////////
   for (int x = 0; x < 48; x++)
   {
      sx = x * xs;
      x0 = (int)sx;

      // Calculate coordinates of the 4 interpolation points
      fracx  = sx - x0;
      ifracx = 1.0f - fracx;

      x1 = x0 + 1;
      if (x1 >= widthSource)
      {
         x1 = x0;
      }

      X0[x] = x0;
      X1[x] = x1;

      fracx_ [x] = fracx;
      ifracx_[x] = ifracx;
   }
   //////////////////////////////////////////////////////////////////

   //////////////////////////////////////////////////////////////////
   for (int y = 0; y < 48; y++)
   {
      sy = y * ys;
      y0 = (int)sy;

      fracy = sy - y0;
      ifracy = 1.0f - fracy;

      y1 = y0 + 1;
      if (y1 >= heightSource)
      {
         y1 = y0;
      }

      int Y0_base = y0 * strideSource;
      int Y1_base = y1 * strideSource;

      for (int x = 0; x < 48; x++)
      {
         x0 = X0[x];
         x1 = X1[x];
         fracx  = fracx_ [x];
         ifracx = ifracx_[x];

         int B1 = Y0_base + x0 * 3;
         int B2 = Y0_base + x1 * 3;
         int B3 = Y1_base + x0 * 3;
         int B4 = Y1_base + x1 * 3;

         // Read source color
         float c1r = (float)psrc[B1 + 0];
         float c1g = (float)psrc[B1 + 1];
         float c1b = (float)psrc[B1 + 2];

         float c2r = (float)psrc[B2 + 0];
         float c2g = (float)psrc[B2 + 1];
         float c2b = (float)psrc[B2 + 2];

         float c3r = (float)psrc[B3 + 0];
         float c3g = (float)psrc[B3 + 1];
         float c3b = (float)psrc[B3 + 2];

         float c4r = (float)psrc[B4 + 0];
         float c4g = (float)psrc[B4 + 1];
         float c4b = (float)psrc[B4 + 2];

         float l0, l1, rf, gf, bf;

         {
            // Red
            l0 = ifracx * c1r + fracx * c2r;
            l1 = ifracx * c3r + fracx * c4r;
            rf = ifracy * l0  + fracy * l1;

            // Green
            l0 = ifracx * c1g + fracx * c2g;
            l1 = ifracx * c3g + fracx * c4g;
            gf = ifracy * l0  + fracy * l1;

            // Blue
            l0 = ifracx * c1b + fracx * c2b;
            l1 = ifracx * c3b + fracx * c4b;
            bf = ifracy * l0  + fracy * l1;

            *pdst++ = ((uint8_t)(rf));
            *pdst++ = ((uint8_t)(gf));
            *pdst++ = ((uint8_t)(bf));

            pdst += (depthDest - 3);
         }
      }
   }
}

void bilinearInterpolationRGB24(
   uint8_t       * pdst,
   const uint8_t * psrc,
   int             heightSource,
   int             widthSource,
   int             strideSource,
   int             depthDest
)
{
   int   X0[24],     
         X1[24];

   float fracx_ [24],
         ifracx_[24];

   float xs = ((float)widthSource ) / ((float)24.0);
   float ys = ((float)heightSource) / ((float)24.0);

   float   fracx, fracy, ifracx, ifracy, sx, sy, l0, l1, rf, gf, bf;

   int     c,   
           x0,  x1,  
           y0,  y1;

   uint8_t c1a, c1r, c1g, c1b,
           c2a, c2r, c2g, c2b,
           c3a, c3r, c3g, c3b,
           c4a, c4r, c4g, c4b;

   //////////////////////////////////////////////////////////////////
   for (int x = 0; x < 24; x++)
   {
      sx = x * xs;
      x0 = (int)sx;

      // Calculate coordinates of the 4 interpolation points
      fracx  = sx - x0;
      ifracx = 1.0f - fracx;

      x1 = x0 + 1;
      if (x1 >= widthSource)
      {
         x1 = x0;
      }

      X0[x] = x0;
      X1[x] = x1;

      fracx_ [x] = fracx;
      ifracx_[x] = ifracx;
   }
   //////////////////////////////////////////////////////////////////

   //////////////////////////////////////////////////////////////////
   for (int y = 0; y < 24; y++)
   {
      sy = y * ys;
      y0 = (int)sy;

      fracy = sy - y0;
      ifracy = 1.0f - fracy;

      y1 = y0 + 1;
      if (y1 >= heightSource)
      {
         y1 = y0;
      }

      int Y0_base = y0 * strideSource;
      int Y1_base = y1 * strideSource;

      for (int x = 0; x < 24; x++)
      {
         x0     = X0[x];
         x1     = X1[x];
         fracx  = fracx_ [x];
         ifracx = ifracx_[x];

         int B1 = Y0_base + x0 * 3;
         int B2 = Y0_base + x1 * 3;
         int B3 = Y1_base + x0 * 3;
         int B4 = Y1_base + x1 * 3;

         // Read source color
         float c1r = (float)psrc[B1 + 0];
         float c1g = (float)psrc[B1 + 1];
         float c1b = (float)psrc[B1 + 2];

         float c2r = (float)psrc[B2 + 0];
         float c2g = (float)psrc[B2 + 1];
         float c2b = (float)psrc[B2 + 2];

         float c3r = (float)psrc[B3 + 0];
         float c3g = (float)psrc[B3 + 1];
         float c3b = (float)psrc[B3 + 2];

         float c4r = (float)psrc[B4 + 0];
         float c4g = (float)psrc[B4 + 1];
         float c4b = (float)psrc[B4 + 2];

         float l0, l1, rf, gf, bf;

         {
            // Red
            l0 = ifracx * c1r + fracx * c2r;
            l1 = ifracx * c3r + fracx * c4r;
            rf = ifracy * l0  + fracy * l1;

            // Green
            l0 = ifracx * c1g + fracx * c2g;
            l1 = ifracx * c3g + fracx * c4g;
            gf = ifracy * l0  + fracy * l1;

            // Blue
            l0 = ifracx * c1b + fracx * c2b;
            l1 = ifracx * c3b + fracx * c4b;
            bf = ifracy * l0  + fracy * l1;

            *pdst++ = ((uint8_t)(rf));
            *pdst++ = ((uint8_t)(gf));
            *pdst++ = ((uint8_t)(bf));

            pdst += (depthDest - 3);
         }
      }
   }
}

/**************************************************************************
 *  rect_fill                                                             *
 *    Draws and fills a rectangle.                                        *
 **************************************************************************/

void fillRect(
   uint8_t * fbuf,
   int32_t   left,
   int32_t   top,
   int32_t   right,
   int32_t   bottom,
   uint8_t   R,
   uint8_t   G,
   uint8_t   B
)
{
   int32_t temp;

   if (top > bottom)
   {
      temp = top;
      top = bottom;
      bottom = temp;
   }
   if (left > right)
   {
      temp = left;
      left = right;
      right = temp;
   }

   int32_t top_offset = PIXEL_AT(top, left); // (top << 8) + (top << 6);
   int32_t bottom_offset = PIXEL_AT(bottom, left); // (bottom << 8) + (bottom << 6);

   uint8_t * ptop_a = fbuf + top_offset;
   uint8_t * pbot_a = fbuf + bottom_offset;

   for (int l = top; l <= bottom; l++)
   {
      uint8_t * ptop = ptop_a;
      uint8_t * pbot = pbot_a;

      for (int i = left; i <= right; i++)
      {
         *ptop++ = R; *ptop++ = G; *ptop++ = B;
         *pbot++ = R; *pbot++ = G; *pbot++ = B;
      }

      ptop_a += (SCREEN_WIDTH);
      pbot_a += (SCREEN_WIDTH);
   }
}