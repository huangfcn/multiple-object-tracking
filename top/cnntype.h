#ifndef __CNN_TYPE_H__
#define __CNN_TYPE_H__
#include <malloc.h>

#define MTCNN_IMGW   (1280)
#define MTCNN_IMGH   ( 720)

#define CNN_BCHSIZ   (   8)

typedef float cnn_type_t;
typedef float acc_type_t;

/* linux / mingw */
// #define _aligned_malloc(x, s) (aligned_alloc((x), (s)))
// #define _aligned_free(x)      (free(x))

#if defined (_MSC_VER)
#define BEGIN_ALIGNED(x) __declspec(align(x))
#define DECLA_ALIGNED(x)

#define BEGIN_NOINLINE    __declspec(noinline)
#define DECLA_NOINLINE

#define __builtin_assume_aligned(v, x)    (v)
#define restrict __restrict
#endif

#if defined (__GNUC__)
#define BEGIN_ALIGNED(x) 
#define DECLA_ALIGNED(x) __attribute__ ((aligned(x)))

#define BEGIN_NOINLINE
#define DECLA_NOINLINE __attribute__ ((noinline))
#endif

typedef struct _bbox_pos_s
{
  int l, t, b, r; 
  int type;
  float score;
} bbox_t;

typedef struct _bbox_chain_s
{
  int nbox; 
  bbox_t bbox[128];
} bbox_chain_t;

typedef struct
{
   float obj_thresh;
   float nms_thresh;
   int   anchors[18];
} yolo3_options_t;

#define mydataFmt cnn_type_t

struct orderScore
{
    cnn_type_t score;
    int        oriOrder;
};

typedef struct 
{
    cnn_type_t score;
    int        index;
} score_t;

#endif  // __CNN_TYPE_H__
