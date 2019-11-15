#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

#include "cnntype.h"

#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>
#include <semaphore.h>

#define MAX_OBJECTS_PER_FRAME    ( 256)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* API of tensorgpu.dll                                                                                                  */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" int tensorStartup(const char * graph_path, const char * input_layer, const char * output_layer[3]);

/* single run */
extern "C" int tensorRunS(
	int tensor_height, 
	int tensor_width, 
	int image_height, 
	int image_width, 
	unsigned char * pimg, 
	bbox_chain_t * pbbox, 
	yolo3_options_t * popt
);

/* batch run */
extern "C" int tensorRunB(
	int batchSize,
	int tensor_height,
	int tensor_width,
	int * image_height,
	int * image_width,
	unsigned char **  pimgbuf,
	bbox_chain_t  **  pbbox,
	yolo3_options_t * popt
);

extern "C" int tensorCleanup();
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static VideoCapture capture;

// #define KCF_TRACKER     (1   )
#define MTCNN_THREADS   (1   )

#define SCREEN_DIS      (1.0 / ((double)MTCNN_IMGW))

#define MAX_GPU_BATCH   (4)

static int bQuit = 0;

/* semaphore of space */
static sem_t sem_imgs_spc;
static sem_t mtx_imgs_spc;
static int   spc_rptr = 0;
static int   spc_wptr = 0;
static Mat * imgs_free[64] = {0};

static sem_t sem_imgs_cap;
static sem_t mtx_imgs_cap;
static int   cap_rptr = 0;
static int   cap_wptr = 0;
static Mat * imgs_capd[64];

static sem_t sem_imgs_prc;
static sem_t mtx_imgs_prc;
static int   prc_rptr = 0;
static int   prc_wptr = 0;
static Mat * imgs_proc[64];
static bbox_chain_t * bbox_proc[64] = {0};

static sem_t sem_imgs_trk;
static sem_t mtx_imgs_trk;
static int   trk_rptr = 0;
static int   trk_wptr = 0;
static Mat * imgs_trkd[64];

void * pthread_mtcnn_capr(void *)
{
    while (!bQuit)
    {
        /* waiting for a free space */
        sem_wait(&sem_imgs_spc);
        if (bQuit){break;};

        sem_wait(&mtx_imgs_spc);
        Mat * pimg = imgs_free[spc_rptr];
        spc_rptr = (spc_rptr + 1) & 63;
        sem_post(&mtx_imgs_spc);

        if (pimg == NULL){pimg = new Mat;}

        capture >> (*pimg);
		// imwrite("capturea.png", (*pimg));

        sem_wait(&mtx_imgs_cap);
        imgs_capd[cap_wptr] = pimg;
        cap_wptr = (cap_wptr + 1) & 63;
        sem_post(&mtx_imgs_cap);

        sem_post(&sem_imgs_cap);
    }

	return (0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct _chain_t 
{
    struct _chain_t * next;
} chain_t;

static sem_t     mtx_bbox_res;
static chain_t * bbox_free_chain_first = NULL;

bbox_chain_t * bbox_alloc()
{
    bbox_chain_t * pbbox = NULL;
    sem_wait(&mtx_bbox_res);
    if (bbox_free_chain_first == NULL)
    {
        pbbox = (bbox_chain_t *)malloc(sizeof(bbox_chain_t));
    }
    else
    {
        pbbox = (bbox_chain_t *)bbox_free_chain_first;
        bbox_free_chain_first = bbox_free_chain_first->next;
    }
    sem_post(&mtx_bbox_res);

    return (pbbox);
}

void bbox_free(bbox_chain_t * pbbox)
{
    sem_wait(&mtx_bbox_res);
    ((chain_t *)pbbox)->next = bbox_free_chain_first;
    bbox_free_chain_first = (chain_t *)pbbox;
    sem_post(&mtx_bbox_res);
}
///////////////////////////////////////////////////////////////////////////////////////////////////

void * pthread_mtcnn_run(void *)
{
    yolo3_options_t opts = {
        0.50,
        0.45,
        {55, 69, 75, 234, 133, 240, 136, 129, 142, 363, 203, 290, 228, 184, 285, 359, 341, 260}
    };
	int   image_width[MAX_GPU_BATCH];
	int   image_height[MAX_GPU_BATCH];

	int nStartFrames = 0;

	/* image width & height */
	for (int i = 0; i < MAX_GPU_BATCH; i++)
	{
		image_width[i] = MTCNN_IMGW;
		image_height[i] = MTCNN_IMGH;
	}

	unsigned int nFramesProcessed = 0;

    // mtcnn found(MTCNN_IMGH, MTCNN_IMGW);
    while (!bQuit)
    {
        sem_wait(&sem_imgs_cap);
        if (bQuit){break;}

		int navail = 0;
		int nbatch = 0;

		Mat           * pimgs[MAX_GPU_BATCH];
		unsigned char * pimgbuf[MAX_GPU_BATCH];
		bbox_chain_t  * pbbox[MAX_GPU_BATCH];

        sem_wait(&mtx_imgs_cap);
		navail = (cap_wptr < cap_rptr) ? (cap_wptr + 64 - cap_rptr) : (cap_wptr - cap_rptr);
		nbatch = min(navail, MAX_GPU_BATCH);
		for (int i = 0; i < nbatch; i++)
		{
			pimgs[i] = imgs_capd[cap_rptr];
			cap_rptr = (cap_rptr + 1) & 63;
		}
		sem_post(&mtx_imgs_cap);

		/* consume more captured images */
		for (int i = 0; i < (nbatch - 1); i++)
		{
			sem_wait(&sem_imgs_cap);
		}
	
		/* setup image-bufs and bbox */
		for (int i = 0; i < (nbatch); i++)
		{
			pimgbuf[i] = pimgs[i]->data;
			pbbox[i] = bbox_alloc();
		}

        tensorRunB(nbatch, 480, 480, image_height, image_width, pimgbuf, pbbox, &opts);

        sem_wait(&mtx_imgs_prc);
		for (int i = 0; i < nbatch; i++)
		{
			//////////////////////////////////////
			/* skip detection on some frames */
			// if ((nFramesProcessed % 3) == 0)
			// {
			// 	  pbbox[i]->nbox = 0;
			// }
			// ++nFramesProcessed;
			//////////////////////////////////////

			imgs_proc[prc_wptr] = pimgs[i];
			bbox_proc[prc_wptr] = pbbox[i];
			prc_wptr = (prc_wptr + 1) & 63;
			sem_post(&mtx_imgs_prc);
		}
        sem_post(&sem_imgs_prc);
    }

	return (0);
}

void tracker_predict(void * ptracker, float * rgb, bbox_t * pbox);
void tracker_update (void * ptracker, float * rgb, bbox_t * pbox);
void tracker_delete (void * ptracker                            );
void * tracker_new  (bbox_t * pbox                              );

void assignmentoptimal(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns);

extern "C" void drawRect(
   uint8_t * fbuf,
   int32_t   left,
   int32_t   top,
   int32_t   right,
   int32_t   bottom,
   uint32_t  RGB
);

extern "C" void rgb2Gray(
   float   * pgra,
   uint8_t * prgb,
   int32_t   left,
   int32_t   top,
   int32_t   right,
   int32_t   bottom
   );

extern "C" void bilinearInterpolationGray(
          float * pdst,
    const float * psrc,
          int     rows_s,
          int     cols_s,
          int     rows_d,
          int     cols_d
    );

#define min(x, y)  (((x) < (y)) ? (x) : (y))
#define max(x, y)  (((x) > (y)) ? (x) : (y))

#define ASSIGNMENT_THRESHOLD  (0.95)
#define MAX_INVISIBLE_COUNTS  (20  )
#define MIN_AGE_COUNTS        (10  )
#define VISIBILITY_THRESHOLD  (0.60)

typedef struct tracker_helper_s
{
    uint32_t tid;
    uint32_t color;

    void * ptracker;
    void   (*predict)(void *, bbox_t *);
    void   (*update )(void *, bbox_t *);

    int age;
    int totalVisibleCount;
    int consecutiveInvisibleCount;

    bbox_t bbox;

    int    rows;
    int    cols;

    float * grayImage;
} tracker_info_t;

#define make_x_in_range(x)  (min(max(0, (x)), (MTCNN_IMGW-1)))
#define make_y_in_range(y)  (min(max(0, (y)), (MTCNN_IMGH-1)))

static uint32_t hashcolor( uint32_t a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

void * pthread_mtcnn_trkn(void *)
{
    int      ntrackers = 0;
    int      ndetected = 0;
    uint32_t tracker_id = 0;

    tracker_info_t tracker_info[MAX_OBJECTS_PER_FRAME];

    double   distance[MAX_OBJECTS_PER_FRAME * MAX_OBJECTS_PER_FRAME];
    double * pdist;
    int      assigned[MAX_OBJECTS_PER_FRAME];
    double   cost;

    int      assigned_trackers[MAX_OBJECTS_PER_FRAME];
    int      assigned_detected[MAX_OBJECTS_PER_FRAME];
    int      losted_trackers  [MAX_OBJECTS_PER_FRAME];

    while (!bQuit)
    {
        /* waiting for processed img */
        sem_wait(&sem_imgs_prc);
        if (bQuit){break;};

        sem_wait(&mtx_imgs_prc);
        Mat          * pimg  = imgs_proc[prc_rptr];
        bbox_chain_t * pdetected = bbox_proc[prc_rptr];
        prc_rptr = (prc_rptr + 1) & 63;
        sem_post(&mtx_imgs_prc);

        ndetected = pdetected->nbox;

        printf("\n\n\ndetected %d, tracking %d faces.\n", ndetected, ntrackers);
        // for (int i = 0; i < ndetected; i++)
        // {
        //     printf("pos: (%3d, %3d, %3d, %3d)\n", pdetected->bbox[i].t, pdetected->bbox[i].l, pdetected->bbox[i].b, pdetected->bbox[i].r);
        // }

        /* tracker predict face positions */
        for (int i = 0; i < ntrackers; i++)
        {
            #if defined(KCF_TRACKER)
            /* resize input image */
            rgb2Gray(
                tracker_info[i].grayImage + MTCNN_IMGH * MTCNN_IMGH, 
                pimg->data,
                tracker_info[i].bbox.l,
                tracker_info[i].bbox.t,
                tracker_info[i].bbox.r,
                tracker_info[i].bbox.b
                );

            bilinearInterpolationGray(
                tracker_info[i].grayImage,
                tracker_info[i].grayImage + MTCNN_IMGH * MTCNN_IMGH,
                tracker_info[i].bbox.b - tracker_info[i].bbox.t + 1,
                tracker_info[i].bbox.r - tracker_info[i].bbox.l + 1,
                tracker_info[i].rows,
                tracker_info[i].cols 
                );

            // {
            //     char filename[1024];
            //     sprintf(filename, "G_%03d_%03d.dat", tracker_info[i].rows, tracker_info[i].cols);
            //     FILE * fd = fopen(filename, "wb");
            //     fwrite(tracker_info[i].grayImage, sizeof(float), tracker_info[i].rows * tracker_info[i].cols, fd);
            //     fclose(fd);
            // }
            #endif

            tracker_predict(tracker_info[i].ptracker, tracker_info[i].grayImage, &(tracker_info[i].bbox));

            /* make x & y in range */
            tracker_info[i].bbox.l = make_x_in_range(tracker_info[i].bbox.l);
            tracker_info[i].bbox.r = make_x_in_range(tracker_info[i].bbox.r);
            tracker_info[i].bbox.t = make_y_in_range(tracker_info[i].bbox.t);
            tracker_info[i].bbox.b = make_y_in_range(tracker_info[i].bbox.b);

            printf("predicted: %4d: (%4d, %4d) - (%4d, %4d);\n", i, tracker_info[i].bbox.l, tracker_info[i].bbox.t, tracker_info[i].bbox.r, tracker_info[i].bbox.b);
        }

        /* calculate distance between detected bbox & tracked bbox */
        pdist = distance;
        if (ntrackers < ndetected)
        {
            for (int j = 0; j < ndetected; j++)
            {
                for (int i = 0; i < ntrackers; i++)
                {
                    int maxl = max(tracker_info[i].bbox.l, pdetected->bbox[j].l);
                    int maxt = max(tracker_info[i].bbox.t, pdetected->bbox[j].t);
                    int minr = min(tracker_info[i].bbox.r, pdetected->bbox[j].r);
                    int minb = min(tracker_info[i].bbox.b, pdetected->bbox[j].b);

                    double intersect = (minr - maxl) * (minb - maxt);
                    intersect = (intersect < 0) ? 0 : intersect;
                    double uniona = (tracker_info[i].bbox.b - tracker_info[i].bbox.t) * (tracker_info[i].bbox.r - tracker_info[i].bbox.l) +
                                    (pdetected->bbox[j].b   - pdetected->bbox[j].t  ) * (pdetected->bbox[j].r   - pdetected->bbox[j].l  ) - intersect;

                    double dista = 0.0; // - intersect / uniona;
                    // if (dista > ASSIGNMENT_THRESHOLD)
                    {
                        int centxi = (tracker_info[i].bbox.l + tracker_info[i].bbox.r) >> 1;
                        int centyi = (tracker_info[i].bbox.t + tracker_info[i].bbox.b) >> 1;

                        int centxj = (pdetected->bbox[j].l + pdetected->bbox[j].r) >> 1;
                        int centyj = (pdetected->bbox[j].t + pdetected->bbox[j].b) >> 1;

                        dista += (sqrt((centxi - centxj) * (centxi - centxj) + (centyi - centyj) * (centyi - centyj)) * SCREEN_DIS);
                    }
					if (tracker_info[i].bbox.type != (pdetected->bbox[j].type))
					{
						dista += 1.0;
					}
                    *pdist++ = dista;
                }
            }
        }
        else
        {
            for (int i = 0; i < ntrackers; i++)
            {
                for (int j = 0; j < ndetected; j++)
                {
                    int maxl = max(tracker_info[i].bbox.l, pdetected->bbox[j].l);
                    int maxt = max(tracker_info[i].bbox.t, pdetected->bbox[j].t);
                    int minr = min(tracker_info[i].bbox.r, pdetected->bbox[j].r);
                    int minb = min(tracker_info[i].bbox.b, pdetected->bbox[j].b);

                    double intersect = (minr - maxl) * (minb - maxt);
                    intersect = (intersect < 0) ? 0 : intersect;
                    double uniona = (tracker_info[i].bbox.b - tracker_info[i].bbox.t) * (tracker_info[i].bbox.r - tracker_info[i].bbox.l) +
                                    (pdetected->bbox[j].b   - pdetected->bbox[j].t  ) * (pdetected->bbox[j].r   - pdetected->bbox[j].l  ) - intersect;

                    double dista = 0.0; // 1.0 - intersect / uniona;
                    // if (dista > ASSIGNMENT_THRESHOLD)
                    {
                        int centxi = (tracker_info[i].bbox.l + tracker_info[i].bbox.r) >> 1;
                        int centyi = (tracker_info[i].bbox.t + tracker_info[i].bbox.b) >> 1;

                        int centxj = (pdetected->bbox[j].l + pdetected->bbox[j].r) >> 1;
                        int centyj = (pdetected->bbox[j].t + pdetected->bbox[j].b) >> 1;

                        dista += (sqrt((centxi - centxj) * (centxi - centxj) + (centyi - centyj) * (centyi - centyj)) * SCREEN_DIS);
                    }
					if (tracker_info[i].bbox.type != (pdetected->bbox[j].type))
					{
						dista += 1.0;
					}
                    *pdist++ = dista;
                }
            }
        }

        /* assign tracking bbox to detected bbox */
        if (ndetected && ntrackers)
        {
            if (ntrackers < ndetected)
            {
                assignmentoptimal(assigned, &cost, distance, ntrackers, ndetected);
            }
            else
            {
                assignmentoptimal(assigned, &cost, distance, ndetected, ntrackers);   
            }
        }

        for (int i = 0; i < ntrackers; i++)
        {
            assigned_trackers[i] = (-1);
        }
        for (int i = 0; i < ndetected; i++)
        {
            assigned_detected[i] = (-1);
        }

        if (ntrackers < ndetected)
        {
            for (int i = 0; i < ntrackers; i++)
            {
                int j = assigned[i];
                {
                    assigned_trackers[i] = j;
                    assigned_detected[j] = i;
                }
            }
        }
        else
        {
            for (int j = 0; j < ndetected; j++)
            {
                int i = assigned[j];
                {
                    assigned_trackers[i] = j;
                    assigned_detected[j] = i;
                }
            }
        }

        printf("assigned : ");
        for (int i = 0; i < ntrackers; i++)
        {
            printf("%4d->%4d ", i, assigned_trackers[i]);
        }
        printf("\n");

        /* update assigned trackers */
        for (int i = 0; i < ntrackers; i++)
        {
            int j = assigned_trackers[i];
            if (j < 0) continue;

            #if defined(KCF_TRACKER)
            /* resize input image */
            rgb2Gray(
                tracker_info[i].grayImage + MTCNN_IMGH * MTCNN_IMGH, 
                pimg->data,
                pdetected->bbox[j].l,
                pdetected->bbox[j].t,
                pdetected->bbox[j].r,
                pdetected->bbox[j].b
                );

            bilinearInterpolationGray(
                tracker_info[i].grayImage,
                tracker_info[i].grayImage + MTCNN_IMGH * MTCNN_IMGH,
                pdetected->bbox[j].b - 
                pdetected->bbox[j].t + 1,
                pdetected->bbox[j].r - 
                pdetected->bbox[j].l + 1,
                tracker_info[i].rows,
                tracker_info[i].cols 
                );
            #endif

            tracker_update(tracker_info[i].ptracker, tracker_info[i].grayImage, &(pdetected->bbox[j]));

            /* update tracking bbox as detected bbox */
            tracker_info[i].bbox = pdetected->bbox[j];
            tracker_info[i].totalVisibleCount++;
            tracker_info[i].age++;
            tracker_info[i].consecutiveInvisibleCount = 0;
        }

        /* update unassigned trackers */
        for (int i = 0; i < ntrackers; i++)
        {
            int j = assigned_trackers[i];
            if (j >= 0) continue;

            tracker_info[i].age++;
            tracker_info[i].consecutiveInvisibleCount++;

            #if defined(KCF_TRACKER)
            /* resize input image */
            rgb2Gray(
                tracker_info[i].grayImage + MTCNN_IMGH * MTCNN_IMGH, 
                pimg->data,
                tracker_info[i].bbox.l,
                tracker_info[i].bbox.t,
                tracker_info[i].bbox.r,
                tracker_info[i].bbox.b
                );

            bilinearInterpolationGray(
                tracker_info[i].grayImage,
                tracker_info[i].grayImage + MTCNN_IMGH * MTCNN_IMGH,
                tracker_info[i].bbox.b - 
                tracker_info[i].bbox.t + 1,
                tracker_info[i].bbox.r - 
                tracker_info[i].bbox.l + 1,
                tracker_info[i].rows,
                tracker_info[i].cols 
                );
            #endif

            tracker_update(tracker_info[i].ptracker, tracker_info[i].grayImage, &(tracker_info[i].bbox));
        }

        /* delete lost trackers */
        {
            int n = 0;
            for (int i = 0; i < ntrackers; i++)
            {
                int lost = ((tracker_info[i].age < MIN_AGE_COUNTS) && (tracker_info[i].totalVisibleCount * 5 < 3 * tracker_info[i].age)) ||
                           ((tracker_info[i].consecutiveInvisibleCount >= MAX_INVISIBLE_COUNTS));

                if (!lost)
                {
                    if (n != i) {tracker_info[n] = tracker_info[i];};
                    ++n;
                }
                else
                {
                    tracker_delete(tracker_info[i].ptracker );
                    _aligned_free (tracker_info[i].grayImage);
                }

                // printf("info %d: (%2d, %2d, %2d)\n", i, tracker_info[i].age, tracker_info[i].totalVisibleCount, tracker_info[i].consecutiveInvisibleCount);
            }

            ntrackers = n;

            // printf("\n");
        }

        /* add new trackers */
        for (int j = 0; j < ndetected; j++)
        {
            if (assigned_detected[j] >= 0){continue;};

            int i = ntrackers;

            memset(&(tracker_info[i]), 0, sizeof(tracker_info_t));
            tracker_info[i].tid       = tracker_id++;
            tracker_info[i].color     = (hashcolor(tracker_id)) & 255;
            tracker_info[i].bbox      = pdetected->bbox[j];
            tracker_info[i].ptracker  = tracker_new(&(pdetected->bbox[j]));
            tracker_info[i].grayImage = (float *)_aligned_malloc(MTCNN_IMGW * MTCNN_IMGH * sizeof(float) * 2, 16);

            /* original bbox size */
            tracker_info[i].rows     = pdetected->bbox[j].b - pdetected->bbox[j].t + 1;
            tracker_info[i].cols     = pdetected->bbox[j].r - pdetected->bbox[j].l + 1;

            #if defined(KCF_TRACKER)
            /* first update */
            rgb2Gray(
                tracker_info[i].grayImage, 
                pimg->data,
                pdetected->bbox[j].l,
                pdetected->bbox[j].t,
                pdetected->bbox[j].r,
                pdetected->bbox[j].b
                );

            tracker_update(tracker_info[i].ptracker, tracker_info[i].grayImage, &(tracker_info[i].bbox));
            #endif

            ++ntrackers;
        }

        /* add tracking rectangle */
        for (int j = 0; j < ntrackers; j++)
        {
            int i = j;
            printf("tracking : %4d: (%4d, %4d) - (%4d, %4d);\n", i, tracker_info[i].bbox.l, tracker_info[i].bbox.t, tracker_info[i].bbox.r, tracker_info[i].bbox.b);

            static uint32_t colormap[] =
            {
               0x000000, 0x800000, 0x008000, 0x808000, 0x000080, 0x800080, 
               0x008080, 0xc0c0c0, 0x808080, 0xff0000, 0x00ff00, 0xffff00, 
               0x0000ff, 0xff00ff, 0x00ffff, 0xffffff,
               0x000000, 0x00005f, 0x000087, 0x0000af, 0x0000d7, 0x0000ff,
               0x005f00, 0x005f5f, 0x005f87, 0x005faf, 0x005fd7, 0x005fff,
               0x008700, 0x00875f, 0x008787, 0x0087af, 0x0087d7, 0x0087ff,
               0x00af00, 0x00af5f, 0x00af87, 0x00afaf, 0x00afd7, 0x00afff,
               0x00d700, 0x00d75f, 0x00d787, 0x00d7af, 0x00d7d7, 0x00d7ff,
               0x00ff00, 0x00ff5f, 0x00ff87, 0x00ffaf, 0x00ffd7, 0x00ffff,
               0x5f0000, 0x5f005f, 0x5f0087, 0x5f00af, 0x5f00d7, 0x5f00ff,
               0x5f5f00, 0x5f5f5f, 0x5f5f87, 0x5f5faf, 0x5f5fd7, 0x5f5fff,
               0x5f8700, 0x5f875f, 0x5f8787, 0x5f87af, 0x5f87d7, 0x5f87ff,
               0x5faf00, 0x5faf5f, 0x5faf87, 0x5fafaf, 0x5fafd7, 0x5fafff,
               0x5fd700, 0x5fd75f, 0x5fd787, 0x5fd7af, 0x5fd7d7, 0x5fd7ff,
               0x5fff00, 0x5fff5f, 0x5fff87, 0x5fffaf, 0x5fffd7, 0x5fffff,
               0x870000, 0x87005f, 0x870087, 0x8700af, 0x8700d7, 0x8700ff,
               0x875f00, 0x875f5f, 0x875f87, 0x875faf, 0x875fd7, 0x875fff,
               0x878700, 0x87875f, 0x878787, 0x8787af, 0x8787d7, 0x8787ff,
               0x87af00, 0x87af5f, 0x87af87, 0x87afaf, 0x87afd7, 0x87afff,
               0x87d700, 0x87d75f, 0x87d787, 0x87d7af, 0x87d7d7, 0x87d7ff,
               0x87ff00, 0x87ff5f, 0x87ff87, 0x87ffaf, 0x87ffd7, 0x87ffff,
               0xaf0000, 0xaf005f, 0xaf0087, 0xaf00af, 0xaf00d7, 0xaf00ff,
               0xaf5f00, 0xaf5f5f, 0xaf5f87, 0xaf5faf, 0xaf5fd7, 0xaf5fff,
               0xaf8700, 0xaf875f, 0xaf8787, 0xaf87af, 0xaf87d7, 0xaf87ff,
               0xafaf00, 0xafaf5f, 0xafaf87, 0xafafaf, 0xafafd7, 0xafafff,
               0xafd700, 0xafd75f, 0xafd787, 0xafd7af, 0xafd7d7, 0xafd7ff,
               0xafff00, 0xafff5f, 0xafff87, 0xafffaf, 0xafffd7, 0xafffff,
               0xd70000, 0xd7005f, 0xd70087, 0xd700af, 0xd700d7, 0xd700ff,
               0xd75f00, 0xd75f5f, 0xd75f87, 0xd75faf, 0xd75fd7, 0xd75fff,
               0xd78700, 0xd7875f, 0xd78787, 0xd787af, 0xd787d7, 0xd787ff,
               0xd7af00, 0xd7af5f, 0xd7af87, 0xd7afaf, 0xd7afd7, 0xd7afff,
               0xd7d700, 0xd7d75f, 0xd7d787, 0xd7d7af, 0xd7d7d7, 0xd7d7ff,
               0xd7ff00, 0xd7ff5f, 0xd7ff87, 0xd7ffaf, 0xd7ffd7, 0xd7ffff,
               0xff0000, 0xff005f, 0xff0087, 0xff00af, 0xff00d7, 0xff00ff,
               0xff5f00, 0xff5f5f, 0xff5f87, 0xff5faf, 0xff5fd7, 0xff5fff,
               0xff8700, 0xff875f, 0xff8787, 0xff87af, 0xff87d7, 0xff87ff,
               0xffaf00, 0xffaf5f, 0xffaf87, 0xffafaf, 0xffafd7, 0xffafff,
               0xffd700, 0xffd75f, 0xffd787, 0xffd7af, 0xffd7d7, 0xffd7ff,
               0xffff00, 0xffff5f, 0xffff87, 0xffffaf, 0xffffd7, 0xffffff,
               0x080808, 0x121212, 0x1c1c1c, 0x262626, 0x303030, 0x3a3a3a,
               0x444444, 0x4e4e4e, 0x585858, 0x606060, 0x666666, 0x767676,
               0x808080, 0x8a8a8a, 0x949494, 0x9e9e9e, 0xa8a8a8, 0xb2b2b2,
               0xbcbcbc, 0xc6c6c6, 0xd0d0d0, 0xdadada, 0xe4e4e4, 0xeeeeee,
            };

            uint32_t color = colormap[tracker_info[j].color];

            drawRect(
                pimg->data, 
                
                tracker_info[j].bbox.l,
                tracker_info[j].bbox.t,
                tracker_info[j].bbox.r,
                tracker_info[j].bbox.b,

                color
            );

            drawRect(
                pimg->data, 
                
                tracker_info[j].bbox.l + 1,
                tracker_info[j].bbox.t + 1,
                tracker_info[j].bbox.r - 1,
                tracker_info[j].bbox.b - 1,

                color
            );

            drawRect(
                pimg->data, 
                
                tracker_info[j].bbox.l + 2,
                tracker_info[j].bbox.t + 2,
                tracker_info[j].bbox.r - 2,
                tracker_info[j].bbox.b - 2,

                color
            );
        }

        /* free bbox */
        bbox_free(pdetected);

        /* send to tracking buffer */
        sem_wait(&mtx_imgs_trk);
        imgs_trkd[trk_wptr] = pimg;
        trk_wptr = (trk_wptr + 1) & 63;
        sem_post(&mtx_imgs_trk);

        sem_post(&sem_imgs_trk);
    }

	return (0);
}

void * pthread_mtcnn_show(void *)
{
    while (!bQuit)
    {
        /* waiting for a processed image */
        sem_wait(&sem_imgs_trk);
        if (bQuit){break;};

        sem_wait(&mtx_imgs_trk);
        Mat * pimg = imgs_trkd[trk_rptr];
        trk_rptr = (trk_rptr + 1) & 63;
        sem_post(&mtx_imgs_trk);

		// imwrite("capture.png", (*pimg));

        imshow("img", (*pimg));
        waitKey(1);
        pimg->release();

        sem_wait(&mtx_imgs_spc);
        imgs_free[spc_wptr] = pimg;
        spc_wptr = (spc_wptr + 1) & 63;
        sem_post(&mtx_imgs_spc);

        sem_post(&sem_imgs_spc);
    }

	return (0);
}

static pthread_t pid_mtcnn_run, 
                 pid_mtcnn_cap, 
                 pid_mtcnn_trk,
                 pid_mtcnn_show;

int pthread_mtcnn_init()
{
    sem_init(&sem_imgs_spc, 0,  MAX_GPU_BATCH);
    sem_init(&sem_imgs_cap, 0,  0);
    sem_init(&sem_imgs_prc, 0,  0);
    sem_init(&sem_imgs_trk, 0,  0);

    sem_init(&mtx_imgs_spc, 0,  1);
    sem_init(&mtx_imgs_cap, 0,  1);
    sem_init(&mtx_imgs_prc, 0,  1);
    sem_init(&mtx_imgs_trk, 0,  1);

    sem_init(&mtx_bbox_res, 0,  1);

    // for (int i = 0; i < MTCNN_THREADS; i++)
    {
        pthread_create(&pid_mtcnn_run,  NULL, pthread_mtcnn_run,  NULL);
    }
    {
        pthread_create(&pid_mtcnn_show, NULL, pthread_mtcnn_show, NULL);
        pthread_create(&pid_mtcnn_cap,  NULL, pthread_mtcnn_capr, NULL);
        pthread_create(&pid_mtcnn_trk,  NULL, pthread_mtcnn_trkn, NULL);
    }

    return (0);
}

int main(int argc, char ** argv)
{
    int dev = 0;
    if (argc >= 2)
    {
        sscanf(argv[1], "%d", &dev);
    }

    capture.open(dev);
    if (!capture.isOpened())
    {
        printf("cap dev not open\n"); 
        return (-1);
    }

    capture.set(CV_CAP_PROP_FRAME_WIDTH , MTCNN_IMGW);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, MTCNN_IMGH);

    const char * graph = "models/bottle_yolo_2000.pb";
    const char * input_layer = "input_1";
    const char * outputs[3] = { "conv2d_y1/BiasAdd:0", "conv2d_y2/BiasAdd:0", "conv2d_y3/BiasAdd:0" };

	if (argc >= 3)
	{
		graph = (const char *)(argv[2]);
	}
    tensorStartup(graph, input_layer, outputs);

    pthread_mtcnn_init();

    getchar();

    capture.release();

    bQuit = 1;
    
    sem_post(&sem_imgs_cap);
    sem_post(&sem_imgs_spc);
    sem_post(&sem_imgs_prc);

    tensorCleanup();

    return 0;
}
