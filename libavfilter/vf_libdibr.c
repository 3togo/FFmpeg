/*
 * Copyright (c) 2010 Stefano Sabatini
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * libdibr wrapper functions
 */

#include "config.h"
#if HAVE_OPENCV2_CORE_CORE_C_H
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#else
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#endif
#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/file.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
void Image_Minus(IplImage *X, IplImage *Y, IplImage *X_Y);
void Image_Cut(IplImage *X, IplImage *Y1, IplImage *Y2);
void Image_Cut2(IplImage *X, IplImage *Y, IplImage *Y2);
static void fill_iplimage_from_frame(IplImage *img, const AVFrame *frame, enum AVPixelFormat pixfmt)
{
    IplImage *tmpimg;
    int depth, channels_nb;

    if      (pixfmt == AV_PIX_FMT_GRAY8) { depth = IPL_DEPTH_8U;  channels_nb = 1; }
    else if (pixfmt == AV_PIX_FMT_BGRA)  { depth = IPL_DEPTH_8U;  channels_nb = 4; }
    else if (pixfmt == AV_PIX_FMT_BGR24) { depth = IPL_DEPTH_8U;  channels_nb = 3; }
    else return;

    tmpimg = cvCreateImageHeader((CvSize){frame->width, frame->height}, depth, channels_nb);
    *img = *tmpimg;
    img->imageData = img->imageDataOrigin = frame->data[0];
    img->dataOrder = IPL_DATA_ORDER_PIXEL;
    img->origin    = IPL_ORIGIN_TL;
    img->widthStep = frame->linesize[0];
}

static void fill_frame_from_iplimage(AVFrame *frame, const IplImage *img, enum AVPixelFormat pixfmt)
{
    frame->linesize[0] = img->widthStep;
    frame->data[0]     = img->imageData;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_BGR24, AV_PIX_FMT_BGRA, AV_PIX_FMT_GRAY8, AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

typedef struct ODIBRContext {
    const AVClass *class;
    char *name;
    char *params;
    int (*init)(AVFilterContext *ctx, const char *args);
    void (*uninit)(AVFilterContext *ctx);
    void (*end_frame_filter)(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg);
    void *priv;
} ODIBRContext;

typedef struct SmoothContext {
    int type;
    int    param1, param2;
    double param3, param4;
} SmoothContext;

static av_cold int smooth_init(AVFilterContext *ctx, const char *args)
{
    ODIBRContext *s = ctx->priv;
    SmoothContext *smooth = s->priv;
    char type_str[128] = "gaussian";

    smooth->param1 = 3;
    smooth->param2 = 0;
    smooth->param3 = 0.0;
    smooth->param4 = 0.0;

    if (args)
        sscanf(args, "%127[^|]|%d|%d|%lf|%lf", type_str, &smooth->param1, &smooth->param2, &smooth->param3, &smooth->param4);

    if      (!strcmp(type_str, "blur"         )) smooth->type = CV_BLUR;
    else if (!strcmp(type_str, "blur_no_scale")) smooth->type = CV_BLUR_NO_SCALE;
    else if (!strcmp(type_str, "median"       )) smooth->type = CV_MEDIAN;
    else if (!strcmp(type_str, "gaussian"     )) smooth->type = CV_GAUSSIAN;
    else if (!strcmp(type_str, "bilateral"    )) smooth->type = CV_BILATERAL;
    else {
        av_log(ctx, AV_LOG_ERROR, "Smoothing type '%s' unknown.\n", type_str);
        return AVERROR(EINVAL);
    }

    if (smooth->param1 < 0 || !(smooth->param1%2)) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid value '%d' for param1, it has to be a positive odd number\n",
               smooth->param1);
        return AVERROR(EINVAL);
    }
    if ((smooth->type == CV_BLUR || smooth->type == CV_BLUR_NO_SCALE || smooth->type == CV_GAUSSIAN) &&
        (smooth->param2 < 0 || (smooth->param2 && !(smooth->param2%2)))) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid value '%d' for param2, it has to be zero or a positive odd number\n",
               smooth->param2);
        return AVERROR(EINVAL);
    }

    av_log(ctx, AV_LOG_VERBOSE, "type:%s param1:%d param2:%d param3:%f param4:%f\n",
           type_str, smooth->param1, smooth->param2, smooth->param3, smooth->param4);
    return 0;
}

static void smooth_end_frame_filter(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg)
{
    ODIBRContext *s = ctx->priv;
    SmoothContext *smooth = s->priv;
    cvSmooth(inimg, outimg, smooth->type, smooth->param1, smooth->param2, smooth->param3, smooth->param4);
}

static int read_shape_from_file(int *cols, int *rows, int **values, const char *filename,
                                void *log_ctx)
{
    uint8_t *buf, *p, *pend;
    size_t size;
    int ret, i, j, w;

    if ((ret = av_file_map(filename, &buf, &size, 0, log_ctx)) < 0)
        return ret;

    /* prescan file to get the number of lines and the maximum width */
    w = 0;
    for (i = 0; i < size; i++) {
        if (buf[i] == '\n') {
            if (*rows == INT_MAX) {
                av_log(log_ctx, AV_LOG_ERROR, "Overflow on the number of rows in the file\n");
                ret = AVERROR_INVALIDDATA;
                goto end;
            }
            ++(*rows);
            *cols = FFMAX(*cols, w);
            w = 0;
        } else if (w == INT_MAX) {
            av_log(log_ctx, AV_LOG_ERROR, "Overflow on the number of columns in the file\n");
            return AVERROR_INVALIDDATA;
        }
        w++;
    }
    if (*rows > (SIZE_MAX / sizeof(int) / *cols)) {
        av_log(log_ctx, AV_LOG_ERROR, "File with size %dx%d is too big\n",
               *rows, *cols);
        ret = AVERROR_INVALIDDATA;
        goto end;
    }
    if (!(*values = av_mallocz_array(sizeof(int) * *rows, *cols))) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    /* fill *values */
    p    = buf;
    pend = buf + size-1;
    for (i = 0; i < *rows; i++) {
        for (j = 0;; j++) {
            if (p > pend || *p == '\n') {
                p++;
                break;
            } else
                (*values)[*cols*i + j] = !!av_isgraph(*(p++));
        }
    }

end:
    av_file_unmap(buf, size);

#ifdef DEBUG
    {
        char *line;
        if (!(line = av_malloc(*cols + 1)))
            return AVERROR(ENOMEM);
        for (i = 0; i < *rows; i++) {
            for (j = 0; j < *cols; j++)
                line[j] = (*values)[i * *cols + j] ? '@' : ' ';
            line[j] = 0;
            av_log(log_ctx, AV_LOG_DEBUG, "%3d: %s\n", i, line);
        }
        av_free(line);
    }
#endif

    return 0;
}

static int parse_iplconvkernel(IplConvKernel **kernel, char *buf, void *log_ctx)
{
    char shape_filename[128] = "", shape_str[32] = "rect";
    int cols = 0, rows = 0, anchor_x = 0, anchor_y = 0, shape = CV_SHAPE_RECT;
    int *values = NULL, ret = 0;

    sscanf(buf, "%dx%d+%dx%d/%32[^=]=%127s", &cols, &rows, &anchor_x, &anchor_y, shape_str, shape_filename);

    if      (!strcmp(shape_str, "rect"   )) shape = CV_SHAPE_RECT;
    else if (!strcmp(shape_str, "cross"  )) shape = CV_SHAPE_CROSS;
    else if (!strcmp(shape_str, "ellipse")) shape = CV_SHAPE_ELLIPSE;
    else if (!strcmp(shape_str, "custom" )) {
        shape = CV_SHAPE_CUSTOM;
        if ((ret = read_shape_from_file(&cols, &rows, &values, shape_filename, log_ctx)) < 0)
            return ret;
    } else {
        av_log(log_ctx, AV_LOG_ERROR,
               "Shape unspecified or type '%s' unknown.\n", shape_str);
        ret = AVERROR(EINVAL);
        goto out;
    }

    if (rows <= 0 || cols <= 0) {
        av_log(log_ctx, AV_LOG_ERROR,
               "Invalid non-positive values for shape size %dx%d\n", cols, rows);
        ret = AVERROR(EINVAL);
        goto out;
    }

    if (anchor_x < 0 || anchor_y < 0 || anchor_x >= cols || anchor_y >= rows) {
        av_log(log_ctx, AV_LOG_ERROR,
               "Shape anchor %dx%d is not inside the rectangle with size %dx%d.\n",
               anchor_x, anchor_y, cols, rows);
        ret = AVERROR(EINVAL);
        goto out;
    }

    *kernel = cvCreateStructuringElementEx(cols, rows, anchor_x, anchor_y, shape, values);
    if (!*kernel) {
        ret = AVERROR(ENOMEM);
        goto out;
    }

    av_log(log_ctx, AV_LOG_VERBOSE, "Structuring element: w:%d h:%d x:%d y:%d shape:%s\n",
           rows, cols, anchor_x, anchor_y, shape_str);
out:
    av_freep(&values);
    return ret;
}

typedef struct DilateContext {
    int nb_iterations;
    IplConvKernel *kernel;
} DilateContext;

static av_cold int dilate_init(AVFilterContext *ctx, const char *args)
{
    ODIBRContext *s = ctx->priv;
    DilateContext *dilate = s->priv;
    char default_kernel_str[] = "3x3+0x0/rect";
    char *kernel_str = NULL;
    const char *buf = args;
    int ret;

    if (args) {
        kernel_str = av_get_token(&buf, "|");

        if (!kernel_str)
            return AVERROR(ENOMEM);
    }

    ret = parse_iplconvkernel(&dilate->kernel,
                              (!kernel_str || !*kernel_str) ? default_kernel_str
                                                            : kernel_str,
                              ctx);
    av_free(kernel_str);
    if (ret < 0)
        return ret;

    if (!buf || sscanf(buf, "|%d", &dilate->nb_iterations) != 1)
        dilate->nb_iterations = 1;
    av_log(ctx, AV_LOG_VERBOSE, "iterations_nb:%d\n", dilate->nb_iterations);
    if (dilate->nb_iterations <= 0) {
        av_log(ctx, AV_LOG_ERROR, "Invalid non-positive value '%d' for nb_iterations\n",
               dilate->nb_iterations);
        return AVERROR(EINVAL);
    }
    return 0;
}

static av_cold void dilate_uninit(AVFilterContext *ctx)
{
    ODIBRContext *s = ctx->priv;
    DilateContext *dilate = s->priv;

    cvReleaseStructuringElement(&dilate->kernel);
}

void Image_Minus(IplImage *X, IplImage *Y, IplImage *X_Y)
{
    //图像差分函数，将图像1中像素和图像2中对应像素想减，要求X、Y、X_Y大小相同
    int i,j,width,height,step,chanel;
    unsigned char *dataX, *dataY, *dataX_Y;
    width = X->width;
    height = X->height;
    dataX = (unsigned char *)X->imageData;
    dataY = (unsigned char *)Y->imageData;
    dataX_Y = (unsigned char *)X_Y->imageData;
    step = X->widthStep/sizeof(char);
    chanel = X->nChannels;
    for(i=0; i<height; i++)
        for(j=0; j<width*chanel; j++)
            dataX_Y[i*step+j] = abs( dataX[i*step+j] - dataY[i*step+j]);
}


void Image_Cut(IplImage *X, IplImage *Y1, IplImage *Y2)
{
    int i,j,width,height,step,chanel,j_mid,count=0;
    unsigned char *dataX, *dataY1, *dataY2;
    width = X->width;
    height = X->height;
    dataX = (unsigned char *)X->imageData;
    dataY1 = (unsigned char *)Y1->imageData;
    dataY2 = (unsigned char *)Y2->imageData;
    step = X->widthStep/sizeof(char);
    chanel = X->nChannels;
    for(i=0; i<height; i++)
        for(j=0; j<width*chanel/2; j++) {
            dataY1[i*step/2+j] =dataX[i*step+j];
            dataY2[i*step/2+j] =dataX[i*step+j+width*chanel/2];
        }
}


//void Image_Cut2(IplImage *X, IplImage *Y)
//{
    //int i,j,width,height,channel,count=0;
    //unsigned char *dataX, *dataY;
    //width = X->width;
    //height = X->height;
    //dataX = (unsigned char *)X->imageData;
    //dataY = (unsigned char *)Y->imageData;
    //int step = X->widthStep/sizeof(char);
    //int stepy = Y->widthStep/sizeof(char);
    //channel = X->nChannels;
    //printf("newstep=%d, step=%d, channel=%d, width=%d, height=%d, X->widthStep=%d, stepy=%d------------------------\n",
            //newstep, step, channel, width, height, X->widthStep, stepy);
    //for(i=0; i<height; i++)
        //for(j=0; j<width*channel/2; j++) {
            //dataY[i*stepy+j] = dataX[i*step+j];
            //count++;
        //}
    ////printf("%d %d %d",width, height, count);
//}


void Image_Cut2(IplImage *X, IplImage *Y, IplImage *Y2)
{
    int i,j,width,height,channel,count=0;
    unsigned char *dataX, *dataY, *dataY2;
    width = X->width;
    height = X->height;
    dataX = (unsigned char *)X->imageData;
    dataY = (unsigned char *)Y->imageData;
    dataY2 = (unsigned char *)Y2->imageData;

    int step = X->widthStep/sizeof(char);
    int stepy = Y->widthStep/sizeof(char);
    int stepy2 = Y2->widthStep/sizeof(char);
    channel = X->nChannels;
    printf("step=%d, channel=%d, width=%d, height=%d, X->widthStep=%d, stepy=%d------------------------\n",
            step, channel, width, height, X->widthStep, stepy);
    for(i=0; i<height; i++)
        for(j=0; j<width/2; j++) {
            for (int k=0; k< 3; k++)
                dataY[i*stepy+j*3+k] = dataX[i*step+3*j+k];
            dataY2[i*stepy2+j] = dataX[i*step+3*j+step/2];
            count++;
        }
    //printf("%d %d %d",width, height, count);
}


static void dilate_end_frame_filter(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg)
{
    IplImage *inimg2=0;
    int h, w, w2;
    ODIBRContext *s = ctx->priv;
    DilateContext *dilate = s->priv;
    //cvDilate(inimg, outimg, dilate->kernel, dilate->nb_iterations);
    w = inimg->width;
    h = inimg->height;
    w2 = (int)(w/2);
    IplImage *srcimg = cvCreateImage(cvSize(w/2, h),
                               inimg->depth,
                               inimg->nChannels);
    IplImage *depimg_gray = cvCreateImage(cvSize(w/2, h),
                               inimg->depth,
                               1);
    IplImage *depimg = cvCreateImage(cvSize(w/2, h),
                               inimg->depth,
                               inimg->nChannels);

    //IplImage *srcimg;
    Image_Cut2(inimg, srcimg, depimg_gray);
    //cvCopy(inimg, inimg2, NULL);
    //inimg2 = cvCloneImage(inimg);
    //cvSetImageROI(inimg,cvRect(0,0,w2,h));
    //cvCopy(inimg, srcimg, NULL);
    //cvResetImageROI(inimg);
    //cvSetImageROI(inimg,cvRect(w2,0,w2,h));
    //cvCopy(inimg, depimg, NULL);
        //cvCvtColor(depimg_gray,depimg,CV_GRAY2BGR);
        cvResize(srcimg, outimg, CV_INTER_LINEAR);
    //Image_Minus(inimg, outimg, outimg);
    //CvMat outmat, mat_src, mat_dep;
    //cvGetMat(outimg, &outmat, 0, 0);
    //h = outmat.rows;
    //w = outmat.cols;

    //printf("size=(%d,%d)___________________________\n", h, w);
    //mat_src=outmat(CvRect(0,0,w2.h));
    //h = mat_src.rows;
    //w = mat_src.cols;
    //printf("size=(%d,%d)___________________________==================\n", h, w);
    //cvGetImage(&outmat, outimg);
    //cvSobel(inimg, outimg, 1, 1, 3);
    //cvReleaseImage(inimg2);
    //cvReleaseImage(&srcimg);
    //cvReleaseImage(&depimg);
}

static void erode_end_frame_filter(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg)
{
    ODIBRContext *s = ctx->priv;
    DilateContext *dilate = s->priv;
    cvErode(inimg, outimg, dilate->kernel, dilate->nb_iterations);
}

typedef struct ODIBRFilterEntry {
    const char *name;
    size_t priv_size;
    int  (*init)(AVFilterContext *ctx, const char *args);
    void (*uninit)(AVFilterContext *ctx);
    void (*end_frame_filter)(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg);
} ODIBRFilterEntry;

static const ODIBRFilterEntry odibr_filter_entries[] = {
    { "dilate", sizeof(DilateContext), dilate_init, dilate_uninit, dilate_end_frame_filter },
    { "erode",  sizeof(DilateContext), dilate_init, dilate_uninit, erode_end_frame_filter  },
    { "smooth", sizeof(SmoothContext), smooth_init, NULL, smooth_end_frame_filter },
};

static av_cold int init(AVFilterContext *ctx)
{
    ODIBRContext *s = ctx->priv;
    int i;

    if (!s->name) {
        av_log(ctx, AV_LOG_ERROR, "No libdibr filter name specified\n");
        return AVERROR(EINVAL);
    }
    for (i = 0; i < FF_ARRAY_ELEMS(odibr_filter_entries); i++) {
        const ODIBRFilterEntry *entry = &odibr_filter_entries[i];
        if (!strcmp(s->name, entry->name)) {
            s->init             = entry->init;
            s->uninit           = entry->uninit;
            s->end_frame_filter = entry->end_frame_filter;

            if (!(s->priv = av_mallocz(entry->priv_size)))
                return AVERROR(ENOMEM);
            return s->init(ctx, s->params);
        }
    }

    av_log(ctx, AV_LOG_ERROR, "No libdibr filter named '%s'\n", s->name);
    return AVERROR(EINVAL);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ODIBRContext *s = ctx->priv;

    if (s->uninit)
        s->uninit(ctx);
    av_freep(&s->priv);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    ODIBRContext *s = ctx->priv;
    AVFilterLink *outlink= inlink->dst->outputs[0];
    AVFrame *out;
    IplImage inimg, outimg;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    fill_iplimage_from_frame(&inimg , in , inlink->format);
    fill_iplimage_from_frame(&outimg, out, inlink->format);
    s->end_frame_filter(ctx, &inimg, &outimg);
    fill_frame_from_iplimage(out, &outimg, inlink->format);

    av_frame_free(&in);

    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(ODIBRContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
static const AVOption odibr_options[] = {
    { "filter_name",   NULL, OFFSET(name),   AV_OPT_TYPE_STRING, .flags = FLAGS },
    { "filter_params", NULL, OFFSET(params), AV_OPT_TYPE_STRING, .flags = FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(odibr);

static const AVFilterPad avfilter_vf_odibr_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad avfilter_vf_odibr_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_odibr = {
    .name          = "odibr",
    .description   = NULL_IF_CONFIG_SMALL("Apply transform using libdibr."),
    .priv_size     = sizeof(ODIBRContext),
    .priv_class    = &odibr_class,
    .query_formats = query_formats,
    .init          = init,
    .uninit        = uninit,
    .inputs        = avfilter_vf_odibr_inputs,
    .outputs       = avfilter_vf_odibr_outputs,
};

