#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

void ncnnmat_save(ncnn::Mat out)
{
    FILE* fp = fopen("featuremap.txt", "w");
    for (int c = 0; c < out.c; c++)
    {
        const float* c_ptr = out.channel(c);
        for (int i = 0; i < out.h; i++)
        {
            for (int j = 0; j < out.w; j++)
            {
                fprintf(fp, "%f\n", c_ptr[j]);
            }
            c_ptr += out.w;
        }
    }
    fclose(fp);
}

static int detect_mobilenet(const cv::Mat& bgr, std::vector<float>& cls_scores, const char* parampath, const char* binpath)
{
    ncnn::Net mobilenet;

    mobilenet.opt.use_vulkan_compute = true;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    mobilenet.load_param(parampath);
    mobilenet.load_model(binpath);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f, 0.017f, 0.017f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc7", out);
    printf("C x H x W : [%d, %d, %d]\n", out.c, out.h, out.w);
    //ncnnmat_save(out);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath] [param] [bin]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* parampath = argv[2];
    const char* binpath = argv[3];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_mobilenet(m, cls_scores, parampath, binpath);

    print_topk(cls_scores, 3);

    return 0;
}