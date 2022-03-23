#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolov3(const cv::Mat& bgr, std::vector<std::vector<Object> >& objects, const char* param, const char* bin)
{
    ncnn::Net yolov3;

    yolov3.opt.use_vulkan_compute = true;

    yolov3.load_param(param);
    yolov3.load_model(bin);

    const int target_size = 416;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();

    ex.input("data", in);

    ncnn::Mat out0;
    ncnn::Mat out1;
    ex.extract("yolo0", out0);
    ex.extract("yolo1", out1);

    //ncnnmat_save(out0);
    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    objects.resize(2);
    objects[0].resize(out0.h);
    objects[1].resize(out1.h);

    for (int i = 0; i < out0.h; i++)
    {
        const float* values = out0.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects[0].push_back(object);
    }

    for (int i = 0; i < out1.h; i++)
    {
        const float* values = out1.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects[1].push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<std::vector<Object> >& objects)
{
    static const char* class_names[] = {"background",
                                        "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                                        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                                        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                                        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                                        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
                                        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy bear", "hair drier", "toothbrush"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        for (size_t j = 0; j < objects[i].size(); j++)
        {
            const Object& obj = objects[i][j];

            if (obj.label == 0) continue;

            if (obj.prob <= 0.3) continue;

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath] [param] [bin]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* param = argv[2];
    const char* bin = argv[3];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<std::vector<Object> > objects;
    detect_yolov3(m, objects, param, bin);

    draw_objects(m, objects);

    return 0;
}
