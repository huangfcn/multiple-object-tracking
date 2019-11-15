/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <Windows.h>

// #include <fstream>
// #include <utility>
#include <vector>

#include "opencv2/opencv.hpp"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using namespace cv;
using namespace std;

typedef struct
{
	char input_layer[128];
	char output_layer[3][128];
} yolo3d_task_param_t;

yolo3d_task_param_t yolo3d_task_params;

typedef struct _bbox_pos_s
{
	int l, t, b, r; int type; float score;
} bbox_t;

typedef struct _bbox_chain_s
{
	int nbox; bbox_t bbox[128];
} bbox_chain_t;

typedef struct
{
	float obj_thresh;
	float nms_thresh;
	int   anchors[18];
} yolo3_options_t;

typedef struct {
	float x, y, u, w;
	int   c;
	float s;
} predecode_t;

typedef struct _detection_t {
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	int classes;
	float objectness;
} detection_t;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
static Status LoadGraph(
	const char * graph_file_name,
	std::unique_ptr<tensorflow::Session>* session
)
{
	bool allow_growth = false;

	tensorflow::SessionOptions session_options;

	tensorflow::GraphDef graph_def;
	Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok())
	{
		return tensorflow::errors::NotFound(
			"Failed to load compute graph at '",
			graph_file_name,
			"'"
		);
	}
	session->reset(tensorflow::NewSession(session_options));
	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok())
	{
		return session_create_status;
	}
	return Status::OK();
}

static void decode_netout(
	vector<predecode_t> & boxes,
	const float * out_vector,
	const int   * yolo_anchors,
	float obj_thresh,
	int tensor_height,
	int tensor_width,
	int grid_h,
	int grid_w,
	int num_classes
)
{
	int nb_box = 3 * (5 + num_classes);
	// float net[MAX_GRID_H][MAX_GRID_W][MAX_NBOXES];

	vector<vector<vector<float>>> net(grid_h, vector<vector<float>>(grid_w, vector<float>(nb_box, 0)));
	for (int i = 0; i < grid_h; i++)
	{
		for (int j = 0; j < grid_w; j++)
		{
			for (int k = 0; k < nb_box; k++)
			{
				switch (k % (5 + num_classes))
				{
				case 2:
				case 3:
					net[i][j][k] = exp(out_vector[(i * grid_w * nb_box + j * nb_box + k)]);
					break;
				default:
					net[i][j][k] = (1 / (1 + exp(-(out_vector[(i * grid_w * nb_box + j * nb_box + k)]))));
					break;
				}
			}
		}
	}

	float objectness;
	float scores;

	for (int i = 0; i < (grid_h * grid_w); i++) {
		int row = i / grid_w;
		int col = i % grid_w;
		for (int b = 0; b < nb_box / (5 + num_classes); b++) {
			objectness = net[row][col][b * (5 + num_classes) + 4];
			for (int j = 0; j < num_classes; j++) {
				scores = net[row][col][b * (5 + num_classes) + 5 + j] * objectness;
				if (scores >= obj_thresh) {
					predecode_t box;

					box.x = (col + net[row][col][b * (5 + num_classes) + 0]) / grid_w;
					box.y = (row + net[row][col][b * (5 + num_classes) + 1]) / grid_h;
					box.u = yolo_anchors[2 * b + 0] * net[row][col][b * (5 + num_classes) + 2] / tensor_width;
					box.w = yolo_anchors[2 * b + 1] * net[row][col][b * (5 + num_classes) + 3] / tensor_height;
					box.s = scores;
					box.c = j;
					boxes.push_back(box);
				}
			}
		}
	}
}

static void correct_yolo_boxes(
	vector<detection_t> & cboxes,
	vector<predecode_t> & boxes,
	int tensor_height,
	int tensor_width,
	int image_height,
	int image_width,
	int num_classes
)
{
	detection_t cbox;
	float x, y, w, h;
	if (boxes.size() == 0) {
		cbox.objectness = {};
		cboxes.push_back(cbox);
	}
	else
	{
		float new_w, new_h;
		if ((float(tensor_width) / float(image_width)) <
			(float(tensor_height) / float(image_height)))
		{
			new_w = float(tensor_width);
			new_h = round(float(image_height) * tensor_width / float(image_width));
		}
		else {
			new_h = float(tensor_height);
			new_w = round(float(image_width) * tensor_height / float(image_height));
		}
		float x_offset, x_scale, y_offset, y_scale;
		for (int i = 0; i < boxes.size(); i++)
		{
			x_offset = (tensor_width - new_w) / 2.0 / tensor_width;
			x_scale = float(new_w) / tensor_width;
			y_offset = (tensor_height - new_h) / 2.0 / tensor_height;
			y_scale = float(new_h) / tensor_height;
			x = (boxes[i].x - x_offset) / x_scale * float(image_width);
			y = (boxes[i].y - y_offset) / y_scale * float(image_height);
			w = (boxes[i].u) / x_scale * float(image_width);
			h = (boxes[i].w) / y_scale * float(image_height);
			cbox.xmin = x - w / 2;
			cbox.xmax = x + w / 2;
			cbox.ymin = y - h / 2;
			cbox.ymax = y + h / 2;
			cbox.objectness = (boxes[i].s);
			cbox.classes = boxes[i].c;
			cboxes.push_back(cbox);
		}
	}
}

static void sort(vector<detection_t> & cboxes, vector<int> & indices)
{
	for (int i = 0; i < cboxes.size(); i++)
	{
		indices[i] = i;
	}

	for (int i = 0; i < cboxes.size(); i++)
	{
		for (int j = i + 1; j < cboxes.size(); j++)
		{
			if (cboxes[indices[j]].objectness > cboxes[indices[i]].objectness)
			{
				// float x_tmp = x[i];
				int index_tmp = indices[i];
				// x[i] = x[j];
				indices[i] = indices[j];
				// x[j] = x_tmp;
				indices[j] = index_tmp;
			}
		}
	}
	// return indices;
}

static void do_nms(
	vector<detection_t> & nboxes,
	vector<detection_t> & boxes,
	float nms_thresh,
	int num_classes
)
{
	vector<int> is_suppressed;
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	float overWidth, overHeight;
	float box_area1, box_area2;

	for (int c = 0; c < num_classes; c++)
	{
		vector<detection_t> cboxes;
		for (int j = 0; j < boxes.size(); j++)
		{
			if (boxes[j].classes == c)
			{
				cboxes.push_back(boxes[j]);
			}
		}
		for (int i = 0; i < cboxes.size(); i++)
		{
			is_suppressed.push_back(0);
		}

		vector<int> indices(cboxes.size()); sort(cboxes, indices);
		for (int i = 0; i < cboxes.size(); i++)
		{
			if (!is_suppressed[indices[i]])
			{
				for (int j = i + 1; j < cboxes.size(); j++)
				{
					maxX = min(cboxes[indices[j]].xmax, cboxes[indices[i]].xmax);
					maxY = min(cboxes[indices[j]].ymax, cboxes[indices[i]].ymax);
					minX = max(cboxes[indices[j]].xmin, cboxes[indices[i]].xmin);
					minY = max(cboxes[indices[j]].ymin, cboxes[indices[i]].ymin);
					overWidth = maxX - minX + 1;
					overHeight = maxY - minY + 1;
					if (overWidth > 0 & overHeight > 0)
					{
						box_area1 = (cboxes[indices[j]].xmax - cboxes[indices[j]].xmin + 1) * (cboxes[indices[j]].ymax - cboxes[indices[j]].ymin + 1);
						box_area2 = (cboxes[indices[i]].xmax - cboxes[indices[i]].xmin + 1) * (cboxes[indices[i]].ymax - cboxes[indices[i]].ymin + 1);

						//求交并比IOU
						IOU = (overWidth * overHeight) / (box_area1 + box_area2 - overWidth * overHeight);
						if (IOU > nms_thresh)
						{
							is_suppressed[indices[j]] = 1;
						}
					}
				}
			}
		}

		for (int i = 0; i < cboxes.size(); i++)  // 遍历所有输入窗口
		{
			if (!is_suppressed[indices[i]])  // 将未发生抑制的窗口信息保存到输出信息中
			{
				detection_t nbox;
				nbox.xmax = cboxes[indices[i]].xmax;
				nbox.xmin = cboxes[indices[i]].xmin;
				nbox.ymax = cboxes[indices[i]].ymax;
				nbox.ymin = cboxes[indices[i]].ymin;
				nbox.objectness = cboxes[indices[i]].objectness;
				nbox.classes = cboxes[indices[i]].classes;
				nboxes.push_back(nbox);
			}
		}
		cboxes.clear();
	}
	// return nboxes;
}

std::unique_ptr<tensorflow::Session> session;

extern "C" __declspec(dllexport) int __cdecl tensorStartup(
	const char * graph_path,
	const char * input_layer,
	const char * output_layer[3]
)
{
	int argc = 1;
	const char* argv[] = { "tensorinfer" };

	tensorflow::port::InitMain("tensorinfer", &argc, ((char***)(&argv)));

	Status load_graph_status = LoadGraph(graph_path, &session);
	if (!load_graph_status.ok()) 
	{
		LOG(ERROR) << load_graph_status;
		return (-1);
	}
	
	/* input/output layer */
	strcpy(yolo3d_task_params.input_layer, input_layer);
	strcpy(yolo3d_task_params.output_layer[0], output_layer[0]);
	strcpy(yolo3d_task_params.output_layer[1], output_layer[1]);
	strcpy(yolo3d_task_params.output_layer[2], output_layer[2]);

	return (0);
}

extern "C" __declspec(dllexport) int __cdecl tensorCleanup()
{
	session->Close();
	return (0);
}

extern "C" __declspec(dllexport) int __cdecl tensorRunB(
	int batchSize,
	int tensor_height,
	int tensor_width,
	int * image_height,
	int * image_width,
	unsigned char **  pimgbuf,
	bbox_chain_t  **  pbbox,
	yolo3_options_t * popt
)
{
	int * anchors = popt->anchors;
	int   grid_h  = tensor_height / 32;
	int   grid_w  = tensor_width  / 32;
	int   tensor_depth = 3;

	float obj_thresh = popt->obj_thresh;
	float nms_thresh = popt->nms_thresh;

	/* image resizing */
	std::vector<cv::Mat> imgArrayB;

	// creating a Tensor for storing the data
	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ batchSize, tensor_height, tensor_width, tensor_depth }));
	auto input_tensor_mapped = input_tensor.tensor<float, 4>();

	for (int bz = 0; bz < batchSize; ++bz)
	{
		int new_h = image_height[bz]; // imgArray.at(bz).rows;
		int new_w = image_width[bz];  // imgArray.at(bz).cols;

		if ((float(tensor_width) / float(new_w)) < (float(tensor_height) / float(new_h)))
		{
			new_h = int(round(float(new_h) * tensor_width / float(new_w)));
			new_w = int(tensor_width);

		}
		else
		{
			new_w = int(round(float(new_w) * tensor_height / float(new_h)));
			new_h = int(tensor_height);
		}

		cv::Size ssize = cv::Size(image_width[bz], image_height[bz]);
		cv::Size dsize = cv::Size(new_w, new_h);

		cv::Mat resized; 
		const cv::Mat orig(ssize, CV_8UC3, (unsigned char *)pimgbuf[bz]);
		cv::resize(orig, resized, dsize, 0, 0, cv::INTER_LINEAR);
		
		uchar * source_data; source_data = resized.data;
		for (int y = 0; y < tensor_height; ++y)
		{
			for (int x = 0; x < tensor_width; ++x)
			{
				input_tensor_mapped(bz, y, x, 0) = 0.5;
				input_tensor_mapped(bz, y, x, 1) = 0.5;
				input_tensor_mapped(bz, y, x, 2) = 0.5;
			}
		}
		for (int y = (tensor_height - new_h) / 2; y < (tensor_height + new_h) / 2; ++y)
		{
			uchar * source_row = source_data + ((y - (tensor_height - new_h) / 2) * new_w * tensor_depth);
			for (int x = (tensor_width - new_w) / 2; x < (tensor_width + new_w) / 2; ++x)
			{
				uchar * source_pixel = source_row + ((x - (tensor_width - new_w) / 2) * tensor_depth);
				/* preprocessing */
				float B = source_pixel[0] / 255.0;
				float G = source_pixel[1] / 255.0;
				float R = source_pixel[2] / 255.0;

				input_tensor_mapped(bz, y, x, 0) = R;
				input_tensor_mapped(bz, y, x, 1) = G;
				input_tensor_mapped(bz, y, x, 2) = B;
			}
		}
	}

	// Actually run the image through the model.
	std::vector<Tensor> outputs; Status run_status = session->Run(
		{ { yolo3d_task_params.input_layer, input_tensor } },
		{ yolo3d_task_params.output_layer[0], yolo3d_task_params.output_layer[1], yolo3d_task_params.output_layer[2] },
		{}, &outputs
	);

	auto scores_flat_0 = outputs[0].flat<float>();
	auto scores_flat_1 = outputs[1].flat<float>();
	auto scores_flat_2 = outputs[2].flat<float>();

	float * output_data_0 = scores_flat_0.data();
	float * output_data_1 = scores_flat_1.data();
	float * output_data_2 = scores_flat_2.data();

	for (int b = 0; b < batchSize; b++)
	{
		int eachBatchSize = scores_flat_0.size() / batchSize;

		std::vector<predecode_t> boxes; 
		std::vector<detection_t> nboxes; 
		std::vector<detection_t> cboxes;

		int num_classes = eachBatchSize / grid_h / grid_w / 3 - 5;

		decode_netout(boxes, output_data_0 + ((b * eachBatchSize) << 0), anchors + 12, obj_thresh, tensor_height, tensor_width, grid_h << 0, grid_w << 0, num_classes);
		decode_netout(boxes, output_data_1 + ((b * eachBatchSize) << 2), anchors +  6, obj_thresh, tensor_height, tensor_width, grid_h << 1, grid_w << 1, num_classes);
		decode_netout(boxes, output_data_2 + ((b * eachBatchSize) << 4), anchors +  0, obj_thresh, tensor_height, tensor_width, grid_h << 2, grid_w << 2, num_classes);

		correct_yolo_boxes(cboxes, boxes, tensor_height, tensor_width, image_height[b], image_width[b], num_classes);
		do_nms(nboxes, cboxes, nms_thresh, num_classes);

		/* save into desired structure */
		int nbox = 0;
		for (int i = 0; i < nboxes.size(); i++)
		{
			nboxes[i].ymin = max(nboxes[i].ymin, 0);
			nboxes[i].xmin = max(nboxes[i].xmin, 0);
			nboxes[i].ymax = min(nboxes[i].ymax, (image_height[b]-1));
			nboxes[i].xmax = min(nboxes[i].xmax, (image_width[b]-1));

			if ((nboxes[i].ymin > nboxes[i].ymax) ||
				(nboxes[i].xmin > nboxes[i].xmax) ||
				(nboxes[i].ymin < 0)              ||
				(nboxes[i].xmin < 0)              ||
				(nboxes[i].xmax >= image_width[b])||
				(nboxes[i].ymax >= image_height[b]))
			{  
				continue;
			}

			pbbox[b]->bbox[nbox].t = nboxes[i].ymin;
			pbbox[b]->bbox[nbox].l = nboxes[i].xmin;
			pbbox[b]->bbox[nbox].b = nboxes[i].ymax;
			pbbox[b]->bbox[nbox].r = nboxes[i].xmax;
			pbbox[b]->bbox[nbox].type = nboxes[i].classes;
			pbbox[b]->bbox[nbox].score = nboxes[i].objectness;
			++nbox;
		}
		pbbox[b]->nbox = nbox;
	}

	return (0);
}

extern "C" __declspec(dllexport) int __cdecl tensorRunS(
	int tensor_height, int tensor_width,
	int image_height,  int image_width,
	unsigned char * pimgbuf,
	bbox_chain_t * pbbox,
	yolo3_options_t * popt
)
{
	return tensorRunB(1, tensor_height, tensor_width, &image_height, &image_width, &pimgbuf, &pbbox, popt);
}

int tensorRun(
	bbox_chain_t * pbbox,
	std::string & image_path,
	int tensor_height, int tensor_width,
	float obj_thresh, float nms_thresh
)
{
	yolo3_options_t opts = {
		obj_thresh,
		nms_thresh,
		{55, 69, 75, 234, 133, 240, 136, 129, 142, 363, 203, 290, 228, 184, 285, 359, 341, 260}
	};

	cv::Mat img = cv::imread(image_path);
	
	tensorRunS(tensor_height, tensor_width, img.rows, img.cols, img.data, pbbox, &opts);

	/* save into desired structure */
	for (int i = 0; i < pbbox->nbox; i++)
	{
		float x, y, w, h;
		x = pbbox->bbox[i].l;
		y = pbbox->bbox[i].t;
		w = pbbox->bbox[i].r - pbbox->bbox[i].l;
		h = pbbox->bbox[i].b - pbbox->bbox[i].t;

		Rect rect = Rect(x, y, w, h);
		rectangle(img, rect, Scalar(0, 0, 255));
	}

	cv::imshow("test", img);
	cv::waitKey();

	return (0);
}

int main(int argc, char* argv[]) 
{
	// These are the command-line flags the program can understand.
	// They define where the graph and input data is located, and what kind of
	// input the model expects. If you train your own model, or use something
	// other than inception_v3, then you'll need to update these.
	std::string image = "b000020.jpg";
	const char * graph = "models/bottle_yolo_2000.pb";

	int tensor_width = 480;
	int tensor_height = 480;

	float obj_thresh = 0.5;
	float nms_thresh = 0.45;

	string input_layer = "input_1";
	string output_layer = "output";

	const char * outputs[3] = { "output_0", "output_1", "output_2" };
	if (tensorStartup(graph, "input_1", outputs) < 0) {
		return (-1);
	}

	// Get the image from disk as a float array of numbers, resized and normalized
	// to the specifications the main graph expects.
	std::vector<string> image_paths;

	image_paths.push_back(image);
	// image_paths.push_back(image);
	// image_paths.push_back(image);

	bbox_chain_t bbox;
	if (tensorRun(&bbox, image, tensor_height, tensor_width, obj_thresh, nms_thresh) < 0) {
		return (-1);
	}

	// printf("\nscore = [%9.4f, %9.4f]\n", scores[0], scores[1]);
	return 0;
}
