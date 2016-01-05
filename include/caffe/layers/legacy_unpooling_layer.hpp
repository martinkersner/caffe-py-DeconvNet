#ifndef CAFFE_LEGACY_UNPOOLING_LAYER_HPP_
#define CAFFE_LEGACY_UNPOOLING_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Unools the input image .
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class UnpoolingLayer : public Layer<Dtype> {
 public:
  explicit UnpoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  //virtual inline LayerParameter_LayerType type() const {
  //virtual inline V1LayerParameter_LayerType V1type() const { // Martin Kersner, 2015/12/16
  //  //return LayerParameter_LayerType_UNPOOLING;
  //  return V1LayerParameter_LayerType_UNPOOLING; // Martin Kersner, 2015/12/16
  //}
  virtual inline const char* type() const { return "Unpooling"; } // Martin Kersner, 2015/12/29
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxBottomBlobs() const {
    return (this->layer_param_.unpooling_param().unpool() ==
            UnpoolingParameter_UnpoolMethod_MAX) ? 2 : 1;
  }
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    std::cout << "LEGACY UNPOOLING coord_map" << std::endl << std::flush; // Martin Kernser, 2015/12/31
    return FilterMap<Dtype>(kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_);
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int unpooled_height_, unpooled_width_;
};

}

#endif  // CAFFE_LEGACY_UNPOOLING_LAYER_HPP__
