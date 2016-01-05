#ifndef CAFFE_DECONV_LAYER_HPP_
#define CAFFE_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ConvolutionLayer.
 *
 *   ConvolutionLayer computes each output value by dotting an input window with
 *   a filter; DeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
 *   reversed. DeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Dtype>
class DeconvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit DeconvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Deconvolution"; }
  // Martin Kersner, 2016/01/04
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    int kernel_h_ = this->kernel_shape_.cpu_data()[0];
    int kernel_w_ = kernel_h_;
    int stride_h_ = this->stride_.cpu_data()[0];
    int stride_w_ = stride_h_;
    int pad_w_ = this->pad_.cpu_data()[0];
    int pad_h_ = pad_w_;

    std::cout << "KERNEL_W: " << kernel_h_ << std::endl << std::flush; // Martin Kersner, 2016/01/05
    std::cout << "STRIDE_W: " << stride_h_ << std::endl << std::flush; // Martin Kersner, 2016/01/05
    std::cout << "PAD_W: " << pad_h_ << std::endl << std::flush; // Martin Kersner, 2016/01/05

    return FilterMap<Dtype>(kernel_h_, kernel_w_, stride_h_,
        stride_w_, pad_h_, pad_w_);
    //return FilterMap<Dtype>(this->kernel_h_, this->kernel_w_, this->stride_h_,
    //    this->stride_w_, this->pad_h_, this->pad_w_);
  }
  // Martin Kersner, 2016/01/04

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_DECONV_LAYER_HPP_
