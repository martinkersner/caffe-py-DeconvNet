#ifndef CAFFE_CONV_LAYER_HPP_
#define CAFFE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit ConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Convolution"; }
  // Martin Kersner, 2016/01/04
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    //this->compute_output_shape();
    int kernel_h_ = this->kernel_shape_.cpu_data()[0];
    int kernel_w_ = kernel_h_;
    int stride_h_ = this->stride_.cpu_data()[0];
    int stride_w_ = stride_h_;
    int pad_w_ = this->pad_.cpu_data()[0];
    int pad_h_ = pad_w_;

    std::cout << "KERNEL_W: " << kernel_h_ << std::endl << std::flush; // Martin Kersner, 2016/01/05
    std::cout << "STRIDE_W: " << stride_h_ << std::endl << std::flush; // Martin Kersner, 2016/01/05
    std::cout << "PAD_W: " << pad_h_ << std::endl << std::flush; // Martin Kersner, 2016/01/05

    //ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    //std::cout << "K2: " << conv_param.kernel_h() << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << "S2: " << conv_param.stride_h() << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << "P2: " << conv_param.pad_h() << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << std::endl << std::flush;

    //std::cout << "K3: " << conv_param.kernel_size_size() << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << "S3: " << conv_param.stride_size() << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << "P3: " << conv_param.pad_size() << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << std::endl << std::flush;

    //std::cout << "K4: " << this->kernel_shape_.cpu_data()[0] << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << "S4: " << this->stride_.cpu_data()[0] << std::endl << std::flush; // Martin Kersner, 2016/01/05
    //std::cout << "P4: " << this->pad_.cpu_data()[0] << std::endl << std::flush; // Martin Kersner, 2016/01/05

    return FilterMap<Dtype>(kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_).inv();
    //return FilterMap<Dtype>(this->kernel_h_, this->kernel_w_, this->stride_h_,
    //    this->stride_w_, this->pad_h_, this->pad_w_).inv();
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
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
