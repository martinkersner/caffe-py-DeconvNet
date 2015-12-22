#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

  virtual void CreatePrefetchThread(); // Martin Kersner, 2015/12/18
  virtual void JoinPrefetchThread(); // Martin Kersner, 2015/12/18

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;
  Blob<Dtype> prefetch_data_; // Martin Kersner, 2015/12/18
  Blob<Dtype> prefetch_label_; // Martin Kersner, 2015/12/18
};

// Martin Kersner, 2015/15/18
/** Jay add
 * @brief prefetching data layer which also prefetches data dimensions
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDimPrefetchingDataLayer : public BasePrefetchingDataLayer<Dtype> {
/*
 notice:
 this code is based on the following implementation.
 https://bitbucket.org/deeplab/deeplab-public/
 */

 public:
  explicit ImageDimPrefetchingDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDimPrefetchingDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_dim_;
  bool output_data_dim_;
};

// Martin Kersner, 2015/12/22
template <typename Dtype>
class WindowClsDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit WindowClsDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowClsDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  //virtual inline LayerParameter_LayerType type() const {}
  virtual inline V1LayerParameter_LayerType V1type() const { // Martin Kersner, 2015/12/16
    //return LayerParameter_LayerType_IMAGE_DATA;
    return V1LayerParameter_LayerType_IMAGE_DATA; // Martin Kersner, 2015/12/16
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) {} // Martin Kersner, 2015/12/22

 protected:
  Blob<Dtype> seg_label_buffer_;
  Blob<Dtype> transformed_label_;
  Blob<Dtype> computed_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  typedef struct SegItems {
    std::string imgfn;
    std::string segfn;
    int x1, y1, x2, y2;
  } SEGITEMS;

  vector<SEGITEMS> lines_;
  int lines_id_;
  int label_dim_;
};

template <typename Dtype>
class WindowInstSegDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit WindowInstSegDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowInstSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  //virtual inline LayerParameter_LayerType type() const {}
  virtual inline V1LayerParameter_LayerType V1type() const { // Martin Kersner, 2015/12/16
    //return LayerParameter_LayerType_IMAGE_DATA;
    return V1LayerParameter_LayerType_IMAGE_DATA; // Martin Kersner, 2015/12/16
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) {} // Martin Kersner, 2015/12/22

 protected:
  Blob<Dtype> transformed_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  typedef struct InstItems {
    std::string imgfn;
    std::string segfn;
    std::string instfn;
    int x1, y1, x2, y2, inst_label;
  } INSTITEMS;

  vector<INSTITEMS> lines_;
  int lines_id_;
};
// Martin Kersner, 2015/12/22

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
