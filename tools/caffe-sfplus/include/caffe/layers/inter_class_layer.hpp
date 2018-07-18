#ifndef CAFFE_INTER_CLASS_LAYER_HPP_
#define CAFFE_INTER_CLASS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Also known as a "marginal fully-connected" layer, computes an marginal inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InterClassLayer : public LossLayer<Dtype> {
 public:
  explicit InterClassLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InterClass"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
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

  int M_;
  int K_;
  int N_;
  
  InterClassParameter_InterClassType type_;

  // for between-class scatter(bcs) 
  Blob<Dtype> weight_mean_;
  Blob<Dtype> temp_mean_norm_gpu;
  Blob<int> alpha_stepvalues;
  Blob<Dtype> weight_wise_diff_;
  Blob<Dtype> weight_wise_dist_sq_;

  int iter_;
  int alpha_index_;
  Dtype alpha_; // for between-class scatter(bcs) 

};

}  // namespace caffe

#endif  // CAFFE_MAEGIN_INNER_PRODUCT_LAYER_HPP_
