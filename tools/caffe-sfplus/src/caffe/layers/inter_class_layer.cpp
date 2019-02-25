#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/inter_class_layer.hpp"

namespace caffe {

template <typename Dtype>
void InterClassLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Number of labels must match number of output; "
      << "DO NOT support multi-label this version."
      << "e.g., if prediction shape is (M X N), "
      << "label count (number of labels) must be M, "
      << "with integer values in {0, 1, ..., N-1}.";
  type_ = this->layer_param_.inter_class_param().type();
  iter_ = this->layer_param_.inter_class_param().iteration();
  alpha_ = (Dtype)0.;

  const int num_output = this->layer_param_.inter_class_param().num_output();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inter_class_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inter_class_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InterClassLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  LossLayer<Dtype>::Reshape(bottom, top);
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inter_class_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  

  // if needed, reshape top[1] to output lambda & alpha
  if (top.size() == 2) {
    //output #0 is lambda,output #1 is alpha 
    vector<int> alpha_shape(1, 1); 
    top[1]->Reshape(alpha_shape);
  }
  
  // alpha_stepvalue
  const InterClassParameter& param = this->layer_param_.inter_class_param();
  Dtype alpha_start_iter_ = param.alpha_start_iter();
  CHECK_GE(alpha_start_iter_, (Dtype)0) 
       << "alpha_start_iter_ should be great or equal to zero";
  Dtype alpha_stepvalue_size = param.alpha_stepvalue_size();
  if (alpha_stepvalue_size != 0){
    vector<int> alpha_stepvalues_shape(1, alpha_stepvalue_size);
    alpha_stepvalues.Reshape(alpha_stepvalues_shape);
    int* mutable_alpha_stepvalue_data = alpha_stepvalues.mutable_cpu_data();
    for (int i = 0; i < alpha_stepvalue_size; ++i) {
      int tmp_alpha_stepvalue = 0;
      CHECK_GT(param.alpha_stepvalue(i),alpha_start_iter_) 
          << "alpha_stepvalue should be great or equal to alpha_start_iter_";
      if (param.alpha_stepvalue_size() == 1) {
        tmp_alpha_stepvalue = param.alpha_stepvalue(0);
      } else if (param.alpha_stepvalue_size() > 1) {
        if(i>0){
          CHECK_GT(param.alpha_stepvalue(i),param.alpha_stepvalue(i-1))
            << "alpha_stepvalue should ascend";
        }
        tmp_alpha_stepvalue = param.alpha_stepvalue(i);
      }
      mutable_alpha_stepvalue_data[i] = tmp_alpha_stepvalue;
    }
  }


  // optional temp variables
  switch (type_) {
    case InterClassParameter_InterClassType_MEAN:{
      vector<int> weight_mean_shape(1,K_);
      weight_mean_.Reshape(weight_mean_shape);
      vector<int> temp_mean_norm_gpu_shape(1,1);
      temp_mean_norm_gpu.Reshape(temp_mean_norm_gpu_shape);
      break;
    }
    case InterClassParameter_InterClassType_AMONG:{
      vector<int> weight_mean_shape(1,K_);
      weight_mean_.Reshape(weight_mean_shape);
      vector<int> temp_mean_norm_gpu_shape(1,1);
      temp_mean_norm_gpu.Reshape(temp_mean_norm_gpu_shape);
      vector<int> weight_wise_diff_shape(1,K_);
      weight_wise_diff_.Reshape(weight_wise_diff_shape);
      vector<int> weight_wise_dist_sq_shape(1,1);
      weight_wise_dist_sq_.Reshape(weight_wise_dist_sq_shape);


      break;
    }
    default:{
      LOG(FATAL) << "Unknown InterClassType.";
    }
  }
}

template <typename Dtype>
void InterClassLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  iter_ += (Dtype)1.;

  Dtype alpha_start_iter_ = this->layer_param_.inter_class_param().alpha_start_iter();
  Dtype alpha_start_value_ = this->layer_param_.inter_class_param().alpha_start_value();
  Dtype alpha_step_ = this->layer_param_.inter_class_param().alpha_step();
  Dtype alpha_stepvalue_size = this->layer_param_.inter_class_param().alpha_stepvalue_size();
  Dtype normalize_ = this->layer_param_.inter_class_param().normalize();
  if (alpha_stepvalue_size != 0){
    const int* alpha_stepvalue_data = alpha_stepvalues.cpu_data();
    if (alpha_start_iter_ == iter_){
      alpha_ = alpha_start_value_;
    }
    else if(alpha_start_iter_ < iter_) {
      if(alpha_stepvalue_data[alpha_index_] == iter_ && alpha_index_<alpha_stepvalue_size){
        alpha_ += alpha_step_;
        alpha_index_ += (Dtype)1.;
      }
    }
  }
  else{
    if (alpha_start_iter_ == iter_){
      alpha_ = alpha_start_value_;
    }
  }
  if (top.size() == 2) {
    top[1]->mutable_cpu_data()[0] = alpha_;
  }

  ///************************* normalize weight *************************/

  // weight_mean_ = (\Sigma w) / N_ 
  Dtype* mutable_weight_mean_data = weight_mean_.mutable_cpu_data();
  Dtype* mutable_weight_tmp = this->blobs_[0]->mutable_cpu_data();
  
  caffe_copy(K_,mutable_weight_tmp, mutable_weight_mean_data);
  for (int i = 1; i < N_; i++) {
    caffe_add(K_, mutable_weight_mean_data,
              mutable_weight_tmp + i * K_, mutable_weight_mean_data);
  }
  caffe_scal(K_, (Dtype)1. / N_, mutable_weight_mean_data);
  if(normalize_){
    // weight_mean_norm = weight_mean_ / ||weight_mean_||
    Dtype temp_mean_norm = (Dtype)0.;
    temp_mean_norm = caffe_cpu_dot(K_, mutable_weight_mean_data, mutable_weight_mean_data);
    temp_mean_norm = (Dtype)1./sqrt(temp_mean_norm + (Dtype)1e-5);
    caffe_scal(K_, temp_mean_norm, mutable_weight_mean_data);
  }

  /************************* Forward *************************/
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype inter_class_dist = (Dtype)0.;
  Dtype tmp_dist = (Dtype)0.;
  const Dtype* weight_mean_data =  weight_mean_.cpu_data();

  /************************* Hyperspherical Energy, Alias Inter Class Loss *************************/
  switch (type_) {
    case InterClassParameter_InterClassType_MEAN: {
      for (int i = 0; i < M_; i++) {
        const int label_value = static_cast<int>(label[i]);
        tmp_dist -= caffe_cpu_dot(K_, weight_mean_data, weight + label_value * K_);
      }
      
      inter_class_dist = tmp_dist / M_ + (Dtype)1.; 
      break;
    }
    case InterClassParameter_InterClassType_AMONG: {
      for(int i = 0; i < N_; i++){
        for(int j = 0 ; j < M_; j++){
          const int label_value = static_cast<int>(label[j]);
          caffe_sub(K_, weight + i * K_, weight + label_value * K_, weight_wise_diff_.mutable_cpu_data());
          tmp_dist += caffe_cpu_dot(K_, weight_wise_diff_.cpu_data(), weight_wise_diff_.cpu_data());
        }
      }
      //inter_class_loss = (Dtype)1. * N_ / (tmp_dist + (Dtype)1e-5); // minimize inter_class_loss
      inter_class_dist = tmp_dist / (Dtype)N_; 
      weight_wise_dist_sq_.mutable_cpu_data()[0] = inter_class_dist; // storage dist
      break;
    }
    default: {
      LOG(FATAL) << "Unknown InterClassType.";
    }
  }
  // In order to monitor the [inter class dist] more easily and speed up in GPU version,
  // we output the [inter class dist] in log but not the [inter class loss].
  top_data[0] = inter_class_dist;
}

template <typename Dtype>
void InterClassLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* label = bottom[1]->cpu_data();
  Dtype* mutable_weight_diff = this->blobs_[0]->mutable_cpu_diff();

  // Gradient with respect to weight
  if (this->param_propagate_down_[0]) {
    switch (type_) {
      case InterClassParameter_InterClassType_MEAN: {
        Dtype* mutable_weight_mean_data = weight_mean_.mutable_cpu_data();
        for (int i = 0; i < M_; i++) {
          const int label_value = static_cast<int>(label[i]);
          caffe_cpu_axpby(K_, (Dtype)1.*alpha_/(Dtype)M_, mutable_weight_mean_data, 
                          (Dtype)1.,mutable_weight_diff + label_value * K_);
        }
        break;
      }
      case InterClassParameter_InterClassType_AMONG: {
        const Dtype* weight_wise_dist_sq = weight_wise_dist_sq_.cpu_data();
        const Dtype* weight = this->blobs_[0]->cpu_data();
        Dtype* mutable_weight_mean_data = weight_mean_.mutable_cpu_data();
        Dtype temp_coff = caffe_cpu_dot(1,weight_wise_dist_sq,weight_wise_dist_sq);
        Dtype total_coff = (Dtype)-4. * (Dtype)alpha_ / ((Dtype)M_ * temp_coff + (Dtype)1e-5);
        for (int i = 0; i < M_; i++) {
          const int label_value = static_cast<int>(label[i]);
          caffe_sub(K_, weight+ label_value * K_, mutable_weight_mean_data, 
            weight_wise_diff_.mutable_cpu_data());
          caffe_cpu_axpby(K_,total_coff, weight_wise_diff_.cpu_data(), 
            (Dtype)1., mutable_weight_diff + label_value * K_);
        }
        break;
      }
      default: {
        LOG(FATAL) << "Unknown InterClassType.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InterClassLayer);
#endif

INSTANTIATE_CLASS(InterClassLayer);
REGISTER_LAYER_CLASS(InterClass);

}  // namespace caffe
