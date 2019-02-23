#include <vector>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/inter_class_layer.hpp"

//#include "stdio.h"

namespace caffe {


template <typename Dtype>
__global__ void Weight_mean_gpu(int nthreads, const int K_, const int N_,
          const Dtype* weight, Dtype* weight_mean) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    for (int i = 0; i < K_; i++) {
      weight_mean[i] += weight[index * K_ + i];
    }
  }
}

template <typename Dtype>
__global__ void Weight_mean_normA_gpu(int nthreads, Dtype* temp_mean_norm, Dtype* weight_mean) {
  temp_mean_norm[0] = (Dtype)0.;
  CUDA_KERNEL_LOOP(index, nthreads) {
    temp_mean_norm[0] += weight_mean[index] * weight_mean[index];
  }
}
template <typename Dtype>
__global__ void Weight_mean_normB_gpu(int nthreads,Dtype* temp_mean_norm,  Dtype* weight_mean) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    weight_mean[index] = weight_mean[index] / sqrt(temp_mean_norm[0] + (Dtype)1e-5);
  }
}

// We have inter_class_loss == pow(inter_class_dist, -1)

/************ Inter class type: Mean ************/
template <typename Dtype>
__global__ void InterClassMean_forward_gpu(int nthreads, const int K_, Dtype alpha_,
            const Dtype* label, const Dtype * weight_mean_data, const Dtype * weight,
            Dtype* inter_class_dist) {
  inter_class_dist[0] = (Dtype)0.; //initialized top[0]
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    Dtype cosine_dist = (Dtype)0.;
    for(int i = 0; i < K_; i++){
      cosine_dist += weight_mean_data[i] * weight[label_value * K_ +i];
    }
    inter_class_dist[0] += (Dtype)1. / (Dtype)nthreads - cosine_dist / (Dtype)nthreads; 
  }
  //inter_class_loss = (Dtype)1. / inter_class_dist[0]; 
}

template <typename Dtype>
__global__ void InterClassMean_not_forward_gpu(int nthreads, const int K_, Dtype alpha_,
            const Dtype* label, const Dtype * weight_mean_data, const Dtype * weight,
            Dtype* inter_class_dist) {
  inter_class_dist[0] = (Dtype)0.; // initialized top[0]
}


template <typename Dtype>
__global__ void InterClassMean_backward_gpu(int nthreads, const int K_, Dtype alpha_,
            const Dtype* label, const Dtype * weight_mean_data, Dtype * weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    for(int i = 0; i < K_; i++){
      weight_diff[label_value * K_ + i] += (Dtype)1.*alpha_*weight_mean_data[i]/nthreads;
    }
  }
}
/****end****** Inter class type: Mean ************/


/************ Inter class type: Among ************/
template <typename Dtype>
__global__ void InterClassAmong_forward_gpu(int nthreads, const int K_, Dtype alpha_,
            Dtype* weight_wise_dist_sq, Dtype* weight_wise_diff_data,
            const Dtype * weight, Dtype* inter_class_dist) {
  // inter_class_dist approximates a constant 
  // because weights have been normalized and the number of weights is large enough
  inter_class_dist[0] = (Dtype)0.;
  Dtype tmp = (Dtype)0.;
  CUDA_KERNEL_LOOP(i, nthreads) {
    for (int j = 0; j < nthreads; j++){
      if(i != j){
        for (int k = 0; k < K_; k++){
          tmp += pow((weight[i * K_ + k] - weight[j * K_ + k]),2);
        }
      }
    }
  }
  inter_class_dist[0] = tmp / (Dtype)nthreads; 
  weight_wise_dist_sq[0] = tmp / (Dtype)nthreads; //storage dist
  //inter_class_loss = (Dtype)1. * nthreads / (tmp + (Dtype)1e-5); // minimize inter_class_loss
}


template <typename Dtype>
__global__ void InterClassAmong_batch_forward_gpu(int nthreads, const int K_, Dtype alpha_,
            Dtype* weight_wise_dist_sq, Dtype* weight_wise_diff_data,
            const Dtype * weight, Dtype* inter_class_dist,const Dtype* label,
            const int M_) {
  inter_class_dist[0] = (Dtype)0.;
  weight_wise_dist_sq[0] = (Dtype)0.;
  CUDA_KERNEL_LOOP(i, nthreads) {
    Dtype tmp = (Dtype)0.;
    for (int j = 0; j < M_; j++){
      const int label_value = static_cast<int>(label[j]);
      for (int k = 0; k < K_; k++){
        tmp += pow((weight[i * K_ + k] - weight[label_value * K_ + k]),2);
      }
    }
    inter_class_dist[0] += tmp / (Dtype)nthreads; 
    weight_wise_dist_sq[0] += tmp / (Dtype)nthreads; //storage dist
    //inter_class_loss += (Dtype)1. * nthreads / (tmp + (Dtype)1e-5); //minimize inter class loss
  }
}


template <typename Dtype>
__global__ void InterClassAmong_not_forward_gpu(int nthreads, const int K_, Dtype alpha_,
            Dtype* weight_wise_dist_sq, Dtype* weight_wise_diff_data,
            const Dtype * weight, Dtype* inter_class_dist) {
  inter_class_dist[0] = (Dtype)0.;
}

//template <typename Dtype> // too slow O(n^3)
//__global__ void InterClassAmong_backward_gpu(int nthreads, const int K_, Dtype alpha_,
//          const Dtype* weight, const Dtype* weight_wise_dist_sq, Dtype * weight_diff,
//          const Dtype* inter_class_dist) {
//  //Dtype temp_coff = pow(weight_wise_dist_sq[0],2);
//  //Dtype total_coff = (Dtype)-4. * alpha_ / (temp_coff + (Dtype)1e-5);
//  Dtype total_coff = (Dtype)-4. * alpha_;
//  CUDA_KERNEL_LOOP(i, nthreads) {
//    for (int j = 0; j < nthreads; j++){
//      if(i != j){
//        for (int k = 0; k < K_; k++){
//          weight_diff[i * K_ + k] += (weight[i * K_ + k] - weight[j * K_ + k]) * total_coff; 
//        }
//      }
//    }
//  }
//}

template <typename Dtype> // faster implement O(n^2)
__global__ void InterClassAmong_backward_gpu(int nthreads, const int K_, Dtype alpha_,
          const Dtype* weight, const Dtype * weight_mean_data, const Dtype* weight_wise_dist_sq,
          Dtype * weight_diff) {
  Dtype temp_coff = pow(weight_wise_dist_sq[0],2);
  Dtype total_coff = (Dtype)-4. * alpha_ / (temp_coff + (Dtype)1e-5);
  CUDA_KERNEL_LOOP(index, nthreads) {
    for (int i = 0; i < K_; i++){
      weight_diff[index * K_ + i] += total_coff * (weight[index * K_ + i] - weight_mean_data[i]);
    }
  }
}

template <typename Dtype> // faster implement O(n^2) // only minibatch
__global__ void InterClassAmong_batch_backward_gpu(int nthreads, const int K_, Dtype alpha_,
          const Dtype* weight, const Dtype * weight_mean_data, const Dtype* weight_wise_dist_sq,
          Dtype * weight_diff,const Dtype* label) {
  Dtype temp_coff = pow(weight_wise_dist_sq[0],2);
  Dtype total_coff = (Dtype)-4. * (Dtype)alpha_/((Dtype)nthreads* temp_coff + (Dtype)1e-5);
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    for (int i = 0; i < K_; i++){
      weight_diff[label_value * K_ + i] += total_coff * (weight[label_value * K_ + i] - weight_mean_data[i]);
    }
  }
}

/****end***** Inter class type: Among ************/



template <typename Dtype>
void InterClassLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  int nthreads = N_;
  
  switch (type_) {
    case InterClassParameter_InterClassType_MEAN:{
      // weight_mean_ = (\Sigma w) / N_ 
      nthreads = N_;
      Weight_mean_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, N_, weight,
                                    weight_mean_.mutable_gpu_data());
      
      // weight_mean_norm = weight_mean_ / ||weight_mean_||
      if(normalize_){
      nthreads=K_;
      Weight_mean_normA_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, temp_mean_norm_gpu.mutable_gpu_data(), 
                                    weight_mean_.mutable_gpu_data());
      Weight_mean_normB_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, temp_mean_norm_gpu.mutable_gpu_data(), 
                                    weight_mean_.mutable_gpu_data());
      }
      
      // compute inter_class_dist
      if(iter_ % 10 == 1){
        nthreads = M_;
        const Dtype* weight_mean_data = weight_mean_.gpu_data();
        InterClassMean_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, label, weight_mean_data,
                                     weight, top_data);
      }
      else{
        nthreads = M_;
        const Dtype* weight_mean_data = weight_mean_.gpu_data();
        //not compute inter_class_dist
        InterClassMean_not_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, label, weight_mean_data,
                                     weight, top_data); 
      }
      break;
    }
    case InterClassParameter_InterClassType_AMONG:{
      nthreads = N_;
      Weight_mean_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                  CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, N_, weight,
                                weight_mean_.mutable_gpu_data());
      if(normalize_){
      nthreads=K_;
      Weight_mean_normA_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, temp_mean_norm_gpu.mutable_gpu_data(), 
                                    weight_mean_.mutable_gpu_data());
      Weight_mean_normB_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, temp_mean_norm_gpu.mutable_gpu_data(), 
                                    weight_mean_.mutable_gpu_data());
      }
      if(iter_ % 10 == 1){ // iter_size == 1
      // because forward propagation is very slow
      // use the same interclass_dist 10 times when back propagation 

      //if(iter_%20==1 ||iter_%20==2){ // iter_size == 2


        //compute inter_class_dist
        nthreads = N_;

        // computing all weights is very slow
        //InterClassAmong_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        //    CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, weight_wise_dist_sq_.mutable_gpu_data(),
        //                              weight_wise_diff_.mutable_gpu_data(), weight, top_data);


        // computing weights of minibatch approximates computing all weights
        InterClassAmong_batch_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, weight_wise_dist_sq_.mutable_gpu_data(),
                                      weight_wise_diff_.mutable_gpu_data(), weight, top_data,label,M_);                                      
      }//
      else{
        // not compute inter_class_dist
        // use the same interclass_dist, saving in weight_wise_dist_sq, 10 times when back propagation 
        nthreads = N_;
        InterClassAmong_not_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, weight_wise_dist_sq_.mutable_gpu_data(),
                                      weight_wise_diff_.mutable_gpu_data(), weight, top_data);
      }
      break;
    }
    default:{
      LOG(FATAL) << "Unknown InterClassType.";
    }
  }
}

template <typename Dtype>
void InterClassLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();

  if (this->param_propagate_down_[0]) {
    switch (type_) {
      case InterClassParameter_InterClassType_MEAN:{
        const Dtype* weight_mean_data = weight_mean_.gpu_data();
        int nthreads = M_;
        // maximize inter_class_dist
        InterClassMean_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, label,
                                   weight_mean_data, weight_diff);
        break;
      }
      case InterClassParameter_InterClassType_AMONG:{
        int nthreads = N_;
        // maximize inter_class_dist O(n^3)
        //InterClassAmong_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        //    CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, weight, weight_wise_dist_sq_.gpu_data(), 
        //                              weight_diff, top_data);

        // maximize inter_class_dist O(n^2)
        //InterClassAmong_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        //    CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, weight, weight_mean_.gpu_data(), 
        //                              weight_wise_dist_sq_.gpu_data(), weight_diff);
        
        // only update minibatch inter_class_dist
        nthreads = M_;
        const Dtype* weight_mean_data = weight_mean_.gpu_data();
        InterClassAmong_batch_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, alpha_, weight, weight_mean_.gpu_data(), 
                                      weight_wise_dist_sq_.gpu_data(), weight_diff, label);
        break;
      }
      default:{
        LOG(FATAL) << "Unknown InterClassType.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InterClassLayer);

}  // namespace caffe
