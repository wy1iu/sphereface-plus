# *SphereFace+* : Training Notes

### Training details

1. Interclass distance fluctuates near a constant because of using weight norm. The value of this constant depends on the batch size.
2. When finetuning Hyperparameter `alpha_start_value`, be careful about A-softmax loss exploding. Too big value of `alpha_start_value` causes loss exploding and too small reduces the ability of MHE regularization.
3. Value of `alpha_start_value`: Batch size 256 needs 1\~10, batch size 128 needs 0.1\~2 empirically. (Not sure)
4. Because of the trade-off between A-softmax-loss and MHE regularization, A-softmax loss of SphereFace+ with MHE is a little larger than SphereFace. Don't worry, it is reasonable.
5. When A-softmax loss explodes，kill the training and restart it. (You may need to finetune the `alpha_start_value`.)
6. Higher value of SphereFace parameter(like m=4), lower stability of training and lower gain of testing. Trying m=1 / m=2, you will find the training process is more stable. Results can be seen in README.md.
7. We highly recommend a noise-controlled dataset, [ECCV2018-IMDb-Face](https://github.com/fwang91/IMDb-Face). Interested users can try to train SphereFace+ / SphereFace on their IMDb-Face dataset. 
8. (To be supplemented... :)
