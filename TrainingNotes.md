# SphereFace+* : Training Notes

1. Interclass distance fluctuates near a constant because of using weight norm. The value of this constant depends on the batch size.
2. When finetuning Hyperparameter 'alpha', be careful about A-softmax loss exploding. Too big value of 'alpha' causes loss exploding and too small reduces the ability of MHE regularization.
3. Value of 'alpha': Batch size 256 needs 1~10, batch size 128 needs 0.1~2 empirically. (Not sure)
4. Because of the trade-off between A-softmax-loss and MHE regularization, A-softmax loss of SphereFace+ with MHE is a little larger than SphereFace. Don't worry, it is reasonable.
5. (To be supplemented... :)