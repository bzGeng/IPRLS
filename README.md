# IPRLS code for Iterative Pruning with Regularization for Lifelong Sentiment Classification
## requirements

- Python >=3.7
- Pytorch 1.2.0
- transformers

## bert-base-uncased version BERT model need to be download from https://huggingface.co/bert-base-uncased , set it under path BERT/

### You can run IPRLS with

```bash
$ bash experiment/run_IPRLS.sh 
```

### After completing the above process, you need to run following bash to obtain final results

```bash
$ bash experiment/eval_middle_results.sh 
```
###  Run IPRLS with random task order
```bash
$ bash experiment/run_with_random_task_order.sh 
```
###  To evaluate shuffle order results
```bash
$ bash experiment/eval_shuffle_middle_results.sh 
```