# Data Preparation

## About Sherlock Data Set

This papers used [Sherlock data set](http://bigdata.ise.bgu.ac.il/sherlock/). You could contact BGU team to get the original data. Due to the copyright issue, we only provide a sample (10 entries) of our data, through which you could get an idea about the data set's format. Please contact the BGU team to get the full data if needed.

## Brief Intro

### trajectory_data/

Contains the raw data under each granularity (already splitted into train / test / validation set). Each row is a pair of historical trajectory and future trajectory.

Explanation of each column (seperated by tab) (more details can be found in Section 5.2.1 in the paper):

1. User ID
2. Start timestamp of the entire trajectory
3. End timestamp of the entire trajectory
4. Historical trajectory (T_wh), seperated by comma
5. Future trajectory (T_wf), seperated by comma
6. Corresponding timestamps of T_wh, seperated by comma
7. Corresponding timestamps of T_wf, seperated by comma

### context_data/

Context data used for multi-modal learning (see Figure 7). Please refer to the related code for more details. 

### pretrained_logits/

Pre-trained logits for multi-modal learning (see Figure 7(c)). The logits are categorized into 3 parts and you don't need to know the details for now. Check the training code if you really need to take everything in control :).

