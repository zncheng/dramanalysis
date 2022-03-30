# Source Code for Measurement and Prediction
The repository includes the source code of measurement for server failures due to DRAM errors and source code of prediction for such server failures based on the dataset of DRAM errors and server failures collected at Alibaba.

## Publication
Zhinan Cheng, Shujie Han, Patrick P. C. Lee, Xin Li, Jiongzhou Liu, and Zhan Li.

**"An In-Depth Study Between DRAM Errors and Server Failures in Production Data Centers."**
Proceedings of the 40th International Symposium on Reliable Distributed Systems (SRDS 2022), September 2022.

## Directories
+ `data/` ([Dataset downloaded from Alibaba](https://github.com/alibaba-edu/dcbrain/tree/master/dramdata)), spans eight months, including
	+ `mcelog.tar.gz`: the collected mcelog that describes details of DRAM errors.
	+ `inventory.tar.gz`: the inventory log that describes the hardware configurations of used DIMMs.
	+ `trouble_ticket.tar.gz`: the trouble tickets that describes the detailed of  server failures due to DRAM errors.
+ `measurement/`: includes the source code of the predictable analysis of server failures due to DRAM errors.
+ `prediciton/`: includes the source code of our workflow for server failure prediction.

## Contact
Please email to Zhinan Cheng (zncheng@cse.cuhk.edu.hk) if you have any questions.

