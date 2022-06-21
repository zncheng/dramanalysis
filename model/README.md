# Well-trained Models

+ This directories in this folder include all the well-trained models (saved by `pickle.dump` function) used in our paper: 
	+ Please find the models for each experiments under the following directories, using `tar xvf file.tar.xz` for decompress 
	+ `exp1/exp1_models.tar.xz`: models used in exp#1
		+ each file  with name format `exp1_month_$m_types_$t_features_$f`
			+ `$m` ranges from 1 to 3, indicating which testing month the models used for
			+ `$t` ranges from 1 to 3, indicating which failure type the models used for, 1 for UE-driven failures, 2 for CE-driven failures, and 3 for Misc failures
			+ `$f` ranges from 1 to 4, indicating the number of feature groups used for trained the models. 
	+ `exp2/exp2_models.tar.xz`: models used in exp#2
		+ each file with name format `exp2_month_$m_types_$t_predictors_$p`
			+ `$m` ranges from 1 to 3, indicating which testing month the models used for
			+ `$t` ranges from 1 to 3, indicating which failure type the models used for, 1 for UE-driven failures, 2 for CE-driven failures, and 3 for Misc failures
			+ `$p` indicates the models, LR or SVM or GBDT or MLP or RF
	+ `exp3/exp3_models.tar.xz`: models used in exp#3
		+ each file with name format `exp3_month_$m_types_$t_interval_$i`
			+ `$m` ranges from 1 to 3, indicating which testing month the models used for
			+ `$t` ranges from 1 to 3, indicating which failure type the models used for, 1 for UE-driven failures, 2 for CE-driven failures, and 3 for Misc failures
			+ `$i`, the prediction time interval, 5m or 30m or 1h or 1d
	+ `exp4/exp4_models.tar.xz`: models used in exp#4
		+ each file with name format `exp4_month_$m_types_$t`
			+ `$m` ranges from 1 to 3, indicating which testing month the models used for
			+ `$t` ranges from 1 to 3, indicating which failure type the models used for, 1 for UE-driven failures, 2 for CE-driven failures, and 3 for Misc failures
