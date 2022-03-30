# Measurement Source Code
We use the scripts to analyze the predictability of server failures due to DRAM errors as well as the impacting factors on server failures.

## Prerequisite
Python 3: please install [numpy](https://numpy.org/) and [pandas](https://pandas.pydata.org/)

## Usage
+ The raw data is stored under `../data/`
	+ `python3 measurement.py ../data/`
+ The results will be stored under `./result/`

## Functions
+ `overall_distribution`: output the percentage of servers with CEs and percentage of servers with server failures per month in Finding 1.
	+ Results: `overall_distribution.txt`
+ `predictable_analysis`: output the relative percentage of predictable server failures for different prediction windows in Finding 2.
	+ Results: `predictable_analysis.txt` 
+ `num_ce_analysis`: output the average number of CEs for different types of server failures with different prediction windows in Finding 3.
	+ Results: `num_ce_analysis.txt`
+ `mtbe_analysis`: output the median mean time between errors (MTBE) per predictable server failures for different types of failures with different prediction windows in Finding 4 
	+ Results: `mtbe_analysis.txt`
+ `frac_failure_per_component`: output the relative fraction of predictable servers that associated with different memory subsystem component failures for different types of failures when the prediction window is five minutes in Finding 5
	+ Results: `frac_failure_per_component_5m.txt`
+  `frac_ce_per_component`: output the relative fraction of CEs that are associated with different memory subsystem component failures for different typs of failures whehn the prediction window is five minute in Finding 5
	+ Results: `frac_ce_per_component_5m.txt`
+ `hardware_configuration_impact_analysis`: output the relative percentage of predictable server failures breakdown by different hardware configures factors in Findings 6-8
	+ Results:
		+ `DRAM_model_breakdown.txt` for DRAM models
		+ `DIMM_number_breakdown.txt` for number of attached DIMMs per server
		+ `server_manufacturer_breakdown.txt` for server manufacturer
+ `read_scrubbing_analysis`: output the average number of read errors and scrubbing errors per predictable server failures for different types of server failures when the prediction window is five minutes in Finding 9
	+ Results: `read_error_mean.txt` and `scrub_error_mean.txt`)
+ `hard_soft_analysis`: output the average number of hard errors and soft errors per predictable server failures for different typs of server failures when the prediction window is five minutes in Finding 10.
	+ Results: `hard_erorr_mean.txt` and `soft_error_mean.txt` 

## Contact
Please email to Zhinan Cheng (zncheng@cse.cuhk.edu.hk) if you have any questions.
