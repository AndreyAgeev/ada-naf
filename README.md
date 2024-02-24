# ADA-NAF

## Prerequisites

It is recommended to use an isolated **Python 3.10** environment, like [venv](https://docs.python.org/3/library/venv.html).

For example:

```bash
python3 -m venv NAF
source NAF/bin/activate
```

## Usage
The following code is for an article about ADA-NAF and is based on https://github.com/andruekonst/NAF:
```
anomaly_detection = AnomalyDetection(num_seeds=3,
                                     num_cross_val=3,
                                     num_trees=100,
                                     count_epoch=50)
anomaly_detection.start_model_naf("NAF-1-LAYER")  # Options: "NAF-3-LAYER", "NAF-MH-3-HEAD-1-LAYER"
```