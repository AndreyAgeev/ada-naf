from tools.anomaly_detection import AnomalyDetection


if __name__ == "__main__":
    anomaly_detection = AnomalyDetection(num_seeds=3,
                                         num_cross_val=3,
                                         num_trees=100,
                                         count_epoch=50,
                                         contaminations=0)
    anomaly_detection.start_model_naf("ADA-NAF-1-LAYER")
    anomaly_detection._file_logger.end_logger("END")
