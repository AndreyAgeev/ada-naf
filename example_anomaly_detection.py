from tools.anomaly_detection import AnomalyDetection


if __name__ == "__main__":
    anomaly_detection = AnomalyDetection(num_seeds=3,
                                         num_cross_val=3,
                                         num_trees=100,
                                         count_epoch=50,
                                         contaminations=0)
    # anomaly_detection.start_model_other("IF")
    # anomaly_detection.start_model_other("DIF")
    # anomaly_detection.start_model_naf("AUTOENCODER-1-LAYER")
    # anomaly_detection.start_model_naf("NAF-1-LAYER")
    # anomaly_detection.start_model_naf("NAF-3-LAYER")
    # anomaly_detection.start_model_naf("NAF-MH-3-HEAD-1-LAYER")
    #
    # anomaly_detection.start_model_naf_impact_rf("NAF-1-LAYER")
    # anomaly_detection.start_model_naf_impact_rf("NAF-3-LAYER")
    # anomaly_detection.start_model_naf_impact_rf("NAF-MH-3-HEAD-1-LAYER")
    #
    # anomaly_detection.start_model_naf_injection("NAF-1-LAYER")
    # anomaly_detection.start_model_naf_injection("NAF-3-LAYER")
    # anomaly_detection.start_model_naf_injection("NAF-MH-3-HEAD-1-LAYER")

    anomaly_detection._file_logger.end_logger("END")
