import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.metrics.cluster import pair_confusion_matrix
from medpy.metric.binary import hd as HD
from medpy.metric.binary import assd as ASSD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score as ARI
import time

class Metrics():
    def __init__(self, metrics=["dsc", "hd", "icc", "ari", "pbd", "vs", "sensitivity", "specificity", "precision", "npv", "acc", "assd"], no_bin_check=False):
        """
        Params:
            metrics: A list of metrics that shall be computed. Possible values are:
                DSC, HD, ICC, ARI, sensitivity, specificity, precision, NPV, ACC
        """
        self.results = {}
        self.no_bin_check = no_bin_check
        self.available_metrics = ["dsc", "hd", "icc", "ari", "pbd", "vs", "sensitivity", "specificity", "precision", "npv", "acc", "assd"]
        if not isinstance(metrics, list):
            metrics = list(metrics)
        self.metrics = set([x.lower() for x in metrics])

    def compute_metrics(self, in1, in2, voxelspacing=(1., 1., 1.)):
        """
        Params:
            in1, in2: label maps. Must be numpy array including only ones and zeros.
        Returns:
            results: Dictionary with results. Includes values:
                DSC, HD, ICC, ARI, sensitivity, specificity, precision, NPV, ACC
        """

        # Convert to numpy array.
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)

        assert in1.min() >= 0
        assert in1.max() <= 1
        assert in2.min() >= 0
        assert in2.max() <= 1
        assert np.unique(in1).shape[0] <= 2
        assert np.unique(in2).shape[0] <= 2

        # Check data shapes.
        if in1.shape != in2.shape:
            raise ValueError("Shape mismatch: in1 and in2 must have the same shape, but have {} and {}.".format(in1.shape, in2.shape))

        # Convert to binary label maps.
        b1 = in1.astype(np.bool)
        b2 = in2.astype(np.bool)

        # Check if input is binary.
        if not self.no_bin_check:
            if not np.array_equal(b1, in1):
                print("First input is not binary! This may lead to wrong computations!")
            if not np.array_equal(b2, in2):
                print("Second input is not binary! This may lead to wrong computations!")

        # Define values that can be reused.
        if (set(["dsc"]) & self.metrics):
            b1_sum = b1.sum()
            b2_sum = b2.sum()
        if (set(["icc", "pbm"]) & self.metrics):
            b1_mean = b1.mean()
            b2_mean = b2.mean()

        if (set(["sensitivity", "specificity", "precision", "npv", "acc", "ari", "icc", "vs"]) & self.metrics):
            b1_f, b2_f = b1.flatten(), b2.flatten()

        if (set(["sensitivity", "specificity", "precision", "npv", "acc", "vs"]) & self.metrics):
            tp = np.sum(np.logical_and(b1_f == 1, b2_f == 1), dtype=np.float64)
            fn = np.sum(np.logical_and(b1_f == 1, b2_f == 0), dtype=np.float64)
            tn = np.sum(np.logical_and(b1_f == 0, b2_f == 0), dtype=np.float64)
            fp = np.sum(np.logical_and(b1_f == 0, b2_f == 1), dtype=np.float64)

        # Dice-Sørensen Coefficient (DSC).
        if "dsc" in self.metrics:
            intersection = np.logical_and(b1, b2)
            try:
                self.results["DSC"] = 2. * intersection.sum() / (b1_sum + b2_sum)
            except ZeroDivisionError:
                self.results["DSC"] = 0.0

        # Hausdorff Distance (HD).
        if "hd" in self.metrics:
            self.results["HD"] = HD(b1, b2, voxelspacing=voxelspacing)

        # Interclass Correlation (ICC).
        if "icc" in self.metrics:
            grandmean = (b1_mean + b2_mean) / 2
            n_elements = len(b1)

            m = np.mean([b1_f, b2_f], axis=0)
            ssw = np.sum(np.sum([np.power(b2_f - m, 2), np.power(b1_f - m, 2)], axis=0), axis=0)
            ssb = np.sum(np.power(m - grandmean, 2), axis=0)

            ssw = ssw / n_elements
            ssb = ssb / (n_elements - 1) * 2
            self.results["ICC"] = (ssb - ssw)/(ssb + ssw + 0.0000001)

        # Adjusted Rand Index (ARI)
        if "ari" in self.metrics:
            (TN, FP), (FN, TP) = pair_confusion_matrix(b1_f, b2_f)

            # Special cases: empty data or full agreement
            if FN == 0 and FP == 0:
                self.results["ARI"] = 1.0
            else:
                TN = TN.astype(np.float64)
                TP = TP.astype(np.float64)
                FP = FP.astype(np.float64)
                FN = FN.astype(np.float64)

                self.results["ARI"] = 2. * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

        # Probablistic distance metric.
        if "pbd" in self.metrics:
            in1_f = in1.flatten().astype(np.int16)
            in2_f = in2.flatten().astype(np.int16)
            prob_joint = np.sum(np.multiply(in1_f, in2_f))
            prob_diff = np.sum(np.abs(np.subtract(in1_f, in2_f)))

            self.results["PBD"] = -1
            if prob_joint != 0:
                self.results["PBD"] = prob_diff / (2 * prob_joint)

        # Volumetric Similarity.
        if "vs" in self.metrics:
            self.results["VS"] = 1 - abs(fn - fp) / (2*tp + fp + fn)

            
        # Sensitivity, Recall, True positive rate, hit rate.
        if "sensitivity" in self.metrics:
            self.results["sensitivity"] = tp / (tp + fn)

        # Specificity, selectivity, true negative rate.
        if "specificity" in self.metrics:
            self.results["specificity"] = tn / (tn + fp)

        # Precision, positive predictive value
        if "precision" in self.metrics:
            self.results["precision"] = tp / (tp + fp)

        # Negative predictive value (NPV)
        if "npv" in self.metrics:
            self.results["NPV"] = fn / (fn + tp)

        # Accuracy (ACC)
        if "acc" in self.metrics:
            self.results["ACC"] = (tp + tn) / (tp + tn + fp + fn)

        # Average symmetric surface distance
        if "assd" in self.metrics:
            self.results["ASSD"] = ASSD(b1, b2, voxelspacing=voxelspacing)

        return self.results.copy()