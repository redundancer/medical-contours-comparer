from os import path
import numpy as np
import os, sys, math, csv
from os.path import join
from metrics import Metrics
import SimpleITK as sitk
import argparse

def compare_files_in_folder(root_dir, substring_gt, substring_pred):
    root_dir = join(root_dir)
    
    folders = os.listdir(root_dir)
    m = Metrics(["DSC", "HD", "ARI", "ASSD", "VS", "PBD"])
    file_path_tuples = get_files_in_folders_with_substrings(root_dir, folders, substring_gt, substring_pred)
    results = compute_metrics(file_path_tuples, m)
    root_dir_name = os.path.basename(os.path.normpath(root_dir))
    results_path_name = join("results", root_dir_name)
    write_summarized_results_into_file(results, results_path_name)
    write_detailed_results_into_file(results, results_path_name)
    
def write_summarized_results_into_file(results, foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filename = foldername + "/summarized_results.csv"

    attributes = [""]
    t_min = ["min"]
    t_max = ["max"]
    t_mean = ["mean"]
    t_median = ["median"]
    t_std = ["std"]

    for idx, (key, val) in enumerate(results.items()):
      if idx != 0:
        attributes.append(key)
        t_min.append(round(np.min(val), 2))
        t_max.append(round(np.max(val), 2))
        t_mean.append(round(np.mean(val), 2))
        t_median.append(round(np.median(val), 2))
        t_std.append(round(np.std(val), 2))

    with open(filename, 'w') as f:
      writer = csv.writer(f, delimiter=",")
      writer.writerow(attributes)
      writer.writerow(t_min)
      writer.writerow(t_max)
      writer.writerow(t_mean)
      writer.writerow(t_median)
      writer.writerow(t_std)

def write_detailed_results_into_file(results, foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filename = foldername + "/detailed_results.csv"
    
    attributes = []
    list_of_results = []
    for key, val in results.items():
        attributes.append(key)
        list_of_results.append(tuple(val))

    list_of_results = np.array(list_of_results).transpose()

    with open(filename, 'w') as f:
      writer = csv.writer(f, delimiter=",")
      writer.writerow(attributes)
      for row in list_of_results:
        writer.writerow(row)

def compute_metrics(file_path_tuples, m):
    results = {
            "patient": [],
            "DSC": [],
            "HD": [],
            "ARI": [],
            "ASSD": [],
            "VS": [],
            "PBD": [],
          }
    for path in file_path_tuples:
        gt_itk = sitk.ReadImage(join(path[0]), sitk.sitkUInt8)

        pred_itk = sitk.ReadImage(join(path[1]), sitk.sitkUInt8)
        voxelspacing_pred = pred_itk.GetSpacing()
        pred = sitk.GetArrayFromImage(pred_itk)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pred_itk)
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        gt_itk = resampler.Execute(gt_itk)

        voxelspacing_gt = gt_itk.GetSpacing()
        gt = sitk.GetArrayFromImage(gt_itk)

        result = m.compute_metrics(gt, pred, voxelspacing_gt)
        
        patient_name = os.path.basename(os.path.normpath(join(path[0])))
        results["patient"].append(patient_name)
        for key in result.keys():
            results[key].append(result[key])
    return results


def get_files_in_folders_with_substrings(root_dir, folders, substring_gt, substring_pred):
    file_tuples = []
    for folder in folders:
        full_path = join(root_dir, folder)
        gt_files = []
        pred_files = []
        files = os.listdir(full_path)
        for f in files:
            for substring in substring_gt:
                if substring in f:
                    gt_file = join(full_path, f)
                    gt_files.append(gt_file)

            for substring in substring_pred:
                if substring in f:
                    pred_file = join(full_path, f)
                    pred_files.append(pred_file)

        if len(pred_files) != 1 or len(gt_files) != 1:
            raise ValueError("The selected substrings are not unique! Exactly one file per substring should be found per patient, but fount \n{}\n{}".format(gt_files, pred_files))
        file_tuples.append((gt_files[0], pred_files[0]))
    return file_tuples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select substrings of contour file names.')
    parser.add_argument('--root_dir', type=str, help='Path to root folder.')
    parser.add_argument('--gt', type=str, nargs='+', help='Substring for ground truth files.')
    parser.add_argument('--pred', type=str,  nargs='+', help='Substring for predicted contours.')

    args = parser.parse_args()

    compare_files_in_folder(root_dir=args.root_dir, substring_gt=args.gt, substring_pred=args.pred)
