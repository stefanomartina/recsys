def list_ID_stats(ID_list, label):
    min_val = min(ID_list)
    max_val = max(ID_list)
    unique_value = len(set(ID_list))
    missing_value = 1 - unique_value / (max_val - min_val)

    print("{} data, ID: min {}, max {}, unique {}, missing {:.2f} %".format(label, min_val, max_val, unique_value, missing_value * 100))