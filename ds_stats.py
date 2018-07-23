import json

def _print_stats(statistics):
    max_value = 0
    tot_entries = 0
    for key, value in statistics.items():
        tot_entries += value
        if value > max_value:
            max_value = value
            max_cat = key
    for key, value in sorted(statistics.items()):
        if key == max_cat:
            print("\033[1m", end="")
        print(" * {}: {}/{} ({:2.3%})".format(key, value, tot_entries, value / tot_entries), end="")
        if key == max_cat:
            print("\033[0m", end="")
        print("")

def _compute_stats(json_filename):

    print("Computing statistics for " + json_filename + ":")

    cnt_categories = {}
    cnt_supcat = {}

    with open(json_filename) as json_file:
        json_data = json.load(json_file)

    for key, value in json_data.items():
        for v in value:
            category = v["label"]

            # NOTE: Skipping entries starting with G2, for the current
            # version of the dataset, they can be considered as noise.
            if category.startswith("G2"):
                continue

            if category in cnt_categories:
                cnt_categories[category] += 1
            else:
                cnt_categories[category] = 1

            supcat_end_idx = category.find(":")
            supcat = category[:supcat_end_idx]
            if supcat in cnt_supcat:
                cnt_supcat[supcat] += 1
            else:
                cnt_supcat[supcat] = 1

    print("Super-categories:")
    _print_stats(cnt_supcat)

    print("Categories:")
    _print_stats(cnt_categories)


if __name__ == "__main__":
    training_filename = "datasets/Dataset_PatternRecognition/json/dataset_training.json"
    testing_filename = "datasets/Dataset_PatternRecognition/json/dataset_testing.json"

    _compute_stats(training_filename)
    _compute_stats(testing_filename)
