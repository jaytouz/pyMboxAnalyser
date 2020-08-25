def recursive_partition(collection):
    """https://stackoverflow.com/questions/19368375/set-partitions-in-python"""
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in recursive_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


class EmailTheme:

    def __init__(self, dict_theme=None, partition=None):
        self.theme = dict_theme
        self.partition = partition

    def set_theme_partitions(self, max_partition=10):
        """assigne les partitions de la liste de thème. O(2^n) attention"""
        partition = recursive_partition(list(self.theme.keys())[:max_partition])
        set_part = []
        for p in partition:
            if p not in set_part:
                set_part.append(p)
        self.partition = sorted(set_part)

    def save_partition(self, output_file):
        import pickle
        with open(f"{output_file}.pickle", 'wb') as f:
            pickle.dump(self.partition, f)
        print(f'dataClean instance saved to pickle at {output_file}')

    @classmethod
    def from_dict_theme(cls, dict_theme):
        return EmailTheme(dict_theme=dict_theme)

    @classmethod
    def from_pickle(cls, obj_path):
        return EmailTheme(dict_theme=obj_path.theme, partition=obj_path.partition)
