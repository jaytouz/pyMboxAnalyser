def log_list_of_tuple(filename, list_tuple, value='url', idx='email_id'):
    with open(f'{filename}.txt', 'w') as f:
        f.writelines(f'{idx},{value}\n')
        for ur in list_tuple:
            f.writelines(f"{ur[0]},{ur[1].strip()}\n")

    print(f'{filename} created')


def log_text(filename, theme_log):
    with open(f'{filename}.txt', 'w', errors='backslashreplace') as f:
        for d in theme_log:
            f.writelines(f"{d[0]},{d[1]}{d[2]}\n")
    print(f'{filename} created')


class DataLogger:

    def __init__(self, header):
        self.rows = []
        self.rows.append(','.join(header))

    def add_row(self, data: list):
        row = ','.join(data)
        self.rows.append(row)

    def log_to(self, filename, ext='csv'):
        with open(f'{filename}.{ext}', 'w', errors='backslashreplace') as f:
            for r in self.rows:
                f.writelines(self.rows)
