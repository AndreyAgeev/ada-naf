from datetime import datetime


class FileLogger:
    def __init__(self):
        super().__init__()
        self._file = None

    def setup(self, kwargs):
        args_string = '_'.join(['{}_{}'.format(key, value) for key, value in kwargs.items()])
        self.filename = 'output_{}_{}.txt'.format(args_string, datetime.now().strftime('%Y%m%d_%H%M%S'))

        self._file = open(self.filename, "a")

    def info(self, value: str):
        """Process function."""
        self._file.write(value + "\n")
        self._file.flush()

    def log(self, name: str, value: float):
        self._file.write(name + " - " + str(value) + "\n")
        self._file.flush()

    def log_artifact(self, file_name: str):
        pass

    def start_logger(self, name_value: str) -> None:
        self._file.write(f"Start logger: {name_value}\n")
        self._file.flush()

    def end_logger(self, name_value: str) -> None:
        self._file.write(f"End logger: {name_value}\n")
        self._file.flush()
        self._file.close()
