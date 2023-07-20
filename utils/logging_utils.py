import logging
import logging.handlers
from typing import Optional, Tuple
from utils.file_utils import check_path_exists


class DQNLogger:
    def __init__(self, log_file_path: str, name: Optional[str] = None, log_level: int = logging.INFO,
                 verbose: Optional[bool] = False):
        """ Initialize the logger object

        Args:
            log_file_path (str): Path to the log file
            name (str, optional): Name of the logger. Defaults to None.
            log_level (int, optional): Log level. Defaults to logging.INFO.
            verbose (bool, optional): Whether to print the logs to stdout. Defaults to False.
        """
        self.log_file_path = log_file_path
        check_path_exists(self.log_file_path)
        if name is None:
            name = __name__
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.messages = []
        self.verbose = verbose

        self._setup_file_handler()

    def _setup_file_handler(self):
        """ Set up the file handler for logging """
        try:
            file_handler = logging.handlers.RotatingFileHandler(self.log_file_path, maxBytes=1024 * 1024, backupCount=5)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up log file handler: {str(e)}")

    def log_info(self, message: str):
        """ Log an info message """
        self.logger.info(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_warning(self, message: str):
        """ Log a warning message """
        self.logger.warning(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_error(self, message: str):
        """ Log an error message """
        self.logger.error(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_critical(self, message: str):
        """ Log a critical message """
        self.logger.critical(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def print_last_message(self):
        """ Print the last message in the log file """
        if self.messages:
            print(self.messages[-1])
        else:
            print("No messages logged.")


def calculate_time(time1: float, time2: float) -> Tuple[int, int, int]:
    """
    Calculate the time elapsed between two times
    """
    total_time = int(time2 - time1)

    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60

    return hours, minutes, seconds


def get_time(time1: float, time2: float, label: Optional[str] = None):
    """
    Print the time elapsed between two times
    """
    hours, minutes, seconds = calculate_time(time1, time2)
    if label is None:
        label = "Time elapsed: "

    return f"{label} {hours:02d} hour(s) {minutes:02d} minute(s) {seconds:02d} second(s)"
