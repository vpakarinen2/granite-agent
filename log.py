import logging
import sys
import os


class ColoredConsoleFormatter(logging.Formatter):
    """Custom formatter to add ANSI colors to console logs."""
    BRIGHT_WHITE = '\033[97m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_RED = '\033[91m'
    RESET = '\033[0m'

    FORMATS = {
        logging.INFO: BRIGHT_WHITE + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
        logging.WARNING: BRIGHT_YELLOW + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
        logging.ERROR: BRIGHT_RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
        logging.CRITICAL: BRIGHT_RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
        logging.DEBUG: RESET + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name="GraniteAgent", log_file="agent.log", level=logging.INFO):
    """Initializes and returns customized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

agent_logger = setup_logger()
