# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 04-30-2023
# =============================================================================

import logging
import tqdm


class LoggingHandler(logging.Handler):
    """Simple logging handler that writes to tqdm.tqdm.write."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
