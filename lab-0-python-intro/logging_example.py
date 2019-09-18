#!/usr/bin/env python

import logging

top_level_logger = logging.getLogger('app')
subcomponent_logger = logging.getLogger('app.component')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    top_level_logger.debug('starting main function')
    top_level_logger.info('ready')
    subcomponent_logger.debug('initialising')
    subcomponent_logger.debug('ready')
