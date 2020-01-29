import logging
FORMAT = "[%(asctime)s][%(module)s][%(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger(__name__)
