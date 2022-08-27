import logging

logging.basicConfig(
    level = logging.DEBUG, 
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt = '%H:%M:%S',
    filename = 'scripts/console.log',
    filemode = 'w',
    )

console = logging.StreamHandler() # define a Handler which writes INFO messages or higher to the sys.stderr
console.setLevel(logging.INFO) # set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # tell the handler to use this format
console.setFormatter(formatter) # add the handler to the root logger
logging.getLogger().addHandler(console)

logger_dict = {
    'train' : logging.getLogger('train'),
    }