import logging as log

log.basicConfig(
    level=log.INFO,
    format='%(message)s',
    handlers=[log.StreamHandler()]
)
