from .rt_lib import *
from pkg_resources import get_distribution

version = __version__ = get_distribution('tfrt').version
