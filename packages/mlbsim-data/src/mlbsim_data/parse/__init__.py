"""Pure parsers: raw provider JSON -> lists of column-keyed row dicts.

Parsers never touch the network or the database. Their output feeds
``mlbsim_data.validate`` and then ``mlbsim_data.loaders``.
"""

from mlbsim_data.parse.boxscore import BoxscoreParse, parse_boxscore
from mlbsim_data.parse.feed import FeedParse, parse_game_feed
from mlbsim_data.parse.schedule import parse_schedule
from mlbsim_data.parse.statcast import parse_statcast_batted_balls

__all__ = [
    "BoxscoreParse",
    "FeedParse",
    "parse_boxscore",
    "parse_game_feed",
    "parse_schedule",
    "parse_statcast_batted_balls",
]
