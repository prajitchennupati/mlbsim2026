"""Play-by-play fact tables (``warehouse`` schema).

Native list-partitioning of ``plate_appearances`` / ``pitches`` by season is
planned (see docs/DATABASE.md) but deferred to a dedicated migration once it can
be validated against a live database; for now these are plain tables with a
season index.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, Numeric, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from mlbsim_core.db import Base

_SCHEMA = "warehouse"


class PlateAppearance(Base):
    __tablename__ = "plate_appearances"
    __table_args__ = (
        Index("ix_plate_appearances_game_pk", "game_pk"),
        Index("ix_plate_appearances_batter", "batter_id", "game_pk"),
        Index("ix_plate_appearances_pitcher", "pitcher_id", "game_pk"),
        Index("ix_plate_appearances_season", "season"),
        {"schema": _SCHEMA},
    )

    pa_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    game_pk: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{_SCHEMA}.games.game_pk"))
    season: Mapped[int] = mapped_column(SmallInteger)
    at_bat_index: Mapped[int] = mapped_column(SmallInteger)
    inning: Mapped[int] = mapped_column(SmallInteger)
    half: Mapped[str] = mapped_column(String(6))  # top | bottom
    batter_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.players.player_id"))
    pitcher_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.players.player_id"))
    batting_team_id: Mapped[int] = mapped_column(ForeignKey(f"{_SCHEMA}.teams.team_id"))
    bat_side: Mapped[str | None] = mapped_column(String(1))  # L | R
    pitch_hand: Mapped[str | None] = mapped_column(String(1))  # L | R
    batting_order: Mapped[int | None] = mapped_column(SmallInteger)

    outs_start: Mapped[int] = mapped_column(SmallInteger)
    outs_end: Mapped[int] = mapped_column(SmallInteger)
    base_state_start: Mapped[int] = mapped_column(SmallInteger)  # 3-bit mask, 1B=1 2B=2 3B=4
    base_state_end: Mapped[int] = mapped_column(SmallInteger)

    event_type: Mapped[str] = mapped_column(String(32))
    event: Mapped[str | None] = mapped_column(String(48))
    description: Mapped[str | None] = mapped_column(String(400))
    is_out: Mapped[bool] = mapped_column(Boolean, default=False)
    is_sac: Mapped[bool] = mapped_column(Boolean, default=False)
    rbi: Mapped[int] = mapped_column(SmallInteger, default=0)
    runs_on_play: Mapped[int] = mapped_column(SmallInteger, default=0)
    away_score_after: Mapped[int] = mapped_column(SmallInteger, default=0)
    home_score_after: Mapped[int] = mapped_column(SmallInteger, default=0)

    times_through_order: Mapped[int | None] = mapped_column(SmallInteger)
    pitcher_pitch_no: Mapped[int | None] = mapped_column(SmallInteger)  # cumulative, end of PA
    leverage_index: Mapped[float | None] = mapped_column(Numeric(5, 3))
    wpa: Mapped[float | None] = mapped_column(Numeric(6, 4))


class Pitch(Base):
    __tablename__ = "pitches"
    __table_args__ = (
        Index("ix_pitches_pa_id", "pa_id"),
        Index("ix_pitches_season", "season"),
        {"schema": _SCHEMA},
    )

    pitch_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    pa_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{_SCHEMA}.plate_appearances.pa_id"))
    game_pk: Mapped[int] = mapped_column(BigInteger)
    season: Mapped[int] = mapped_column(SmallInteger)
    seq: Mapped[int] = mapped_column(SmallInteger)  # pitch number within the PA
    balls_after: Mapped[int | None] = mapped_column(SmallInteger)
    strikes_after: Mapped[int | None] = mapped_column(SmallInteger)
    pitch_type: Mapped[str | None] = mapped_column(String(4))
    description: Mapped[str | None] = mapped_column(String(48))
    call_code: Mapped[str | None] = mapped_column(String(4))
    is_in_play: Mapped[bool] = mapped_column(Boolean, default=False)
    release_speed: Mapped[float | None] = mapped_column(Numeric(4, 1))
    end_speed: Mapped[float | None] = mapped_column(Numeric(4, 1))
    spin_rate: Mapped[int | None] = mapped_column(SmallInteger)
    spin_direction: Mapped[int | None] = mapped_column(SmallInteger)
    extension: Mapped[float | None] = mapped_column(Numeric(4, 2))
    plate_x: Mapped[float | None] = mapped_column(Numeric(5, 3))
    plate_z: Mapped[float | None] = mapped_column(Numeric(5, 3))
    zone: Mapped[int | None] = mapped_column(SmallInteger)


class BattedBall(Base):
    __tablename__ = "batted_balls"
    __table_args__ = {"schema": _SCHEMA}

    pa_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{_SCHEMA}.plate_appearances.pa_id"), primary_key=True
    )
    launch_speed: Mapped[float | None] = mapped_column(Numeric(4, 1))
    launch_angle: Mapped[float | None] = mapped_column(Numeric(4, 1))
    total_distance: Mapped[int | None] = mapped_column(SmallInteger)
    trajectory: Mapped[str | None] = mapped_column(String(16))
    hardness: Mapped[str | None] = mapped_column(String(10))
    hit_x: Mapped[float | None] = mapped_column(Numeric(6, 2))
    hit_y: Mapped[float | None] = mapped_column(Numeric(6, 2))
    # Statcast enrichment (joined from Baseball Savant, later in M1).
    xba: Mapped[float | None] = mapped_column(Numeric(4, 3))
    xwoba: Mapped[float | None] = mapped_column(Numeric(4, 3))
    is_barrel: Mapped[bool | None] = mapped_column(Boolean)
    is_hardhit: Mapped[bool | None] = mapped_column(Boolean)
