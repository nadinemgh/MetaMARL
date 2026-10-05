"""Lake models behind the fresh-water irrigation example.

The irrigation game needs, every day, the state of one reservoir: its stage, the
flow that enters it, the flow it releases and the rain on the basin. Two
interchangeable models provide them behind the :class:`LakeModel` interface.

``RavenLake`` drives the external Raven hydrological model (Craig et al., 2020),
as the first version of this example did: it copies a prepared model directory,
rewrites the withdrawal series and the end date, runs the executable and reads
the last row of its output files. The no-withdrawal baseline never changes, so
it is run once per episode, up to the last day of the episode, and its daily
rows are read as the episode advances. The model directory (the ``raven/``
folder of the Belwood Lake catchment) and the executable are not part of this
repository, so their locations are options of the run and nothing is
hard-coded.

``SurrogateLake`` is a small mass-balance reservoir with seeded weather. It
needs no external file, which lets the example, its tests and its smoke run work
anywhere. It is **not** a calibrated hydrology: its seasonal inflow, its rain
and its release rule are the simplest ones that make the reservoir react to the
irrigation withdrawals, and its numbers are not those of the Raven model.

Both models also advance a second, parallel reservoir that receives no
withdrawal (the "baseline"). The regulator compares the filled levels of the
two to measure how far the irrigation lowered the lake away from its natural
level.

References
----------
Craig, J. R., Brown, G., Chlumsky, R., Jenkinson, R. W., Jost, G., Lee, K.,
Mai, J., Serrer, M., Sgro, N., Shafii, M., Snowdon, A. P., & Tolson, B. A.
(2020). Flexible watershed simulation with the Raven hydrological modelling
framework. Environmental Modelling & Software, 129, 104728.
https://doi.org/10.1016/j.envsoft.2020.104728
"""

import csv
import logging
import math
import shutil
import stat
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8
SECONDS_PER_DAY = 86400.0
RAVEN_ORIGIN = datetime(1980, 1, 1)
"""First day of the Raven simulation and of the withdrawal series."""

TEMP_C_BY_MONTH = {
    1: -5.0,
    2: -3.0,
    3: 2.0,
    4: 8.0,
    5: 15.0,
    6: 20.0,
    7: 23.0,
    8: 22.0,
    9: 17.0,
    10: 10.0,
    11: 4.0,
    12: -2.0,
}
"""Monthly mean air temperature (degrees Celsius) used by the crop model."""


def estimate_temp_c(date: datetime) -> float:
    """Return the monthly mean air temperature of ``date``.

    Parameters
    ----------
    date : datetime
        Day of the simulation; only its month matters.

    Returns
    -------
    float
        Mean air temperature of the month in degrees Celsius.

    When to use: whenever a temperature is needed and the hydrological model
    provides none, which is the case for both lake models.

    Examples
    --------
    >>> from datetime import datetime
    >>> estimate_temp_c(datetime(1980, 7, 14))
    23.0
    """
    return TEMP_C_BY_MONTH[date.month]


@dataclass(frozen=True)
class LakeReading:
    """State of the lake at the end of a day.

    Attributes
    ----------
    stage_m : float
        Water level of the reservoir (metres above the datum of the model).
    inflow_m3s : float
        Flow entering the reservoir (cubic metres per second).
    outflow_m3s : float
        Flow released downstream (cubic metres per second).
    precip_mm_day : float
        Rain on the basin that day (millimetres per day).
    baseline_stage_m, baseline_inflow_m3s, baseline_outflow_m3s : float
        The same three quantities for the parallel reservoir that receives no
        withdrawal.
    gauges : dict[str, float]
        Extra river gauges (cubic metres per second) the model provides, keyed
        by the field names of :class:`examples.fresh_water.metric_schema.
        WaterMetricSchema`. Empty for the surrogate.

    When to use: it is what :meth:`LakeModel.start` and :meth:`LakeModel.advance`
    return; build one by hand to test code that consumes a reading.

    Examples
    --------
    >>> reading = LakeReading(
    ...     stage_m=420.0,
    ...     inflow_m3s=10.0,
    ...     outflow_m3s=6.0,
    ...     precip_mm_day=0.0,
    ...     baseline_stage_m=420.0,
    ...     baseline_inflow_m3s=10.0,
    ...     baseline_outflow_m3s=6.0,
    ... )
    >>> reading.outflow_m3s
    6.0
    """

    stage_m: float
    inflow_m3s: float
    outflow_m3s: float
    precip_mm_day: float
    baseline_stage_m: float
    baseline_inflow_m3s: float
    baseline_outflow_m3s: float
    gauges: dict[str, float] = field(default_factory=dict)


class LakeModel(ABC):
    """Interface of the lake models: start a day, then advance it day by day.

    Parameters
    ----------
    full_stage_m : float
        Stage of the full reservoir (metres).
    max_depth_m : float
        Depth of the full reservoir, so that the empty stage is
        ``full_stage_m - max_depth_m`` (metres).

    Attributes
    ----------
    full_stage_m, max_depth_m : float
        The constructor arguments.

    When to use: subclass it to plug another hydrological model into
    :class:`examples.fresh_water.regulated_env.WaterRegulatedEnv`; use
    :class:`SurrogateLake` or :class:`RavenLake` otherwise.

    Examples
    --------
    >>> lake = SurrogateLake(full_stage_m=420.0, max_depth_m=10.0, lake_area_m2=1e6)
    >>> lake.level_norm(415.0)
    0.5
    """

    def __init__(self, *, full_stage_m: float, max_depth_m: float) -> None:
        self.full_stage_m = full_stage_m
        self.max_depth_m = max_depth_m

    def level_norm(self, stage_m: float) -> float:
        """Convert a stage into the fraction of the depth that is filled.

        Parameters
        ----------
        stage_m : float
            Water level (metres).

        Returns
        -------
        float
            ``(stage - (full_stage - max_depth)) / max_depth``: ``0`` for an
            empty reservoir and ``1`` for a full one (unclipped).
        """
        return (stage_m - (self.full_stage_m - self.max_depth_m)) / self.max_depth_m

    @abstractmethod
    def start(
        self,
        date: datetime,
        rng: np.random.Generator,
        end_date: Optional[datetime] = None,
    ) -> LakeReading:
        """Begin an episode on ``date`` with no withdrawal so far.

        Parameters
        ----------
        date : datetime
            First day of the episode.
        rng : numpy.random.Generator
            Generator of the environment, for models with random weather.
        end_date : datetime, optional
            Last day the episode can reach. Models that simulate the
            no-withdrawal baseline ahead of time need it; the others ignore it.

        Returns
        -------
        LakeReading
            State of the lake on ``date``.
        """

    @abstractmethod
    def advance(
        self, date: datetime, withdrawal_m3s: float, rng: np.random.Generator
    ) -> LakeReading:
        """Move the lake to the end of ``date``.

        Parameters
        ----------
        date : datetime
            The day that is reached (the day after the previous one).
        withdrawal_m3s : float
            Total irrigation withdrawal of the day that just ended, as a flow
            (cubic metres per second).
        rng : numpy.random.Generator
            Generator of the environment, for models with random weather.

        Returns
        -------
        LakeReading
            State of the lake on ``date``.
        """


class SurrogateLake(LakeModel):
    """Mass-balance reservoir with seeded seasonal weather, no external model.

    The reservoir holds a volume ``V`` between ``0`` and the capacity
    ``lake_area_m2 * max_depth_m``. Each day it receives the inflow ``Q_in``,
    loses the withdrawal ``W`` and releases ``release_fraction * Q_in``; what
    would overflow the capacity is released as well (a spill), and the release
    is reduced when the reservoir would run dry:

        V' = V + (Q_in - W) * 86400 - Q_out * 86400,   0 <= V' <= capacity.

    The inflow follows a seasonal curve with a spring peak around day 100,
    ``Q_in = base * (1 + a * cos(2 pi (doy - 100) / 365)) * exp(s Z - s^2 / 2)``
    with ``Z`` a standard normal draw, so that the noise has mean one. Rain
    falls on a day with probability ``rain_probability`` and its depth is
    exponential with mean ``rain_mean_mm``. The baseline reservoir sees the same
    weather and no withdrawal. Evaporation, rain on the lake surface and the
    fact that water cannot be withdrawn from an empty reservoir are ignored.

    Parameters
    ----------
    full_stage_m, max_depth_m : float
        See :class:`LakeModel`.
    lake_area_m2 : float
        Surface of the reservoir (square metres); with ``max_depth_m`` it gives
        the capacity.
    base_inflow_m3s : float
        Annual mean inflow (cubic metres per second). Default 10.
    seasonal_amplitude : float
        Relative amplitude ``a`` of the seasonal inflow, in ``[0, 1)``.
        Default 0.8.
    inflow_log_sigma : float
        Log-scale spread ``s`` of the daily inflow noise. Default 0.3.
    release_fraction : float
        Share of the inflow released downstream every day, in ``[0, 1]``.
        Default 0.6.
    rain_probability : float
        Probability that a day has rain. Default 0.3.
    rain_mean_mm : float
        Mean rain depth of a rainy day (millimetres). Default 8.
    initial_level_range : tuple[float, float]
        Interval from which the initial filled fraction is drawn uniformly.
        Default ``(0.75, 1.0)``.

    When to use: for the tests, the smoke run and any experiment that has no
    Raven installation; the numbers are illustrative, not calibrated.

    Examples
    --------
    >>> import numpy as np
    >>> lake = SurrogateLake(
    ...     full_stage_m=420.0,
    ...     max_depth_m=10.0,
    ...     lake_area_m2=1e6,
    ...     inflow_log_sigma=0.0,
    ...     seasonal_amplitude=0.0,
    ...     rain_probability=0.0,
    ...     initial_level_range=(0.5, 0.5),
    ... )
    >>> from datetime import datetime
    >>> rng = np.random.default_rng(0)
    >>> start = lake.start(datetime(1980, 5, 10), rng)
    >>> start.stage_m
    415.0
    >>> day = lake.advance(datetime(1980, 5, 11), withdrawal_m3s=2.0, rng=rng)
    >>> round(day.stage_m, 4), round(day.baseline_stage_m, 4)
    (415.1728, 415.3456)
    """

    def __init__(
        self,
        *,
        full_stage_m: float,
        max_depth_m: float,
        lake_area_m2: float,
        base_inflow_m3s: float = 10.0,
        seasonal_amplitude: float = 0.8,
        inflow_log_sigma: float = 0.3,
        release_fraction: float = 0.6,
        rain_probability: float = 0.3,
        rain_mean_mm: float = 8.0,
        initial_level_range: tuple[float, float] = (0.75, 1.0),
    ) -> None:
        super().__init__(full_stage_m=full_stage_m, max_depth_m=max_depth_m)
        self.lake_area_m2 = lake_area_m2
        self.capacity_m3 = lake_area_m2 * max_depth_m
        self.base_inflow_m3s = base_inflow_m3s
        self.seasonal_amplitude = seasonal_amplitude
        self.inflow_log_sigma = inflow_log_sigma
        self.release_fraction = release_fraction
        self.rain_probability = rain_probability
        self.rain_mean_mm = rain_mean_mm
        self.initial_level_range = initial_level_range
        self._volume_m3 = 0.0
        self._baseline_volume_m3 = 0.0

    def _stage(self, volume_m3: float) -> float:
        fraction = volume_m3 / max(self.capacity_m3, EPS)
        return self.full_stage_m - self.max_depth_m + fraction * self.max_depth_m

    def _weather(self, date: datetime, rng: np.random.Generator) -> tuple[float, float]:
        """Draw the inflow (m3/s) and the rain (mm/day) of ``date``."""
        day_of_year = date.timetuple().tm_yday
        seasonal = 1.0 + self.seasonal_amplitude * math.cos(
            2.0 * math.pi * (day_of_year - 100) / 365.0
        )
        sigma = self.inflow_log_sigma
        noise = math.exp(sigma * float(rng.normal()) - 0.5 * sigma**2)
        inflow = self.base_inflow_m3s * seasonal * noise
        rains = float(rng.random()) < self.rain_probability
        precip = float(rng.exponential(self.rain_mean_mm)) if rains else 0.0
        return inflow, precip

    def _balance(
        self, volume_m3: float, inflow_m3s: float, withdrawal_m3s: float
    ) -> tuple[float, float]:
        """Return the next volume and the release of one day."""
        before_release = volume_m3 + (inflow_m3s - withdrawal_m3s) * SECONDS_PER_DAY
        outflow = self.release_fraction * inflow_m3s
        volume = before_release - outflow * SECONDS_PER_DAY

        if volume > self.capacity_m3:
            outflow += (volume - self.capacity_m3) / SECONDS_PER_DAY
            volume = self.capacity_m3
        elif volume < 0.0:
            outflow = max(0.0, outflow + volume / SECONDS_PER_DAY)
            volume = max(0.0, before_release - outflow * SECONDS_PER_DAY)

        return volume, outflow

    def start(
        self,
        date: datetime,
        rng: np.random.Generator,
        end_date: Optional[datetime] = None,
    ) -> LakeReading:
        """Draw an initial level and the weather of ``date``.

        Parameters
        ----------
        date : datetime
            First day of the episode.
        rng : numpy.random.Generator
            Generator of the environment.
        end_date : datetime, optional
            Ignored: the baseline is advanced day by day with the main
            reservoir.

        Returns
        -------
        LakeReading
            Initial state. The release of the first day is
            ``release_fraction * inflow``, and the baseline equals the main
            reservoir.
        """
        low, high = self.initial_level_range
        level = float(rng.uniform(low, high))
        self._volume_m3 = level * self.capacity_m3
        self._baseline_volume_m3 = self._volume_m3
        inflow, precip = self._weather(date, rng)
        outflow = self.release_fraction * inflow
        stage = self._stage(self._volume_m3)
        return LakeReading(
            stage_m=stage,
            inflow_m3s=inflow,
            outflow_m3s=outflow,
            precip_mm_day=precip,
            baseline_stage_m=stage,
            baseline_inflow_m3s=inflow,
            baseline_outflow_m3s=outflow,
        )

    def advance(
        self, date: datetime, withdrawal_m3s: float, rng: np.random.Generator
    ) -> LakeReading:
        """Apply one day of weather and withdrawal to both reservoirs.

        Parameters
        ----------
        date : datetime
            The day that is reached.
        withdrawal_m3s : float
            Total irrigation withdrawal (cubic metres per second).
        rng : numpy.random.Generator
            Generator of the environment.

        Returns
        -------
        LakeReading
            State on ``date``; the inflow and the rain are common to the main
            and the baseline reservoir.
        """
        inflow, precip = self._weather(date, rng)
        self._volume_m3, outflow = self._balance(
            self._volume_m3, inflow, withdrawal_m3s
        )
        self._baseline_volume_m3, baseline_outflow = self._balance(
            self._baseline_volume_m3, inflow, 0.0
        )
        return LakeReading(
            stage_m=self._stage(self._volume_m3),
            inflow_m3s=inflow,
            outflow_m3s=outflow,
            precip_mm_day=precip,
            baseline_stage_m=self._stage(self._baseline_volume_m3),
            baseline_inflow_m3s=inflow,
            baseline_outflow_m3s=baseline_outflow,
        )


class RavenOutputError(RuntimeError):
    """Raised when a Raven output file or column is missing or empty.

    When to use: catch it to tell a broken Raven installation from a bug of the
    example; the message names the file and the column.

    Examples
    --------
    >>> try:
    ...     raise RavenOutputError("no output")
    ... except RuntimeError as error:
    ...     str(error)
    'no output'
    """


GAUGE_COLUMNS = {
    "gauge_02ga041_m3s": "02GA041 [m3/s]",
    "gauge_02ga041_observed_m3s": "02GA041 (observed) [m3/s]",
    "gauge_02ga014_m3s": "02GA014 [m3/s]",
    "gauge_02ga014_observed_m3s": "02GA014 (observed) [m3/s]",
    "gauge_west_montrose_m3s": "West_Montrose [m3/s]",
    "gauge_west_montrose_observed_m3s": "West_Montrose (observed) [m3/s]",
}
"""Raven hydrograph columns of the river gauges, by metric-schema field name."""

STAGE_FILE = "ohms_canshield_ReservoirStages.csv"
HYDROGRAPH_FILE = "ohms_canshield_Hydrographs.csv"
OUTPUT_DIR = "3_Model_output"
RUN_NAME = "2_Raven/ohms_canshield"


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Return the rows of a Raven output file, with stripped column names."""
    if not csv_path.exists():
        raise RavenOutputError(f"Raven output not found: {csv_path}")

    with csv_path.open(newline="") as handle:
        rows = [
            {k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)
        ]

    if not rows:
        raise RavenOutputError(f"Raven output is empty: {csv_path}")

    return rows


def _column(row: dict[str, str], name: str, *, prefix: bool) -> Optional[float]:
    """Read ``name`` from a row; ``None`` when the column or the value is absent."""
    if name not in row:
        if prefix:
            matches = [key for key in row if key.startswith(name)]
        else:
            matches = [key for key in row if key.strip() == name.strip()]
        if not matches:
            return None
        name = matches[0]

    raw = row[name]
    return None if raw in ("", "---", None) else float(raw)


class _RavenRun:
    """One prepared copy of the Raven model and the files it writes.

    The output files are parsed once after each run and kept until the next
    run. Raven writes one row per day, so the row of a day is found by counting
    back from the last row, which is the end date of the run.
    """

    def __init__(self, *, model_dir: Path, command: str, run_root: Path) -> None:
        self.model_dir = model_dir
        self.command = command
        self.run_root = run_root
        self._rows: dict[str, list[dict[str, str]]] = {}

    def prepare(self) -> None:
        """Copy the model directory to a clean run directory."""
        if self.run_root.exists():
            shutil.rmtree(self.run_root)
        self.run_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self.model_dir, self.run_root, ignore=shutil.ignore_patterns(".cache")
        )
        logger.info("Prepared Raven run at %s", self.run_root)

    def run(self, date: datetime, withdrawals: list[tuple[datetime, float]]) -> None:
        """Write the withdrawal series, set the end date and run Raven."""
        self._rows = {}
        self._write_extraction(date, withdrawals)
        self._patch_end_date(date)
        (self.run_root / OUTPUT_DIR).mkdir(exist_ok=True)
        subprocess.run(
            [self.command, RUN_NAME, "-o", OUTPUT_DIR], cwd=self.run_root, check=False
        )

    def _write_extraction(
        self, date: datetime, withdrawals: list[tuple[datetime, float]]
    ) -> None:
        """Rewrite ``input/Extraction.rvt`` with one value per day to ``date``.

        Raven reads withdrawals as negative flows (cubic metres per second).
        """
        path = self.run_root / "input" / "Extraction.rvt"
        lines = path.read_text(encoding="utf-8").splitlines()
        header = next(
            i
            for i, line in enumerate(lines)
            if line.strip().startswith("1980-01-01 00:00:00")
        )
        end = next(
            i
            for i in range(header + 1, len(lines))
            if lines[i].strip() in (":EndObservationData", ":EndData")
        )
        n_days = (date.date() - RAVEN_ORIGIN.date()).days + 1
        values = ["\t0.0"] * n_days

        for usage_date, usage_m3s in withdrawals:
            index = (usage_date.date() - RAVEN_ORIGIN.date()).days
            if 0 <= index < n_days and abs(usage_m3s) >= 1e-12:
                values[index] = f"\t{-float(usage_m3s):.10f}"

        lines[header] = f"\t1980-01-01 00:00:00\t1\t{n_days}"
        new_lines = lines[: header + 1] + values + [lines[end].strip()]
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def _patch_end_date(self, date: datetime) -> None:
        path = self.run_root / "2_Raven" / "ohms_canshield.rvi"
        end_date = date.strftime("%Y-%m-%d 00:00:00")
        patched = [
            f":EndDate         {end_date}"
            if line.strip().startswith(":EndDate")
            else line
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        path.write_text("\n".join(patched) + "\n", encoding="utf-8")

    def _row(self, file_name: str, days_before_last: int) -> dict[str, str]:
        """Return the row ``days_before_last`` days before the last row."""
        if file_name not in self._rows:
            self._rows[file_name] = _read_rows(self.run_root / OUTPUT_DIR / file_name)
        rows = self._rows[file_name]
        if not 0 <= days_before_last < len(rows):
            raise RavenOutputError(
                f"{file_name} has {len(rows)} rows, so it has no row "
                + f"{days_before_last} days before the last."
            )
        return rows[-1 - days_before_last]

    def stage(self, column: str, *, days_before_last: int = 0) -> float:
        """Stage of the reservoir file; raises when it is missing.

        ``days_before_last`` counts back from the last row (0 is the end date
        of the run).
        """
        row = self._row(STAGE_FILE, days_before_last)
        value = _column(row, column, prefix=True)
        if value is None:
            raise RavenOutputError(f"No stage in column {column!r} of {STAGE_FILE}.")
        return value

    def hydrograph(
        self, column: str, *, required: bool = True, days_before_last: int = 0
    ) -> Optional[float]:
        """Value of a hydrograph column; ``None`` when optional and absent.

        ``days_before_last`` counts back from the last row (0 is the end date
        of the run).
        """
        row = self._row(HYDROGRAPH_FILE, days_before_last)
        value = _column(row, column, prefix=False)
        if value is None and required:
            raise RavenOutputError(
                f"No value in column {column!r} of {HYDROGRAPH_FILE}."
            )
        return value


class RavenLake(LakeModel):
    """Lake driven by the external Raven hydrological model.

    The model directory is copied twice under ``work_dir``: one run receives the
    irrigation withdrawals, the other receives none and gives the baseline. The
    main run is repeated every day from the start of its simulation, with the
    whole withdrawal series written to ``input/Extraction.rvt`` and the end date
    set to the current day; its state is the last row of its output files. The
    baseline input never changes, so the baseline run is made once, in
    :meth:`start`, up to the last day of the episode, and every later day reads
    its own row of that run (the row ``end_date - date`` days before the last).
    This relies on Raven writing one row per day and on a run to a later end
    date giving the same values on the earlier days, as a causal model does.
    The columns read are those of the Belwood Lake model.

    The reuse of the baseline run is verified only against the stand-in
    executable of :func:`write_stand_in_raven`, because the Raven model of the
    Belwood Lake catchment is not part of this repository; it has not been
    checked against the real model.

    Parameters
    ----------
    full_stage_m, max_depth_m : float
        See :class:`LakeModel`.
    raven_cwd : str or Path
        Model directory, holding ``2_Raven/ohms_canshield.rvi`` and
        ``input/Extraction.rvt``. It is only read.
    raven_cmd : str or Path
        Raven executable, run from the copied directory.
    key : str
        Name of the run directories, for example ``"m_0_seed_0"``; runs with
        the same key share their directories.
    work_dir : str or Path or None
        Where the run directories are created. ``None`` uses
        ``<raven_cwd>/.cache/prepared_runs``.
    stage_col, inflow_col, outflow_col, precip_col : str
        Output columns of the stage, of the reservoir inflow, of the release
        and of the rain.

    Raises
    ------
    RavenOutputError
        From :meth:`start` and :meth:`advance` when an output file or a
        required column is missing, or when a day outside the baseline run is
        read.
    ValueError
        From :meth:`start` without an ``end_date``, and from :meth:`advance`
        for a day after it.

    When to use: for a run on the real Belwood Lake model, with the ``raven/``
    directory and the executable available. Use :class:`SurrogateLake` when they
    are not.

    Examples
    --------
    A stand-in executable plays the role of Raven in this example:

    >>> import tempfile
    >>> from datetime import datetime
    >>> import numpy as np
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     model, command = write_stand_in_raven(Path(tmp))
    ...     lake = RavenLake(
    ...         full_stage_m=420.0,
    ...         max_depth_m=10.0,
    ...         raven_cwd=model,
    ...         raven_cmd=command,
    ...         key="doctest",
    ...         work_dir=Path(tmp) / "runs",
    ...     )
    ...     rng = np.random.default_rng(0)
    ...     start = lake.start(datetime(1980, 5, 1), rng, datetime(1980, 5, 2))
    ...     day = lake.advance(datetime(1980, 5, 2), withdrawal_m3s=2.0, rng=rng)
    >>> start.stage_m, day.stage_m, day.baseline_stage_m
    (420.0, 418.0, 420.0)
    """

    def __init__(
        self,
        *,
        full_stage_m: float,
        max_depth_m: float,
        raven_cwd: str | Path,
        raven_cmd: str | Path,
        key: str,
        work_dir: str | Path | None = None,
        stage_col: str = "Belwood_Lake",
        inflow_col: str = "Belwood_Lake (res. inflow) [m3/s]",
        outflow_col: str = "Belwood_Lake [m3/s]",
        precip_col: str = "precip [mm/day]",
    ) -> None:
        super().__init__(full_stage_m=full_stage_m, max_depth_m=max_depth_m)
        model_dir = Path(raven_cwd).resolve()
        root = (
            Path(work_dir).resolve()
            if work_dir is not None
            else model_dir / ".cache" / "prepared_runs"
        )
        command = str(raven_cmd)
        self._main = _RavenRun(
            model_dir=model_dir, command=command, run_root=root / key
        )
        self._baseline = _RavenRun(
            model_dir=model_dir, command=command, run_root=root / f"baseline_{key}"
        )
        self.stage_col = stage_col
        self.inflow_col = inflow_col
        self.outflow_col = outflow_col
        self.precip_col = precip_col
        self._withdrawals: list[tuple[datetime, float]] = []
        self._end_date: Optional[datetime] = None

    def _days_to_end(self, date: datetime) -> int:
        """Return the days from ``date`` to the end of the baseline run."""
        if self._end_date is None:
            raise ValueError("RavenLake.start must be called before the lake is read.")
        if date.date() > self._end_date.date():
            raise ValueError(
                f"{date.date()} is after the end of the episode "
                + f"({self._end_date.date()}), where the baseline run stops."
            )
        return (self._end_date.date() - date.date()).days

    def _read(self, date: datetime) -> LakeReading:
        # The baseline run ends on the last day of the episode; the row of
        # ``date`` is ``ahead`` days before its last row.
        ahead = self._days_to_end(date)
        gauges = {}
        for name, column in GAUGE_COLUMNS.items():
            value = self._main.hydrograph(column, required=False)
            if value is not None:
                gauges[name] = value

        return LakeReading(
            stage_m=self._main.stage(self.stage_col),
            inflow_m3s=self._main.hydrograph(self.inflow_col),
            outflow_m3s=self._main.hydrograph(self.outflow_col),
            precip_mm_day=self._main.hydrograph(self.precip_col),
            baseline_stage_m=self._baseline.stage(
                self.stage_col, days_before_last=ahead
            ),
            baseline_inflow_m3s=self._baseline.hydrograph(
                self.inflow_col, days_before_last=ahead
            ),
            baseline_outflow_m3s=self._baseline.hydrograph(
                self.outflow_col, days_before_last=ahead
            ),
            gauges=gauges,
        )

    def start(
        self,
        date: datetime,
        rng: np.random.Generator,
        end_date: Optional[datetime] = None,
    ) -> LakeReading:
        """Prepare both run directories, run Raven to ``date`` and the baseline.

        The main model is run up to ``date``. The baseline model is run once,
        up to ``end_date``, and answers every day of the episode.

        Parameters
        ----------
        date : datetime
            First day of the episode.
        rng : numpy.random.Generator
            Unused: the weather comes from the Raven forcing files.
        end_date : datetime
            Last day the episode can reach, at or after ``date``; it is
            required.

        Returns
        -------
        LakeReading
            Last row of the main outputs and the row of ``date`` of the
            baseline, with no withdrawal so far.

        Raises
        ------
        ValueError
            If ``end_date`` is missing or before ``date``.
        """
        if end_date is None or end_date.date() < date.date():
            raise ValueError(
                "RavenLake.start needs an end_date at or after the first day, "
                + f"got {end_date!r} for {date.date()}."
            )
        self._withdrawals = []
        self._end_date = end_date
        self._main.prepare()
        self._baseline.prepare()
        self._main.run(date, self._withdrawals)
        self._baseline.run(end_date, [])
        return self._read(date)

    def advance(
        self, date: datetime, withdrawal_m3s: float, rng: np.random.Generator
    ) -> LakeReading:
        """Record the withdrawal at ``date``, run the main model up to it.

        The baseline is not run again: its row of ``date`` is read from the run
        made by :meth:`start`.

        Parameters
        ----------
        date : datetime
            The day that is reached; the withdrawal is written at this day,
            as the first version of the example did. It must not be after the
            ``end_date`` given to :meth:`start`.
        withdrawal_m3s : float
            Total irrigation withdrawal (cubic metres per second).
        rng : numpy.random.Generator
            Unused.

        Returns
        -------
        LakeReading
            Last row of the main outputs and the row of ``date`` of the
            baseline.

        Raises
        ------
        ValueError
            If ``date`` is after the end of the episode.
        """
        self._days_to_end(date)
        self._withdrawals.append((date, withdrawal_m3s))
        self._main.run(date, self._withdrawals)
        return self._read(date)


_STAND_IN_SCRIPT = """\
#!{python}
import sys
from datetime import date, timedelta
from pathlib import Path

root = Path.cwd()
out = root / sys.argv[sys.argv.index("-o") + 1]
out.mkdir(exist_ok=True)
# One line per execution, in a file beside the executable: the run directory
# is wiped at every episode, so it cannot keep the count.
with (Path(__file__).resolve().parent / "calls.log").open("a") as log:
    log.write(root.name + "\\n")

rvi = (root / (sys.argv[1] + ".rvi")).read_text().splitlines()


def read_date(key):
    line = next(l for l in rvi if l.strip().startswith(key))
    return date.fromisoformat(line.split()[1])


first = read_date(":StartDate")
last = read_date(":EndDate")
lines = (root / "input" / "Extraction.rvt").read_text().splitlines()
header = next(i for i, l in enumerate(lines) if l.strip().startswith("1980-01-01"))
ends = [i for i, l in enumerate(lines) if l.strip().startswith(":End")]
values = [float(v) for v in lines[header + 1:ends[0]]]

stages = ["time, Belwood_Lake [m]"]
hydrographs = [
    "time, Belwood_Lake (res. inflow) [m3/s], Belwood_Lake [m3/s], precip [mm/day], "
    "West_Montrose [m3/s]"
]
withdrawn = 0.0
for index in range((last - first).days + 1):
    day = first + timedelta(days=index)
    if index < len(values):
        withdrawn -= values[index]
    stages.append("%s, %r" % (day, 420.0 - withdrawn))
    hydrographs.append(
        "%s, %r, %r, {rain!r}, 9.0" % (day, 5.0 + 0.01 * index, 3.0 + 0.01 * index)
    )
(out / "ohms_canshield_ReservoirStages.csv").write_text("\\n".join(stages) + "\\n")
(out / "ohms_canshield_Hydrographs.csv").write_text("\\n".join(hydrographs) + "\\n")
"""


def write_stand_in_raven(directory: Path, rain_mm_day: float = 2.0) -> tuple[Path, str]:
    """Write a tiny model directory and an executable that mimics Raven.

    The stand-in reads the withdrawal series from ``input/Extraction.rvt`` and
    the end date from the ``.rvi`` file, and writes the two output files that
    :class:`RavenLake` reads, with one row per day from 1980-01-01 to the end
    date, as Raven does. The stage of day ``k`` (counted from 1980-01-01) is
    ``420`` minus the sum of the withdrawals up to that day (cubic metres per
    second); the hydrograph has an inflow of ``5 + 0.01 k``, a release of
    ``3 + 0.01 k``, a rain of ``rain_mm_day`` and a ``West_Montrose`` gauge of
    9. The rows that change from one day to the next let a test tell which day a
    reading comes from. Every execution appends the name of its run directory to
    ``calls.log``, beside the executable, so that a test can count the runs.

    Parameters
    ----------
    directory : pathlib.Path
        Existing directory to write into.
    rain_mm_day : float, optional
        Daily rain the stand-in reports, in mm/day (default 2.0). A rain above
        the crop's daily use leaves nothing to irrigate, so a test that needs a
        withdrawal passes 0.

    Returns
    -------
    tuple[pathlib.Path, str]
        The model directory and the path of the executable.

    When to use: in tests and doctests of the Raven plumbing, which cannot rely
    on the real model.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     model, command = write_stand_in_raven(Path(tmp))
    ...     sorted(path.name for path in model.iterdir())
    ['2_Raven', 'input']
    """
    model = directory / "raven_model"
    (model / "2_Raven").mkdir(parents=True)
    (model / "input").mkdir()
    (model / "2_Raven" / "ohms_canshield.rvi").write_text(
        ":StartDate       1980-01-01 00:00:00\n:EndDate         1980-01-02 00:00:00\n"
    )
    (model / "input" / "Extraction.rvt").write_text(
        ":ObservationData HYDROGRAPH 1 m3/s\n"
        + "\t1980-01-01 00:00:00\t1\t1\n\t0.0\n:EndObservationData\n"
    )
    command = directory / "stand_in_raven"
    command.write_text(
        _STAND_IN_SCRIPT.format(python=sys.executable, rain=float(rain_mm_day))
    )
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    return model, str(command)
