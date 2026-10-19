"""Preliminary Geonics EM31-MK2 R31 inspection. Requires numpy, pandas, matplotlib."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

MEAS = re.compile(r'^T.([+-]\d{4})([+-]\d{4})\s+(\d+)')

def _degrees(value, hemisphere):
    if not value:
        return np.nan
    x = float(value)
    deg = int(x // 100)
    decimal = deg + (x - 100 * deg) / 60
    return -decimal if hemisphere in ('S', 'W') else decimal

def read_em31(filename):
    """Return measurements and GPS fixes, preserving acquisition segment IDs.

    channel_1 and channel_2 are UNSCALED instrument fields. Verify channel
    identity and scaling against logger documentation before interpretation.
    """
    lines = Path(filename).read_bytes().decode('latin-1').splitlines()
    measurements, fixes = [], []
    segment = -1
    pending_gps = None
    for line in lines:
        if line.startswith('Z'):
            segment += 1
        m = MEAS.match(line)
        if m:
            measurements.append(dict(segment=max(segment, 0),
                                     channel_1=int(m.group(1)),
                                     channel_2=int(m.group(2)),
                                     counter=int(m.group(3))))
        if line.startswith('@$GPGGA'):
            pending_gps = line[1:].strip()
        elif pending_gps is not None and line.startswith('#'):
            pending_gps += line[1:].strip()
        elif pending_gps is not None and line.startswith('!'):
            fields = pending_gps.split(',')
            try:
                fixes.append(dict(segment=max(segment, 0),
                    utc=fields[1],
                    latitude=_degrees(fields[2], fields[3]),
                    longitude=_degrees(fields[4], fields[5]),
                    gps_quality=int(fields[6]),
                    satellites=int(fields[7]),
                    altitude_m=float(fields[9]),
                    counter=int(line[1:].strip())))
            except (ValueError, IndexError):
                pass
            pending_gps = None
    return pd.DataFrame(measurements), pd.DataFrame(fixes)

def plot_em31(filename, save=False):
    """Plot both raw channels and GPS position where available."""
    data, gps = read_em31(filename)
    if data.empty:
        print(f'{Path(filename).name}: no measurement records')
        return data, gps

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                            constrained_layout=True)
    for seg, group in data.groupby('segment'):
        for j, channel in enumerate(('channel_1', 'channel_2')):
            axs[j].plot(group.index, group[channel], '.', ms=3,
                        label=f'Segment {seg + 1}')
            axs[j].set_ylabel(f'{channel} (stored counts)')
            axs[j].grid(alpha=.25)
    axs[0].legend()
    axs[-1].set_xlabel('Measurement index')
    fig.suptitle(Path(filename).name + ' — unscaled EM31 readings')
    if save:
        fig.savefig(Path(filename).with_suffix('.channels.png'), dpi=180)
    plt.show()

    if not gps.empty:
        # Plot native GPS fixes only; do not imply a measurement-to-GPS match.
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sc = ax.scatter(gps.longitude, gps.latitude,
                        c=np.arange(len(gps)), cmap='viridis', s=16)
        fig.colorbar(sc, ax=ax, label='GPS fix sequence')
        ax.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)',
               title=Path(filename).name + ' — GPS track')
        ax.grid(alpha=.25)
        if save:
            fig.savefig(Path(filename).with_suffix('.gps.png'), dpi=180)
        plt.show()
    print(f'{Path(filename).name}: {len(data)} measurements, '
          f'{len(gps)} GPS fixes, {data.segment.nunique()} segments')
    print(data[['channel_1', 'channel_2']].describe().round(2))
    return data, gps

if __name__ == '__main__':
    for file in ('molly2onyx.R31', 'mollyonyx.R31',
                 'onyxsmt.R31', 'onyxtms.R31'):
        plot_em31(file, save=True)


def plot_em31_northing(data, gps, channel="channel_1"):

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32613",
        always_xy=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for seg in sorted(data["segment"].unique()):

        em = data[data["segment"] == seg].sort_values("counter")
        pos = gps[gps["segment"] == seg].sort_values("counter")

        if len(pos) < 2:
            continue

        # Convert GPS positions into meters
        east, north = transformer.transform(
            pos["longitude"].to_numpy(),
            pos["latitude"].to_numpy()
        )

        # Interpolate position to EM timestamps
        em_north = np.interp(
            em["counter"].to_numpy(),
            pos["counter"].to_numpy(),
            north,
            left=np.nan,
            right=np.nan
        )

        valid = np.isfinite(em_north)

        ax.scatter(
            em_north[valid],
            em[channel].to_numpy()[valid],
            s=8,
            label=f"Line {seg + 1}"
        )

    ax.set_xlabel("UTM Northing (m)")
    ax.set_ylabel(f"{channel} (stored counts)")
    ax.set_title("EM-31 Response vs. Northing")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

def plot_em31_map(data, gps, channel="channel_1"):

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32613",
        always_xy=True
    )

    fig, ax = plt.subplots(figsize=(9, 9))

    all_points = []

    for seg in sorted(data["segment"].unique()):

        em = data[data["segment"] == seg].sort_values("counter")
        pos = gps[gps["segment"] == seg].sort_values("counter")

        if len(pos) < 2:
            continue

        east, north = transformer.transform(
            pos["longitude"].to_numpy(),
            pos["latitude"].to_numpy()
        )

        em_east = np.interp(
            em["counter"],
            pos["counter"],
            east,
            left=np.nan,
            right=np.nan
        )

        em_north = np.interp(
            em["counter"],
            pos["counter"],
            north,
            left=np.nan,
            right=np.nan
        )

        valid = np.isfinite(em_east) & np.isfinite(em_north)

        sc = ax.scatter(
            em_east[valid],
            em_north[valid],
            c=em[channel].to_numpy()[valid],
            cmap="viridis",
            s=12,
            vmin=data[channel].min(),
            vmax=data[channel].max()
        )

    ax.set_aspect("equal")
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.set_title(f"EM-31 Spatial Response — {channel}")

    plt.colorbar(
        sc,
        ax=ax,
        label=f"{channel} (stored counts)"
    )

    plt.show()