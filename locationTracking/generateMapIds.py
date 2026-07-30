"""
Regenerate connectionData/mapIds.json from the map image filenames.

The decompiled-ROM rips in maps/ are named "<bank>-<number>-<Name>.png", so the
game's (map_bank, map_number) is already in the filename and nothing has to be
learned in game. This rebuilds the whole table from what's currently in maps/,
replacing the hand-learned table that mapIdMapper.py used to grow one map at a
time.

Map names are the filename stems (e.g. "3-0-PalletTown"), which is what
connections.json and tileData use as map keys.

Format written (one name may own several ids, though rips never do):
    { "3-0-PalletTown": [[3, 0]], "3-19-Route1": [[3, 19]] }

Usage:
    python generateMapIds.py               # write connectionData/mapIds.json
    python generateMapIds.py --dry-run     # print what would be written
    python generateMapIds.py --check       # exit 1 if the file is out of date
"""

import json
import os
import re
import sys

MAPS_DIR = os.path.join(os.path.dirname(__file__), 'maps')
MAP_IDS_PATH = os.path.join(os.path.dirname(__file__), 'connectionData', 'mapIds.json')
CONNECTIONS_PATH = os.path.join(os.path.dirname(__file__), 'connectionData',
                                'connections.json')

# "<bank>-<number>-<Name>" - the name may itself contain hyphens, so only the
# first two numeric segments are the id.
NAME_RE = re.compile(r'^(\d+)-(\d+)-(.+)$')


def parseMapName(fileName):
    """Return (bank, number, mapName) for a rip filename, or None if unnamed.

    None means the file isn't a "<bank>-<number>-<Name>.png" rip - either an old
    hand-cropped map that hasn't been transferred yet, or a stray file.
    """
    stem, ext = os.path.splitext(fileName)
    if ext.lower() != '.png':
        return None
    m = NAME_RE.match(stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), stem


def buildMapIds(mapsDir=None):
    """Return ({mapName: [[bank, number]]}, skipped, conflicts).

    skipped is the list of .png files whose names carry no id; conflicts is a
    list of (bank, number, [mapName, ...]) claimed by more than one map, which
    locationTracker would refuse to use.
    """
    mapsDir = mapsDir or MAPS_DIR
    claims = {}  # (bank, number) -> [mapName, ...]
    skipped = []
    for fileName in sorted(os.listdir(mapsDir)):
        if not fileName.lower().endswith('.png'):
            continue
        parsed = parseMapName(fileName)
        if parsed is None:
            skipped.append(fileName)
            continue
        bank, number, mapName = parsed
        claims.setdefault((bank, number), []).append(mapName)

    conflicts = [(bank, number, names)
                 for (bank, number), names in sorted(claims.items())
                 if len(names) > 1]

    # Sort by id so the file diffs readably as maps get transferred.
    mapIds = {}
    for (bank, number), names in sorted(claims.items()):
        for name in names:
            mapIds.setdefault(name, []).append([bank, number])
    return mapIds, skipped, conflicts


def main(argv):
    dryRun = '--dry-run' in argv
    check = '--check' in argv

    mapIds, skipped, conflicts = buildMapIds()

    print(f'{len(mapIds)} maps with ids in {MAPS_DIR}')
    if skipped:
        print(f'{len(skipped)} png(s) with no id in the name (not transferred yet?):')
        for fileName in skipped:
            print(f'  - {fileName}')
    for bank, number, names in conflicts:
        print(f'ERROR: id ({bank}, {number}) is claimed by {len(names)} maps '
              f'{names} - locationTracker will ignore that id. Fix the filenames.')

    # Maps that are drawn but not yet wired into connections.json can't be
    # navigated to; worth knowing, but not an error for this table.
    if os.path.exists(CONNECTIONS_PATH):
        with open(CONNECTIONS_PATH, 'r') as f:
            conns = json.load(f)
        connMaps = set(conns.get('maps', {}).keys())
        missing = sorted(connMaps - set(mapIds))
        unwired = sorted(set(mapIds) - connMaps)
        if missing:
            print(f'WARNING: {len(missing)} map(s) in connections.json have no '
                  f'matching image: {missing}')
        if unwired:
            print(f'{len(unwired)} map(s) have ids but no connections.json entry yet')

    if check:
        existing = None
        if os.path.exists(MAP_IDS_PATH):
            with open(MAP_IDS_PATH, 'r') as f:
                existing = json.load(f)
        if existing == mapIds:
            print('mapIds.json is up to date')
            return 0
        print('mapIds.json is out of date - rerun without --check')
        return 1

    if dryRun:
        print(json.dumps(mapIds, indent=2))
        return 1 if conflicts else 0

    os.makedirs(os.path.dirname(MAP_IDS_PATH), exist_ok=True)
    with open(MAP_IDS_PATH, 'w') as f:
        json.dump(mapIds, f, indent=2)
    print(f'Wrote {len(mapIds)} entries to {MAP_IDS_PATH}')
    return 1 if conflicts else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
