"""
Learn the mapping between vgmaps map names and the game's (map_bank, map_number).

This is the self-verifying way to "know for sure" which ROM id a map is: while
you are standing on a map, the emulator's GAME_STATE reports the exact
(map_bank, map_number) from RAM, and locationTracker independently identifies the
map by template-matching the screenshot. When both agree with high confidence we
record name -> (bank, number).

Run it once to confirm where you currently are, or with --watch to passively
build the whole table as you play. The result lands in
connectionData/mapIds.json and is used by encounterExtractor to resolve
encounters by map name.

    python mapIdMapper.py             # learn the current map once
    python mapIdMapper.py --watch     # keep learning every few seconds
    python mapIdMapper.py --show      # print what's been learned so far
    python mapIdMapper.py --set Name  # record the current id as Name yourself
    python mapIdMapper.py --offset Name <imageCol> <imageRow>
                                      # measure a cropped rip's tile offset

Interiors need --set. They can't be learned automatically: a room smaller than
the 240x160 screen doesn't scroll, so the screenshot is never a sub-image of the
map and template matching can't identify it. Stand in the room and name it.

Format (a name can map to several ids - shared interiors like Pokemon Centers):
    { "Route01": [[3, 19]], "Pokemon_Center_inside_FRLG": [[4, 3], [5, 7], ...] }
"""

import json
import os
import socket
import sys
import time

from locationTracker import LocationTracker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mGBA'))
import mgba_client  # noqa: E402

MAP_IDS_PATH = os.path.join(os.path.dirname(__file__), 'connectionData', 'mapIds.json')
MAP_OFFSETS_PATH = os.path.join(os.path.dirname(__file__), 'connectionData',
                                'mapOffsets.json')


def loadMapIds():
    if os.path.exists(MAP_IDS_PATH):
        with open(MAP_IDS_PATH, 'r') as f:
            return json.load(f)
    return {}


def saveMapIds(data):
    os.makedirs(os.path.dirname(MAP_IDS_PATH), exist_ok=True)
    with open(MAP_IDS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


class MapIdMapper:
    def __init__(self, host='127.0.0.1', port=54321, scratchDir=None):
        self.tracker = LocationTracker()
        self.scratchDir = scratchDir or os.path.dirname(__file__)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.mapIds = loadMapIds()

    def _gameState(self):
        header, _ = mgba_client.send_command(self.sock, "GAME_STATE")
        if header.startswith("ERR"):
            return None
        try:
            return json.loads(header.split("|", 1)[1])
        except (IndexError, json.JSONDecodeError):
            return None

    def learnOnce(self, minConfidence=None):
        """Observe once. Returns (name, bank, number, confidence, isNew) or None."""
        gs = self._gameState()
        if not gs:
            print("No GAME_STATE (game starting, or not FR/LG).")
            return None
        player = gs.get('player', {})
        bank, number = player.get('map_bank'), player.get('map_number')

        shotPath = os.path.join(self.scratchDir, "mapid_screenshot.png")
        mgba_client.screenshot(self.sock, shotPath)
        fix = self.tracker.locatePlayer(shotPath, gameState=gs)
        if fix is None:
            print(f"GAME_STATE says ({bank},{number}) but no map matched the "
                  f"screenshot (battle/dialog?).")
            return None

        threshold = minConfidence if minConfidence is not None \
            else self.tracker.CONFIDENCE_THRESHOLD
        if fix['confidence'] < threshold:
            print(f"Low confidence {fix['confidence']:.3f} for {fix['mapName']} "
                  f"at ({bank},{number}) - not recording. Move and try again.")
            return None

        name = fix['mapName']
        pair = [bank, number]

        # Never let one id be claimed by two maps. A single mislocated fix
        # recorded here is permanent and poisons every later lookup, which is
        # exactly how a bogus entry can make the tracker teleport you across the
        # world. Refuse the write and let the operator sort it out.
        owner = next((other for other, pairs in self.mapIds.items()
                      if other != name and pair in pairs), None)
        if owner is not None:
            print(f"REFUSING to record {name} <-> ({bank},{number}): already "
                  f"claimed by {owner!r}. One of the two is a bad fix - remove "
                  f"the wrong entry from mapIds.json before continuing.")
            return None

        ids = self.mapIds.setdefault(name, [])
        isNew = pair not in ids
        if isNew:
            ids.append(pair)
            saveMapIds(self.mapIds)
        print(f"{name}  <->  (bank={bank}, number={number})  "
              f"conf={fix['confidence']:.3f}  {'[recorded]' if isNew else '[known]'}")
        return (name, bank, number, fix['confidence'], isNew)

    def setCurrent(self, mapName):
        """Record the current (bank, number) as `mapName` on the operator's word.

        learnOnce can't help with interiors: it only records what the tracker
        already recognises, and a room smaller than the 240x160 screen is not
        template-matchable at all, so it never produces a fix to learn from.
        That's circular for exactly the maps that most need an entry - and until
        one exists, the tracker has nothing to identify the room by. This breaks
        the loop by letting you assert "I am standing in here" yourself.
        """
        gs = self._gameState()
        if not gs:
            print("No GAME_STATE (game starting, or not FR/LG).")
            return None
        player = gs.get('player', {})
        bank, number = player.get('map_bank'), player.get('map_number')

        if mapName not in self.tracker.maps:
            print(f"Unknown map {mapName!r} - no image in maps/. Closest names:")
            lowered = mapName.lower()
            near = [m for m in self.tracker.maps if lowered in m.lower()]
            for m in sorted(near)[:10]:
                print(f"    {m}")
            if not near:
                print("    (no name contains that substring)")
            return None

        pair = [bank, number]
        owner = next((other for other, pairs in self.mapIds.items()
                      if other != mapName and pair in pairs), None)
        if owner is not None:
            print(f"REFUSING: ({bank},{number}) is already recorded as {owner!r}. "
                  f"Remove that entry from mapIds.json first if it's wrong.")
            return None

        ids = self.mapIds.setdefault(mapName, [])
        if pair in ids:
            print(f"{mapName} <-> ({bank},{number}) already recorded.")
            return (mapName, bank, number, None, False)
        ids.append(pair)
        saveMapIds(self.mapIds)
        print(f"{mapName}  <->  (bank={bank}, number={number})  [recorded]")
        return (mapName, bank, number, None, True)

    def setOffset(self, mapName, imageCol, imageRow):
        """Measure this map's RAM->image correction from where you're standing.

        Stand on a tile you can point to in the map image, pass that tile's
        image coordinates, and the difference against RAM is the offset. Needed
        because an interior is too small to template-match, so nothing else can
        detect that its rip was cropped tighter than the real map.
        """
        gs = self._gameState()
        if not gs:
            print("No GAME_STATE (game starting, or not FR/LG).")
            return None
        player = gs.get('player', {})
        x, y = player.get('x'), player.get('y')
        if x is None or y is None:
            print("GAME_STATE has no player coordinates.")
            return None
        if mapName not in self.tracker.maps:
            print(f"Unknown map {mapName!r} - no image in maps/.")
            return None

        offset = [imageCol - x, imageRow - y]
        data = {}
        if os.path.exists(MAP_OFFSETS_PATH):
            with open(MAP_OFFSETS_PATH, 'r') as f:
                data = json.load(f)
        previous = data.get(mapName)
        data[mapName] = offset
        with open(MAP_OFFSETS_PATH, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"{mapName}: RAM ({x},{y}) is image tile ({imageCol},{imageRow})")
        print(f"  offset {offset}  (imageTile = ram + offset)"
              + (f"  (was {previous})" if previous else "  [new]"))
        default = list(self.tracker.BORDER_OFFSET)
        if offset == default:
            print(f"  Matches the standard border offset {default} - this entry "
                  f"is redundant but harmless.")
        return offset

    def watch(self, interval=3.0):
        print("Watching - walk around to learn maps. Ctrl+C to stop.")
        try:
            while True:
                self.learnOnce()
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\nStopped. {len(self.mapIds)} maps known in {MAP_IDS_PATH}")


def main():
    if '--show' in sys.argv:
        ids = loadMapIds()
        for name in sorted(ids):
            print(f"{name}: {ids[name]}")
        print(f"\n{len(ids)} maps mapped.")
        return
    if '--offset' in sys.argv:
        i = sys.argv.index('--offset')
        if i + 3 >= len(sys.argv):
            print("Usage: python mapIdMapper.py --offset <MapName> "
                  "<imageCol> <imageRow>")
            print("  Stand on a tile you can identify in the map image and "
                  "give that tile's image coordinates.")
            return
        MapIdMapper().setOffset(sys.argv[i + 1], int(sys.argv[i + 2]),
                                int(sys.argv[i + 3]))
        return
    if '--set' in sys.argv:
        i = sys.argv.index('--set')
        if i + 1 >= len(sys.argv):
            print("Usage: python mapIdMapper.py --set <MapName>")
            return
        MapIdMapper().setCurrent(sys.argv[i + 1])
        return
    mapper = MapIdMapper()
    if '--watch' in sys.argv:
        mapper.watch()
    else:
        mapper.learnOnce()


if __name__ == '__main__':
    main()
