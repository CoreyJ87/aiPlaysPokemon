import cv2  # needed for template matching
import os  # needed for file path operations
import json  # needed to read connection / instance data
import numpy as np  # needed for image processing


class LocationTracker:
    """
    Tracks the player's location in Pokemon Leaf Green by template matching
    game screenshots against known map images using OpenCV.

    Template matching is the primary locator: because each map image *is* the
    coordinate grid, a match origin converts directly to a tile (pixel // 16)
    with no alignment step.  Two refinements make it usable at scale:

      * Neighbor-restricted search - only the current map and its connection
        neighbors are matched first; a full scan happens only when confidence
        is low.  This is far faster than scanning ~180 maps every call.

      * Instance disambiguation - shared interiors (Pokemon Center / Mart) all
        use one image, so template matching alone can't tell which city's PC you
        are in.  When given a GAME_STATE dict, map_bank/map_number resolves the
        instance via the registry in connections.json.
    """

    SCREEN_WIDTH = 240
    SCREEN_HEIGHT = 160
    PLAYER_OFFSET_X = SCREEN_WIDTH // 2
    PLAYER_OFFSET_Y = SCREEN_HEIGHT // 2
    TILE_SIZE = 16

    # If the best neighbor match is below this, fall back to a full scan.
    CONFIDENCE_THRESHOLD = 0.90

    # Below this, a "best" match is noise and gets rejected outright. Template
    # matching always returns *something* - the arg-max over ~180 maps is never
    # empty - so without a floor an unmatchable screen (a small interior, a
    # dialog, a fade) silently resolves to whichever map is biggest, because a
    # 1920x1920 island has a million more candidate offsets to find a spurious
    # correlation in than a 768x640 town does. Observed real matches sit at
    # 0.95-0.999; observed false positives at 0.77-0.85.
    MIN_CONFIDENCE = 0.90

    # A flat screen (fade to black during a warp, a full-screen dialog, a battle
    # transition) has no texture to match on, and normalized correlation scores a
    # uniform patch against a uniform region as a *perfect* 1.000. Big maps have
    # large blank areas - SeviiIslands-FiveIsland measures std 0.0 in places -
    # so a fading screen reliably "matches" one of them at full confidence. Real
    # overworld frames measure std 30-48, so this rejects flat frames outright.
    MIN_SCREEN_STDDEV = 10.0

    def __init__(self, mapsDirectory=None, connectionDataDir=None):
        if mapsDirectory is None:
            mapsDirectory = os.path.join(os.path.dirname(__file__), 'maps')
        if connectionDataDir is None:
            connectionDataDir = os.path.join(os.path.dirname(__file__), 'connectionData')

        self.maps = {}  # mapName -> cv2 image (BGR)
        self._loadMaps(mapsDirectory)

        # Connection graph + instance registry (used for fast search + disambig)
        self.neighbors = {}    # mapName -> set(neighborMapName)
        self.instances = {}    # instanceId -> {template, bank, number, ...}
        self._instanceByBankNum = {}  # (bank, number) -> instanceId
        self._loadConnections(connectionDataDir)

        # Authoritative (bank, number) -> mapName lookup from game RAM. This is
        # the primary way we resolve map identity; template matching is only
        # used for position within the resolved map (see locatePlayer).
        self._mapNameByBankNum = {}
        self._ambiguousIds = {}       # (bank, number) -> [conflicting names]
        self._undersizedWarned = set()
        self._loadMapIds(connectionDataDir)

        # mapName -> (dx, dy) correcting RAM coords onto the image grid.
        self.mapOffsets = {}
        self._loadMapOffsets(connectionDataDir)

        self.currentMap = None
        self.currentPosition = None  # (x, y) pixel coords of player on the map
        self.currentTile = None      # (col, row)
        self.currentConfidence = 0.0
        self.currentInstance = None
        self._lastMapName = None

    def _loadMaps(self, mapsDirectory):
        supportedExtensions = ('.png', '.jpg', '.jpeg', '.bmp')
        for filename in os.listdir(mapsDirectory):
            if filename.lower().endswith(supportedExtensions):
                filepath = os.path.join(mapsDirectory, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    self.maps[os.path.splitext(filename)[0]] = image
        print(f'LocationTracker: Loaded {len(self.maps)} maps.')

    def _loadConnections(self, connectionDataDir):
        connPath = os.path.join(connectionDataDir, 'connections.json')
        if not os.path.exists(connPath):
            return
        with open(connPath, 'r') as f:
            data = json.load(f)
        for mapName, mapData in data.get('maps', {}).items():
            ns = self.neighbors.setdefault(mapName, set())
            for conn in mapData.get('connections', []):
                toMap = conn.get('toMap', '')
                if toMap and toMap != '@return':
                    ns.add(toMap)
        self.instances = data.get('instances', {})
        for instId, rec in self.instances.items():
            if 'bank' in rec and 'number' in rec:
                self._instanceByBankNum[(rec['bank'], rec['number'])] = instId

    def _loadMapIds(self, connectionDataDir):
        """Build (bank, number) -> mapName from mapIds.json.

        mapIds.json stores mapName -> [[bank, number], ...]; several bank/number
        pairs may point at the same map (e.g. shared interiors), so this reverse
        lookup is many-to-one and unambiguous per (bank, number).
        """
        idsPath = os.path.join(connectionDataDir, 'mapIds.json')
        if not os.path.exists(idsPath):
            return
        with open(idsPath, 'r') as f:
            data = json.load(f)
        claims = {}  # (bank, number) -> [mapName, ...]
        for mapName, pairs in data.items():
            for pair in pairs:
                if len(pair) == 2:
                    claims.setdefault((pair[0], pair[1]), []).append(mapName)

        # A (bank, number) claimed by two different maps is bad data - one of
        # them was recorded from a mislocated fix. Silently letting the last
        # writer win makes the game world teleport, so drop the pair instead and
        # let template matching decide; it's wrong far more visibly.
        for bankNum, names in claims.items():
            if len(names) == 1:
                self._mapNameByBankNum[bankNum] = names[0]
            else:
                self._ambiguousIds[bankNum] = sorted(names)
                print(f'LocationTracker: WARNING map id {bankNum} is claimed by '
                      f'{len(names)} maps {sorted(names)} - ignoring it. Fix '
                      f'mapIds.json; one entry was learned from a bad fix.')

    def locatePlayer(self, screenshotPath, gameState=None):
        """
        Find which map the screenshot belongs to and where the player is on it.

        Args:
            screenshotPath: path to the current game screenshot.
            gameState: optional GAME_STATE dict (from mgba_client). When present,
                map_bank/map_number resolve the exact instance for shared maps.

        Returns:
            dict {mapName, position, tile, confidence, instance, source} or None.
            None means "I don't know" (battle, dialog, fade, or an unmatchable
            screen) - never a low-confidence guess.
        """
        screenshot = cv2.imread(screenshotPath)
        if screenshot is None:
            print(f'LocationTracker: Could not read screenshot at {screenshotPath}')
            return None

        detail = float(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY).std())
        if detail < self.MIN_SCREEN_STDDEV:
            print(f'LocationTracker: screen is flat (stddev {detail:.1f}) - mid '
                  f'warp fade, dialog, or transition. Reporting unknown.')
            return None

        # Phase 0: if game RAM tells us the exact map, that IS the identity.
        # Template matching can't distinguish visually identical maps (e.g. the
        # Route 2 <-> Viridian Forest gates), so bank/number wins outright when
        # available. We still template-match, but only against that one map, to
        # recover the player's pixel position within it.
        stateMap = self._mapFromState(gameState)
        ramTile = self._tileFromState(gameState, stateMap)
        best = None
        if stateMap is not None:
            best = self._matchAgainst(screenshot, [stateMap])
            if best is not None and best[1] >= self.MIN_CONFIDENCE:
                self._checkRamAgreement(stateMap, best, ramTile)
            elif ramTile is not None:
                # The image can't localize us but RAM still knows exactly where
                # we are. This is the normal case indoors: a room smaller than
                # the 240x160 screen doesn't scroll, so the screenshot is never
                # a sub-image of the map and matching is geometrically
                # impossible. Trusting RAM here is what keeps interiors working.
                return self._buildFix(stateMap, ramTile, 1.0, gameState,
                                      source='ram')
            else:
                print(f'LocationTracker: game state map {stateMap!r} did not '
                      f'match the screenshot and no RAM tile was available; '
                      f'falling back to template search.')
                best = None

        # Phase 1: try the current map + its neighbors (fast path).
        if best is None:
            candidates = self._neighborCandidates()
            best = self._matchAgainst(screenshot, candidates) if candidates else None

            # Phase 2: full scan if the fast path was missing or low-confidence.
            if best is None or best[1] < self.CONFIDENCE_THRESHOLD:
                full = self._matchAgainst(screenshot, list(self.maps.keys()))
                if full and (best is None or full[1] > best[1]):
                    best = full

        if best is None:
            print('LocationTracker: No matching map found.')
            return None

        mapName, confidence, location = best
        if confidence < self.MIN_CONFIDENCE:
            print(f'LocationTracker: best match {mapName} at conf '
                  f'{confidence:.3f} is below {self.MIN_CONFIDENCE} - reporting '
                  f'unknown rather than guessing.')
            return None

        playerX = location[0] + self.PLAYER_OFFSET_X
        playerY = location[1] + self.PLAYER_OFFSET_Y
        tile = (playerX // self.TILE_SIZE, playerY // self.TILE_SIZE)

        instance = self.disambiguateInstance(mapName, gameState)

        self.currentMap = mapName
        self.currentPosition = (playerX, playerY)
        self.currentTile = tile
        self.currentConfidence = confidence
        self.currentInstance = instance
        self._lastMapName = mapName

        print(f'LocationTracker: {mapName} tile {tile} '
              f'(conf {confidence:.3f}{", " + instance if instance else ""})')

        return {
            'mapName': mapName,
            'position': (playerX, playerY),
            'tile': tile,
            'confidence': confidence,
            'instance': instance,
            'source': 'template',
        }

    def _loadMapOffsets(self, connectionDataDir):
        """Load per-map RAM->image tile corrections (see mapOffsets.json)."""
        path = os.path.join(connectionDataDir, 'mapOffsets.json')
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
        for mapName, off in data.items():
            if mapName.startswith('_'):      # comment keys
                continue
            if isinstance(off, (list, tuple)) and len(off) == 2:
                self.mapOffsets[mapName] = (int(off[0]), int(off[1]))
        if self.mapOffsets:
            print(f'LocationTracker: {len(self.mapOffsets)} map coordinate '
                  f'offset(s) loaded.')

    def locateFromState(self, gameState):
        """Fix the player from RAM alone - no screenshot, no template match.

        Returns None when the map isn't registered in mapIds.json, in which case
        the caller still needs locatePlayer(). When it does work it is both
        faster and *more* correct than matching: RAM updates to the destination
        tile the moment a step begins, while the screen spends the next several
        frames animating the walk, so a screenshot taken mid-step matches the
        tile the player is leaving rather than the one they're entering.
        """
        mapName = self._mapFromState(gameState)
        if mapName is None:
            return None
        tile = self._tileFromState(gameState, mapName)
        if tile is None:
            return None
        return self._buildFix(mapName, tile, 1.0, gameState, source='ram')

    def _buildFix(self, mapName, tile, confidence, gameState, source):
        """Assemble a fix (and update cached state) from an already-known tile."""
        instance = self.disambiguateInstance(mapName, gameState)
        position = (tile[0] * self.TILE_SIZE, tile[1] * self.TILE_SIZE)

        self.currentMap = mapName
        self.currentPosition = position
        self.currentTile = tile
        self.currentConfidence = confidence
        self.currentInstance = instance
        self._lastMapName = mapName

        print(f'LocationTracker: {mapName} tile {tile} '
              f'(from {source}{", " + instance if instance else ""})')

        return {
            'mapName': mapName,
            'position': position,
            'tile': tile,
            'confidence': confidence,
            'instance': instance,
            'source': source,
        }

    def _tileFromState(self, gameState, mapName=None):
        """Player's (col, row) on the map image, from RAM, or None.

        SaveBlock1's x/y are map-local tile coordinates on the same grid the map
        images use, so this is normally the image tile outright (and
        _checkRamAgreement polices that claim as you play). Rips cropped tighter
        than the real map need a correction, which mapOffsets.json supplies.
        """
        if not gameState:
            return None
        player = gameState.get('player', gameState)
        x, y = player.get('x'), player.get('y')
        if x is None or y is None:
            return None
        dx, dy = self.mapOffsets.get(mapName, (0, 0))
        return (x - dx, y - dy)

    def _checkRamAgreement(self, mapName, best, ramTile):
        """Warn when a confident template match disagrees with RAM coordinates.

        Both should name the same tile. A persistent disagreement on one map
        means that map image's origin is offset from the game's coordinate
        space, which would make the RAM fallback above wrong for it.
        """
        if ramTile is None:
            return
        _name, _conf, location = best
        tile = ((location[0] + self.PLAYER_OFFSET_X) // self.TILE_SIZE,
                (location[1] + self.PLAYER_OFFSET_Y) // self.TILE_SIZE)
        # A one-tile gap is just the walk animation: RAM commits to the
        # destination tile as the step starts, so a screenshot taken mid-step
        # legitimately shows the previous tile. Only a larger, structural gap
        # means the image origin is actually misaligned with the game grid.
        drift = max(abs(tile[0] - ramTile[0]), abs(tile[1] - ramTile[1]))
        if drift > 1 and mapName not in self._undersizedWarned:
            self._undersizedWarned.add(mapName)
            print(f'LocationTracker: WARNING {mapName} template tile {tile} != '
                  f'RAM tile {ramTile}. The map image origin looks offset from '
                  f'the game grid; indoor RAM-based fixes on it will be wrong.')

    def _neighborCandidates(self):
        """Current map + its connection neighbors, in match-priority order."""
        if not self._lastMapName or self._lastMapName not in self.maps:
            return []
        names = [self._lastMapName]
        names += [n for n in self.neighbors.get(self._lastMapName, set())
                  if n in self.maps]
        return names

    def _matchAgainst(self, screenshot, mapNames):
        """Template-match the screenshot against the named maps; return best."""
        best = None
        for mapName in mapNames:
            mapImage = self.maps.get(mapName)
            if mapImage is None:
                continue
            if (mapImage.shape[0] < screenshot.shape[0] or
                    mapImage.shape[1] < screenshot.shape[1]):
                # Smaller than the screen, so it can never contain the
                # screenshot. That's most interiors; they're located from RAM
                # coordinates instead (see locatePlayer). Say so once per map
                # rather than skipping in silence.
                if mapName not in self._undersizedWarned:
                    self._undersizedWarned.add(mapName)
                    print(f'LocationTracker: {mapName} '
                          f'({mapImage.shape[1]}x{mapImage.shape[0]}) is smaller '
                          f'than the {screenshot.shape[1]}x{screenshot.shape[0]} '
                          f'screen - not template-matchable, needs a mapIds.json '
                          f'entry to be locatable.')
                continue
            result = cv2.matchTemplate(mapImage, screenshot, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
            if best is None or maxVal > best[1]:
                best = (mapName, maxVal, maxLoc)
        return best

    @staticmethod
    def _bankNumFromState(gameState):
        """Extract (bank, number) from a GAME_STATE dict, or None."""
        if not gameState:
            return None
        player = gameState.get('player', gameState)
        bank = player.get('map_bank')
        number = player.get('map_number')
        if bank is None or number is None:
            return None
        return (bank, number)

    def _mapFromState(self, gameState):
        """Resolve the authoritative mapName from game RAM, or None.

        Returns None when there's no game state, no bank/number, the pair isn't
        registered in mapIds.json, or the resolved map has no loaded image. In
        every None case locatePlayer falls back to template-matching search.
        """
        bankNum = self._bankNumFromState(gameState)
        if bankNum is None:
            return None
        mapName = self._mapNameByBankNum.get(bankNum)
        if mapName is None or mapName not in self.maps:
            return None
        return mapName

    def disambiguateInstance(self, mapName, gameState):
        """
        Resolve the world instance for a (possibly shared) map.

        Uses GAME_STATE's map_bank/map_number against the instance registry.
        Returns an instanceId, or None when not a shared map / not resolvable.
        """
        bankNum = self._bankNumFromState(gameState)
        if bankNum is None:
            return None
        return self._instanceByBankNum.get(bankNum)

    def getMapNames(self):
        return list(self.maps.keys())


if __name__ == '__main__':
    tracker = LocationTracker()
    result = tracker.locatePlayer('../screenshot.png')
    if result:
        print(f"\nMap:        {result['mapName']}")
        print(f"Tile:       {result['tile']}")
        print(f"Position:   {result['position']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Instance:   {result['instance']}")
    else:
        print('\nCould not determine player location.')
