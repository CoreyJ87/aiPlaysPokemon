#!/usr/bin/env python
"""Generate tileData grids + connections from the pret/pokefirered decomp.

The manual bottleneck in this project is hand-painting every tile of every
map in mapEditor.py. All of that information already exists as data in the
decompilation the map images came from: map.bin holds per-tile collision,
the tileset attribute files hold per-metatile behavior (grass, water,
ledges, doors, escalators), and map.json holds every warp and map edge.

    python generateFromDecomp.py --list             # what would be generated
    python generateFromDecomp.py --check            # diff against painted maps
    python generateFromDecomp.py Route4 MtMoon_1F   # specific maps
    python generateFromDecomp.py --all-missing      # everything unpainted
    python generateFromDecomp.py --force <name>     # overwrite existing paint

Maps that already have tile data are never touched unless --force is given -
several were fixed by hand from live play (the PC escalators), and paint
that survived contact with the game beats regenerated paint.

Coordinates: the repo's grids are aligned to the map *images*, which carry a
border around the real map. The border is centered, so the game->image offset
is ((imgW-mapW)//2, (imgH-mapH)//2), with the image size read from the PNG.

FireRed and LeafGreen share all map data, so one decomp serves both.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).parent
DECOMP = Path.home() / "repos" / "pokefirered"
TILE_DIR = HERE / "tileData"
MAPS_DIR = HERE / "maps"
CONN_PATH = HERE / "connectionData" / "connections.json"

# Repo tile type codes (mapEditor.py TILE_TYPES).
UNKNOWN, WALKABLE, BLOCKED, TALL_GRASS, WATER = 0, 1, 2, 3, 4
CUTTABLE, LEDGE_DOWN, LEDGE_LEFT, LEDGE_RIGHT = 5, 6, 7, 8
DOOR, WARP, BOULDER, SMASH_ROCK, ITEM = 9, 10, 11, 12, 13

# Metatile behaviors (include/constants/metatile_behaviors.h).
MB_TALL_GRASS = {0x02, 0x03}
MB_WATER = {0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x17, 0x1A, 0x1B}
MB_WATERFALL = {0x13}
MB_LEDGE = {0x38: LEDGE_RIGHT, 0x39: LEDGE_LEFT, 0x3B: LEDGE_DOWN}
MB_WARPS = {0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x69,
            0x6C, 0x6D, 0x6E, 0x6F, 0x71}
MB_ESCALATOR = {0x6A, 0x6B}

OBJ_TILES = {
    "OBJ_EVENT_GFX_CUT_TREE": CUTTABLE,
    "OBJ_EVENT_GFX_ROCK_SMASH_ROCK": SMASH_ROCK,
    "OBJ_EVENT_GFX_PUSHABLE_BOULDER": BOULDER,
    "OBJ_EVENT_GFX_ITEM_BALL": ITEM,
}

DIR_NAMES = {"up": "north", "down": "south", "left": "west", "right": "east"}

# Tiles an edge crossing may stand on - water included, because the sea
# routes (19/20/21) connect to their neighbours while surfing.
CROSSABLE = (WALKABLE, TALL_GRASS, WATER)


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z0-9]{1}[a-z])|(?<=[a-z])(?=[A-Z0-9])",
                  "_", name).lower()


class Decomp:
    """Lazy access to the pret data files."""

    def __init__(self, root: Path):
        self.root = root
        self.layouts = {l["id"]: l for l in
                        json.load(open(root / "data/layouts/layouts.json"))
                        ["layouts"] if l}
        groups = json.load(open(root / "data/maps/map_groups.json"))
        self.bankNum = {}
        for bank, gname in enumerate(groups["group_order"]):
            for num, mapName in enumerate(groups[gname]):
                self.bankNum[mapName] = (bank, num)
        self._mapJson, self._attrs, self._blocks = {}, {}, {}
        # Map ids come from each map.json's own "id" field - deriving them
        # from the name gets Route25 wrong (MAP_ROUTE25, not MAP_ROUTE_25).
        self.idToName = {}
        for name in self.bankNum:
            p = root / "data/maps" / name / "map.json"
            if p.exists():
                self.idToName[json.load(open(p))["id"]] = name

    def mapJson(self, name):
        if name not in self._mapJson:
            self._mapJson[name] = json.load(
                open(self.root / "data/maps" / name / "map.json"))
        return self._mapJson[name]

    def _tilesetAttrs(self, tilesetName):
        if tilesetName not in self._attrs:
            dirName = snake(tilesetName.replace("gTileset_", ""))
            for kind in ("primary", "secondary"):
                p = self.root / "data/tilesets" / kind / dirName / \
                    "metatile_attributes.bin"
                if p.exists():
                    raw = p.read_bytes()
                    self._attrs[tilesetName] = struct.unpack(
                        f"<{len(raw)//4}I", raw)
                    break
            else:
                raise FileNotFoundError(f"no attributes for {tilesetName} "
                                        f"(looked for {dirName})")
        return self._attrs[tilesetName]

    def grid(self, name):
        """(width, height, [[(collision, behavior)]]) in game space."""
        layout = self.layouts[self.mapJson(name)["layout"]]
        w, h = layout["width"], layout["height"]
        raw = (self.root / layout["blockdata_filepath"]).read_bytes()
        blocks = struct.unpack(f"<{len(raw)//2}H", raw)
        primary = self._tilesetAttrs(layout["primary_tileset"])
        secondary = self._tilesetAttrs(layout["secondary_tileset"])
        out = []
        for r in range(h):
            row = []
            for c in range(w):
                block = blocks[r * w + c]
                mt = block & 0x03FF
                collision = block & 0x0C00
                attrs = primary[mt] if mt < 640 else secondary[mt - 640]
                row.append((collision, attrs & 0x1FF))
            out.append(row)
        return w, h, out


def png_size(path: Path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    w, h = struct.unpack(">II", head[16:24])
    return w // 16, h // 16


def repo_name(decomp: Decomp, pretName: str) -> str:
    bank, num = decomp.bankNum[pretName]
    return f"{bank}-{num}-{pretName}"


def image_offset(repoName: str, mapW: int, mapH: int):
    """Game->image tile offset, from the centered border in the map PNG."""
    png = MAPS_DIR / f"{repoName}.png"
    if not png.exists():
        return None
    imgW, imgH = png_size(png)
    if imgW < mapW or imgH < mapH:
        raise ValueError(f"{repoName}: image {imgW}x{imgH} smaller than "
                         f"map {mapW}x{mapH}")
    return (imgW - mapW) // 2, (imgH - mapH) // 2, imgW, imgH


def tile_code(collision: int, behavior: int) -> int:
    if behavior in MB_ESCALATOR:
        return BLOCKED            # enterable from one face only; the
        # connection stands on its open neighbour
    if behavior in MB_LEDGE:
        # Ledge metatiles carry collision - the hop is a behavior override
        # in the movement engine - so this must win over the collision bit.
        return MB_LEDGE[behavior]
    if behavior in MB_WARPS:
        # Outdoor door mats are collision-blocked too; the door-opening
        # animation is what carries you through. Same override as ledges.
        return DOOR
    if collision:
        return BLOCKED
    if behavior in MB_TALL_GRASS:
        return TALL_GRASS
    if behavior in MB_WATER:
        return WATER
    if behavior in MB_WATERFALL:
        return BLOCKED
    if behavior == 0x3A:          # MB_JUMP_NORTH has no repo code
        return BLOCKED
    if behavior in MB_WARPS:
        return DOOR
    return WALKABLE


def build_map(decomp: Decomp, pretName: str):
    """(repoName, tileData dict, [connection dicts]) for one map."""
    rname = repo_name(decomp, pretName)
    w, h, grid = decomp.grid(pretName)
    off = image_offset(rname, w, h)
    if off is None:
        offX, offY, imgW, imgH = 0, 0, w, h
        print(f"  {rname}: no map image - grid is unpadded, game-aligned")
    else:
        offX, offY, imgW, imgH = off

    tiles = [[BLOCKED] * imgW for _ in range(imgH)]
    for r in range(h):
        for c in range(w):
            tiles[r + offY][c + offX] = tile_code(*grid[r][c])

    mj = decomp.mapJson(pretName)
    items = {}
    for obj in mj.get("object_events", []):
        code = OBJ_TILES.get(obj.get("graphics_id"))
        if code is None:
            continue
        col, row = obj["x"] + offX, obj["y"] + offY
        if not (0 <= row < imgH and 0 <= col < imgW):
            continue              # maps own sprites past their own seams
        tiles[row][col] = code
        if code == ITEM:
            m = re.search(r"Item(\w+)$", obj.get("script", ""))
            items[f"{row},{col}"] = (re.sub(r"(?<!^)(?=[A-Z0-9])", " ",
                                            m.group(1)) if m else "item")

    connections = []
    # Warps: doors, stairs, ladders, escalators.
    warps = mj.get("warp_events", [])
    for i, warp in enumerate(warps):
        destId = warp["dest_map"]
        if destId == "MAP_DYNAMIC" or destId not in decomp.idToName:
            continue
        destPret = decomp.idToName[destId]
        destRepo = repo_name(decomp, destPret)
        try:
            dw, dh, _dgrid = decomp.grid(destPret)
            doff = image_offset(destRepo, dw, dh) or (0, 0, dw, dh)
        except (FileNotFoundError, KeyError):
            continue
        dwarps = decomp.mapJson(destPret).get("warp_events", [])
        di = int(warp["dest_warp_id"])
        if di >= len(dwarps):
            continue
        col, row = warp["x"] + offX, warp["y"] + offY
        # An escalator body is solid; stand on its open (walkable) neighbour
        # and let the navigator discover the press that boards it.
        behavior = grid[warp["y"]][warp["x"]][1]
        if behavior in MB_ESCALATOR:
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = warp["x"] + dc, warp["y"] + dr
                if (0 <= nc < w and 0 <= nr < h
                        and tile_code(*grid[nr][nc]) == WALKABLE):
                    col, row = nc + offX, nr + offY
                    tiles[row][col] = DOOR
                    break
        # Script-controlled doors (the E4 rooms' sliding barriers, Silph's
        # locked doors) sit on metatiles that read as plain wall; the warp
        # event is the truth. Paint the tile as a door so it is routable -
        # the game itself still decides whether it opens.
        if behavior not in MB_ESCALATOR and tiles[row][col] in (UNKNOWN,
                                                                BLOCKED):
            tiles[row][col] = DOOR
        connections.append({
            "type": "door",
            "fromTile": [col, row],
            "toMap": destRepo,
            "toTile": [dwarps[di]["x"] + doff[0], dwarps[di]["y"] + doff[1]],
        })

    # A door is only useful if the room is reachable from it. Script-managed
    # rooms (the E4 chain) bar the tile just inside the door too; carve the
    # shortest straight path from any isolated door toward the map's centre
    # until it meets open floor, and let the game's own walls arbitrate live.
    OPEN = (WALKABLE, TALL_GRASS, DOOR, WATER)
    for conn in connections:
        col, row = conn["fromTile"]
        if any(0 <= r2 < imgH and 0 <= c2 < imgW and tiles[r2][c2] in OPEN
               for c2, r2 in ((col+1, row), (col-1, row),
                              (col, row+1), (col, row-1))):
            continue
        dc = (1 if col < imgW // 2 else -1) if abs(col - imgW // 2) > \
            abs(row - imgH // 2) else 0
        dr = 0 if dc else (1 if row < imgH // 2 else -1)
        c2, r2 = col + dc, row + dr
        for _ in range(4):
            if not (0 <= r2 < imgH and 0 <= c2 < imgW):
                break
            if tiles[r2][c2] in OPEN:
                break
            tiles[r2][c2] = WALKABLE
            c2, r2 = c2 + dc, r2 + dr

    # Map edges: crossable wherever both sides are walkable.
    for conn in mj.get("connections") or []:
        destId = conn["map"]
        if destId not in decomp.idToName:
            continue
        destPret = decomp.idToName[destId]
        destRepo = repo_name(decomp, destPret)
        try:
            dw, dh, dgrid = decomp.grid(destPret)
            doff = image_offset(destRepo, dw, dh) or (0, 0, dw, dh)
        except (FileNotFoundError, KeyError):
            continue
        direction, shift = conn["direction"], conn["offset"]
        spans = []
        if direction in ("up", "down"):
            rHere = 0 if direction == "up" else h - 1
            rDest = dh - 1 if direction == "up" else 0
            for c in range(w):
                cd = c - shift
                if not 0 <= cd < dw:
                    continue
                if (tile_code(*grid[rHere][c]) in CROSSABLE
                        and tile_code(*dgrid[rDest][cd]) in CROSSABLE):
                    spans.append((c, cd))
        else:
            cHere = 0 if direction == "left" else w - 1
            cDest = dw - 1 if direction == "left" else 0
            for r in range(h):
                rd = r - shift
                if not 0 <= rd < dh:
                    continue
                if (tile_code(*grid[r][cHere]) in CROSSABLE
                        and tile_code(*dgrid[rd][cDest]) in CROSSABLE):
                    spans.append((r, rd))
        # contiguous runs -> one connection each, anchored mid-run
        run = []
        for pair in spans + [None]:
            if pair is not None and (not run or pair[0] == run[-1][0] + 1):
                run.append(pair)
                continue
            if run:
                a, ad = run[len(run) // 2]
                if direction == "up":
                    frm, to = [a + offX, offY - 1], [ad + doff[0],
                                                     doff[1] + dh - 1]
                elif direction == "down":
                    frm, to = [a + offX, offY + h], [ad + doff[0], doff[1]]
                elif direction == "left":
                    frm, to = [offX - 1, a + offY], [doff[0] + dw - 1,
                                                     ad + doff[1]]
                else:
                    frm, to = [offX + w, a + offY], [doff[0], ad + doff[1]]
                # A 0-border image has no room for the outside-the-map tile;
                # anchor the crossing on the boundary tile itself instead.
                frm[0] = max(0, min(imgW - 1, frm[0]))
                frm[1] = max(0, min(imgH - 1, frm[1]))
                tiles[frm[1]][frm[0]] = DOOR   # make the crossing routable
                connections.append({
                    "type": "edge", "fromTile": frm, "toMap": destRepo,
                    "toTile": to, "direction": DIR_NAMES[direction],
                    "width": len(run),
                })
            run = [pair] if pair else []

    data = {"mapName": rname, "widthTiles": imgW, "heightTiles": imgH,
            "tiles": tiles}
    if items:
        data["items"] = items
    return rname, data, connections


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("maps", nargs="*", help="pret map names (e.g. Route4)")
    ap.add_argument("--decomp", type=Path, default=DECOMP)
    ap.add_argument("--all-missing", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="diff generated grids against painted ones")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing tileData")
    args = ap.parse_args()

    decomp = Decomp(args.decomp)

    have_image = {p.stem for p in MAPS_DIR.glob("*.png")}
    painted = {p.stem for p in TILE_DIR.glob("*.json")}
    known = {repo_name(decomp, n): n for n in decomp.bankNum
             if (args.decomp / "data/maps" / n / "map.json").exists()}
    missing = sorted(r for r in known
                     if r in have_image and r not in painted)

    if args.list:
        print(f"{len(missing)} maps with an image but no tile data:")
        for r in missing:
            print(" ", r)
        return

    if args.check:
        agree = total = 0
        for r in sorted(known.keys() & painted):
            try:
                _, gen, _ = build_map(decomp, known[r])
            except Exception as exc:
                print(f"  {r}: SKIP ({exc})")
                continue
            old = json.load(open(TILE_DIR / f"{r}.json"))
            if (old.get("widthTiles") != gen["widthTiles"]
                    or old.get("heightTiles") != gen["heightTiles"]):
                print(f"  {r}: SIZE {old.get('widthTiles')}x"
                      f"{old.get('heightTiles')} vs {gen['widthTiles']}x"
                      f"{gen['heightTiles']}")
                continue
            same = diff = painted_n = 0
            for r_ in range(gen["heightTiles"]):
                for c_ in range(gen["widthTiles"]):
                    o, g = old["tiles"][r_][c_], gen["tiles"][r_][c_]
                    if o == UNKNOWN:
                        continue
                    painted_n += 1
                    if o == g or {o, g} <= {ITEM, WALKABLE} \
                            or {o, g} <= {DOOR, WALKABLE}:
                        same += 1
                    else:
                        diff += 1
            pct = 100 * same / painted_n if painted_n else 0
            print(f"  {r}: {pct:5.1f}% agree ({diff} disagree "
                  f"of {painted_n} painted)")
            agree += same
            total += painted_n
        if total:
            print(f"TOTAL: {100*agree/total:.1f}% agreement")
        return

    targets = args.maps or (missing if args.all_missing else [])
    if not targets:
        ap.error("give map names, --all-missing, --list or --check")

    conns = json.load(open(CONN_PATH))
    written = 0
    for t in targets:
        pretName = known.get(t, t)          # accept repo or pret names
        if pretName not in decomp.bankNum:
            print(f"  {t}: unknown map, skipping")
            continue
        rname = repo_name(decomp, pretName)
        if rname in painted and not args.force:
            print(f"  {rname}: already painted, skipping (--force to redo)")
            continue
        rname, data, connections = build_map(decomp, pretName)
        old_path = TILE_DIR / f"{rname}.json"
        if old_path.exists():
            # Hand-added knowledge survives regeneration: object tags (the
            # nurse, the mart clerk, PCs), encounter overrides, and any
            # hand-marked items the decomp doesn't know about.
            old = json.load(open(old_path))
            for key in ("objects", "objectCategories", "encounters"):
                if old.get(key):
                    data[key] = old[key]
            if old.get("items"):
                merged = dict(data.get("items") or {})
                merged.update(old["items"])
                data["items"] = merged
        json.dump(data, open(old_path, "w"), indent=1)
        entry = conns["maps"].setdefault(rname, {})
        entry["imageFile"] = f"{rname}.png"
        entry["widthTiles"] = data["widthTiles"]
        entry["heightTiles"] = data["heightTiles"]
        if connections or not entry.get("connections"):
            entry["connections"] = connections
        print(f"  {rname}: {data['widthTiles']}x{data['heightTiles']}, "
              f"{len(connections)} connection(s)")
        written += 1

    if written:
        json.dump(conns, open(CONN_PATH, "w"), indent=1)
        print(f"wrote {written} map(s) + connections.json")


if __name__ == "__main__":
    main()
