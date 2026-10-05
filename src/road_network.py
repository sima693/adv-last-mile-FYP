"""
road_network.py
---------------
Loads, caches, and provides access to the real road network for route-feasibility
and actual road network distance calculations (replacing simple haversine Euclidean distances).

Caches graphml locally in data/birmingham_roads.graphml so Overpass API is only
queried once.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
import osmnx as ox
import networkx as nx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_ROAD_CACHE = DATA_DIR / "birmingham_roads.graphml"


def get_road_network(
    place: str = "Birmingham, UK",
    cache_path: Path | str = DEFAULT_ROAD_CACHE,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """Loads the road network graph from disk if cached, or downloads from OpenStreetMap via OSMnx.

    Parameters
    ----------
    place : Place query string for Nominatim (default: "Birmingham, UK")
    cache_path : Path to save/load the .graphml file
    network_type : OSMnx network type (default: "drive")
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"[road_network] Loading cached graph from {cache_path}...")
        G = ox.load_graphml(cache_path)
        print(f"[road_network] Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G

    print(f"[road_network] Downloading road network for '{place}' via OSMnx...")
    ox.settings.use_cache = True
    ox.settings.log_console = True

    G = ox.graph_from_place(place, network_type=network_type)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, cache_path)
    print(f"[road_network] Successfully saved graph to {cache_path}")
    return G


if __name__ == "__main__":
    G = get_road_network()
    print(f"Road network ready with {len(G.nodes):,} nodes and {len(G.edges):,} edges.")
