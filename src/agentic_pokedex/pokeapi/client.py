"""A small client for PokéAPI (https://pokeapi.co).

Every tool talks to PokéAPI through this one class, so HTTP details, caching
and error handling live in a single place.
"""

import requests


class PokeAPINotFound(Exception):
    """Raised when PokéAPI has no entry for the requested resource/name."""


class PokeAPIClient:
    def __init__(self, base_url: str = "https://pokeapi.co/api/v2", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()  # reuses the connection between calls
        self._cache: dict[str, dict] = {}  # url -> JSON, so repeat lookups are free

    @staticmethod
    def normalize(name: str | int) -> str:
        """PokéAPI wants lowercase, hyphenated names: 'Mr. Mime' -> 'mr-mime'."""
        text = str(name).strip().lower()
        for char in (".", "'", ":"):
            text = text.replace(char, "")
        return "-".join(text.split())

    def _get_json(self, url: str) -> dict:
        if url not in self._cache:
            response = self._session.get(url, timeout=self.timeout)
            if response.status_code == 404:
                raise PokeAPINotFound(url)
            response.raise_for_status()
            self._cache[url] = response.json()
        return self._cache[url]

    def list_resources(self) -> list[str]:
        """The main endpoint: names of every resource PokéAPI offers ('pokemon', 'ability', ...)."""
        return sorted(self._get_json(f"{self.base_url}/").keys())

    def get(self, resource: str, name: str | int) -> dict:
        """Fetch one entry, e.g. get('pokemon-species', 'Torchic')."""
        return self._get_json(f"{self.base_url}/{resource}/{self.normalize(name)}/")
