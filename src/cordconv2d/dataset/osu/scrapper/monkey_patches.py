from typing import Optional

from osu import AsynchronousClient, Beatmapset
from osu.path import Path
from osu.results import BeatmapsetSearchResult
from osu.util import BeatmapsetSearchFilter


async def search_beatmapsets(
        self: AsynchronousClient, filters: Optional[BeatmapsetSearchFilter] = None, page: Optional[int | str] = None
    ) -> BeatmapsetSearchResult:
        """
        Search for beatmapsets.

        Requires OAuth and scope public.

        **Parameters**

        filters: Optional[:class:`BeatmapsetSearchFilter`]

        page: Optional[int]

        **Returns**

        :class:`BeatmapsetSearchResult`
        """
        if filters is None:
            filters = {}
        if isinstance(filters, BeatmapsetSearchFilter):
            filters = filters.filters
        resp = await self.http.make_request(Path.beatmapset_search(), cursor_string=page, **filters)

        cursor = resp["cursor"] | {"id": resp["cursor_string"]}
        return BeatmapsetSearchResult(
            list(map(Beatmapset, resp["beatmapsets"])),
            cursor,
            resp["search"],
            resp["recommended_difficulty"],
            resp["error"],
            resp["total"],
        )

def apply_monkey_patches():
     AsynchronousClient.search_beatmapsets = search_beatmapsets