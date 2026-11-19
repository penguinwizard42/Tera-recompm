# Trade evolutions without trading

Scarlet and Violet have no trading in this port, so some Pokémon could never evolve. With the setting
**No trade evolutions** (F1 menu, Graphics tab; `no_trade_evo = 1` in `totk.ini`) they evolve as listed below. The
setting is off by default and takes effect at the next start. `runtime/sv_census.c` (`evo_fix`) rewrites the game's
species table as the game loads it; the levels and items are the choices of
[KarPur](https://github.com/Grunttanamo/KarPur) (MIT).

## On reaching a level

| Pokémon | Evolves into | Level |
|---|---|---|
| Kadabra | Alakazam | 40 |
| Machoke | Machamp | 39 |
| Graveler (both forms) | Golem | 36 |
| Haunter | Gengar | 42 |
| Onix | Steelix | 40 |
| Scyther | Scizor | 30 |
| Boldore | Gigalith | 35 |
| Gurdurr | Conkeldurr | 40 |
| Spritzee | Aromatisse | 20 |
| Swirlix | Slurpuff | 20 |
| Phantump | Trevenant | 30 |
| Pumpkaboo (all sizes) | Gourgeist | 30 |

Onix, Scyther, Spritzee and Swirlix normally also need a held item (Metal Coat, Sachet, Whipped Dream). The game has
no internal item number for those items that a non-trade evolution can use, and the substitute items KarPur tried did
not work in play, so they evolve on a level with no item.

## On levelling up by day while holding an item

The Pokémon must hold the item and gain a level during the day (in-game time).

| Pokémon | Evolves into | Item to hold |
|---|---|---|
| Rhydon | Rhyperior | Protector |
| Seadra | Kingdra | Dragon Scale |
| Electabuzz | Electivire | Electirizer |
| Magmar | Magmortar | Magmarizer |
| Porygon | Porygon2 | Upgrade |
| Porygon2 | Porygon-Z | Dubious Disc |
| Feebas | Milotic | Prism Scale |
| Dusclops | Dusknoir | Reaper Cloth |
| Poliwhirl | Politoed | Leader's Crest (instead of King's Rock) |
| Slowpoke | Slowking | Leader's Crest (instead of King's Rock) |
| Clamperl | Huntail | Prism Scale (instead of Deep Sea Tooth) |
| Clamperl | Gorebyss | Splash Plate (instead of Deep Sea Scale) |

The King's Rock and the two Deep Sea items cannot be named in a non-trade evolution in this game, so Poliwhirl,
Slowpoke and Clamperl use a substitute item. Those substitutes are untested in play; if one does not work, report it.
