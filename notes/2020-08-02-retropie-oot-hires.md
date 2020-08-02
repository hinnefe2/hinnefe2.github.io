---
layout: post
title: "High Resolution Textures for Ocarina of Time on a RetroPie in 2020"
tags:
    - retropie
categories: 
    - retropie
--- 

This is a collection of things I learned while trying (eventually successfully)
to get The Legend of Zelda: Ocarina of Time running with high resolution
textures on a RetroPie.

## TLDR

RetroPie forces OOT to run using the Rice graphics plugin for the mupen64plus
N64 emulator, even when you manually specify a different graphics plugin when
launching the game.  The Rice graphics plugin reads "loose pack" textures from 
```
/home/pi/.local/share/mupen64plus/hires_texture/<internal ROM name>/
```
so copying a "loose pack" of textures for OOT to 
```
/home/pi/.local/share/mupen64plus/hires_texture/THE LEGEND OF ZELDA/
```
 works.

## ROM Header Names

ROM files include a [header](https://sneslab.net/wiki/SNES_ROM_Header) which
contains metadata about the game. One piece of metadata is the internal name of
the game. For OOT this internal name is `THE LEGEND OF ZELDA`.

Note that the internal ROM name is not the same thing as the file name. For
loading texture packs the internal ROM name is what matters, not the filename.

## Texture pack formats

Texture packs for N64 emulators come in a couple different
[formats](https://techtipswiththemuss.wordpress.com/2018/07/29/how-to-use-hi-res-textures-with-the-standalone-mupen64plus-emulator/):
- **"Loose" packs** - collections of PNG images
- **.htc files** - compiled texture "cache" files

Some instructions I found when Googling focused on converting one format to the
other, but that ended up being unnecessary for me. I download textures from
[emulationking.com](https://emulationking.com/legend-zelda-ocarina-time/) which
includes separate download links for each format.

## N64 Emulator Graphics Plugins

The N64 emulator [mupen64plus](https://mupen64plus.org/) can use one of several
different graphics plugins. 
- GLide64 -  outdated, included here only to distinguish it from the distinct plugin Glide**N**64, mentioned below.
- [GLideN64](https://github.com/gonetz/GLideN64) - works with .htc texture pack files
- [Rice](https://github.com/mupen64plus/mupen64plus-video-rice) - works with "loose pack" textures


## Launching games in RetroPie 
### `runcommand.sh`

When you launch a game from the UI, RetroPie calls a script called
[`runcommand`]([https://retropie.org.uk/docs/Runcommand/](https://retropie.org.uk/docs/Runcommand/))
to launch the emulator and load the game. This command [pops up a
window]([https://retropie.org.uk/docs/Runcommand/#runcommand-launch-menu](https://retropie.org.uk/docs/Runcommand/#runcommand-launch-menu))
which lets you configure the emulator. One thing you can configure from the
popup menu is which graphics plugin to use.

The output of `runcommand` is logged to `/dev/shm/runcommand.log` which can be
very helpful for debugging.

### RetroPie emulator-specific startup scripts

The `runcommand` script eventually calls another script to start the actual
emulator. This emulator-specific script (e.g.
[here](https://github.com/RetroPie/RetroPie-Setup/blob/master/scriptmodules/emulators/mupen64plus/mupen64plus.sh)
for the mupen64plus emulator) sets a bunch of configuration options to make the
game run smoothly on a Raspberry Pi.

This emulator-specific script **can override** your selection for which
graphics plugin to use.
[Here](https://github.com/RetroPie/RetroPie-Setup/blob/25314f750371739ab580c3958660f7901b051e5f/scriptmodules/emulators/mupen64plus/mupen64plus.sh#L219-L220)
and
[here](https://github.com/RetroPie/RetroPie-Setup/blob/25314f750371739ab580c3958660f7901b051e5f/scriptmodules/emulators/mupen64plus/mupen64plus.sh#L274-L276)
the startup script for the `mupen64plus` emulator configures the emulator to
use the Rice plugin for any ROM with "zelda" in the file name.
