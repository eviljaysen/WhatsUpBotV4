# CK Alliance War Game — Rules, Objectives & Guardrails

> This file documents the game mechanics WhatsUpBot monitors. It serves two purposes:
> 1. Help developers understand game rules so OCR results can be validated
> 2. Define guardrails — when scanned values violate game rules, something is wrong

---

## Game Overview

CK Alliance is a competitive team-vs-team war game played inside BlueStacks (Android
emulator). Two teams face off in a timed war. Each team defends **6 buildings** on a
map by placing armed vehicles (cars) in them. Points accumulate automatically based on
building strength. The first team to reach the **max points** threshold wins instantly;
otherwise the team with more points when the timer expires wins.

WhatsUpBot scans the game window via screenshot + OCR to generate Discord defense reports
for the alliance. It reads player names, captures build screenshots, tracks HP/ATK/score/
timer values, and produces strategic reports with placement recommendations and alerts.

---

## Objectives

### How to Win
1. **Reach max points before the opponent** — triggers an instant win
2. **Have more points when the timer expires** — standard win condition
3. **Defend all 6 buildings with strong cars** — bonus rate = building strength → faster scoring

### How to Lose
1. **Opponent reaches max points first** — instant loss
2. **Have fewer points when time runs out** — standard loss
3. **Leave buildings empty or weak** — lower bonus rate → slower scoring
4. **Fail to place cars** — empty buildings contribute 0 to team bonus

---

## Core Mechanics

### Buildings (6 total per team)
- Each team has **6 buildings** on the map, numbered 1–6
- Buildings hold defensive cars placed by team members
- Building strength = sum of all car HP + ATK values inside it
- Stronger buildings → higher team bonus → faster point accumulation
- Empty buildings are a strategic liability — contribute nothing

### Cars / Vehicles
- Each car has two stats: **HP** (hit points) and **ATK** (attack power)
- Typical stat range: **1,000 – 145,000** per stat
- Cars are placed into buildings to defend them
- Each car shows as a "slot" when cycling through a building's contents

### Players
- Each player can place **up to 3 cars total** across all 6 buildings
- A player may put all 3 cars in one building or spread them across buildings
- **Max 3 per player is a hard game rule** — seeing > 3 means wrap detection failed
- Players are identified by name displayed on their car's info card

### Scoring
| Component | Description |
|---|---|
| **Team Points** | Our team's accumulated score |
| **Opponent Points** | Enemy team's accumulated score |
| **Max Points** | Victory threshold (e.g., 145,000) |
| **Team Bonus** | Points earned per minute by our team (e.g., +145/min) |
| **Opponent Bonus** | Points earned per minute by the enemy team |

Bonus rates derive from total building strength. Points accumulate automatically —
no player action required once cars are placed.

### Timer
- Wars have a countdown timer (e.g., "17h 23m")
- Maximum realistic war duration: **48 hours**
- When the timer reaches zero, the team with more points wins
- An instant win can end the war before the timer expires

### Instant Win Calculation
```
minutes_to_instant = (max_points - current_score) / bonus_rate

Example:
  max_points = 145,000
  opp_points = 130,000  (needs 15,000 more)
  opp_bonus  = +500/min
  → Opponent instants in 30 minutes
```

### Defending vs Attacking
- Each car slot shows a **DEFENDING** (green) or **ATTACKING** (red) status
- Team-side cars are normally DEFENDING
- The status pixel color determines this: green channel ≥ red = DEFENDING

---

## Game Constraints (Guardrails for Error Detection)

These are hard game rules. When OCR results violate them, the data is wrong
and should be logged as a warning or error.

### Absolute Limits

| Rule | Limit | If Violated |
|---|---|---|
| Max cars per player (total) | 3 | ERROR — impossible, wrap detection failed |
| Max cars per player per building | 3 | WARNING — likely wrap detection failure |
| Max buildings per team | 6 | Hard-coded, cannot vary |
| Max total team slots | 75 | WARNING — scan overcounted |
| Max enemy slots per building | 14 | Config limit, not a game limit |

### Value Ranges (OCR Sanity)

| Value | Valid Range | If Outside |
|---|---|---|
| HP per car | 1,000 – 200,000 | Reject < 1,000 as OCR garbage |
| ATK per car | 1,000 – 200,000 | Reject < 1,000 as OCR garbage |
| Team/Opp Points | 0 – max_points | Reject negative or > max_points |
| Bonus rate | 0 – 500/min | WARNING if > 400 (sanity check) |
| Timer hours | 0 – 48 | Reject > 48 as OCR garbage |
| Timer minutes | 0 – 59 | Reject ≥ 60 |
| Max points | 50,000 – 500,000 | Typical: ~145,000 |

### Structural Rules

| Rule | Expected | If Violated |
|---|---|---|
| Slots per building | 1–15 typical | WARNING if > 20 (wrap failed) |
| Player appears in ≤ 6 buildings | Max 6 (one per building) | WARNING if > 6 |
| Player total slots ≤ 18 | 6 buildings × 3 max | ERROR — impossible |
| Player total slots ≤ 3 | Normal maximum | WARNING if > 3 (placement bug or name collision) |
| Total slots ≤ 75 | `bot_slots_total` config | WARNING if exceeded |
| Building not empty AND strength = 0 | Shouldn't happen | WARNING — OCR failed on all cars |

---

## Error Logging Strategy

The guardrails above map directly to logging levels in the codebase:

### CRITICAL — Immediate attention
- Opponent will instant-win within alert threshold (default: 30 minutes)
- Unhandled crash during scan

### ERROR — Something is definitely wrong
- Player has > 18 slots (impossible — game caps at 6 buildings × 3 cars)
- `on_slot()` callback crashed during scan
- Discord webhook failed to post

### WARNING — Suspicious, investigate
- HP or ATK value < 1,000 rejected (CNN returned garbage like "2")
- Timer string is all-same-digit (e.g., "11111" from CNN garbage)
- Player has > 3 cars in one building (wrap detection may have failed)
- Total slots exceed `bot_slots_total` (75)
- Player has > 6 total slots (suspicious duplication)
- Building strength below alert threshold
- Bonus rate > 400 (sanity check — probably OCR misread)
- Score > max_points (OCR misread or display glitch)
- Frame did not change after clicking Next (stuck detection)
- Slot cycling visited > 20 slots in one building

### INFO — Normal operation
- Scan progress (building N/6, slot N captured)
- Player name matched (which method: CNN, template, Tesseract)
- Wrap detected (name + HP/ATK + fingerprint match)
- Score values read (team, opp, max, bonuses)
- Timer parsed successfully
- Training started/completed

### DEBUG — Verbose diagnostics
- OCR raw text before cleanup
- Fingerprint diff values for wrap comparisons
- Template match confidence scores
- CNN confidence values
- Cache hits/misses for stats and templates

---

## Strategic Context (for Analysis Features)

### Building Strength Analysis
- **Strong building**: High total HP + ATK across all slots
- **Weak building**: Low total — priority target for reinforcement
- **Empty building**: Zero defenders — highest priority to fill
- Alert fires when any building strength < configurable threshold (default: 5,000,000)

### Placement Recommendations
The bot recommends where unplaced players should put their cars:
1. **Fill empty buildings first** — any defense > no defense
2. **Reinforce weakest buildings** — balance overall defense
3. **Recommend strongest available players** — maximize impact

### Score Momentum
Tracks how fast each team is scoring across multiple scans:
- **Velocity**: points gained per minute
- **Acceleration**: change in velocity (gaining/losing momentum)
- **Trend**: accelerating, steady, or decelerating

### Win Projection
Using current scores, bonus rates, and time remaining:
- Calculate if either team will reach max points before timer expires
- Show estimated time to instant win for each team
- Flag danger when opponent is close to instanting

---

## Map Layout Reference

### Team Building Positions (1920×1080 baseline)
```
        B6 (1473, 507)
                            B1 (438, 534)
B5 (1498, 778)
                        B2 (349, 772)
    B4 (1174, 930)
                B3 (739, 912)
```

### HUD Layout
```
┌─────────────────────────────────────────────┐
│           [Opponent Name]                   │
│              [Timer]                        │
│  [Team Pts] [+Bonus]  [Max]  [Opp Pts]     │
│                                             │
│  ┌──────────┐                               │
│  │ Build    │  [Player Name]                │
│  │ Card     │  [DEF/ATK status]             │
│  │ Image    │                               │
│  │          │                               │
│  └──────────┘                               │
│  [HP value]     [ATK value]                 │
│                                             │
│            [NEXT button]                    │
└─────────────────────────────────────────────┘
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Slot** | One car position in a building; cycling through shows each slot |
| **Wrap** | When slot cycling returns to the first car (all slots visited) |
| **Instant** | Winning by reaching max points before timer expires |
| **Bonus** | Points-per-minute rate derived from building strength |
| **Build card** | The vehicle screenshot showing the car in a building |
| **Fingerprint** | 32×32 grayscale thumbnail used for fast image comparison |
| **Lobby** | Screen showing DEFEND/ATTACK choice before entering a building |
| **HUD** | Heads-Up Display — score/timer overlay on the game screen |
