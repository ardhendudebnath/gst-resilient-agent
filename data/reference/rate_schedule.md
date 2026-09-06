# GST Rate Schedule Reference — as in force today

**Status:** v1.0. The schedule structure and the label space below are
**verified against the Gazette text** of Notification 9/2025 (52 pages) and its
amending Notification 19/2025. Individual entries are still checked one at a
time as they are used — see [Verification checklist](#verification-checklist).

This file is the *only* authority an annotator may consult when assigning a
slab. If a product cannot be placed using this file plus the Customs Tariff
headings, the example is `UNANSWERABLE` — it is never resolved by intuition,
by a search engine, or by asking a model.

---

## 1. The governing notifications

| Notification | Dated | In force from | Effect |
|---|---|---|---|
| **9/2025–Central Tax (Rate)** | 17 Sep 2025 | 22 Sep 2025 | Supersedes 1/2017–CT(R). Rated goods, Schedules I–VII |
| **10/2025–Central Tax (Rate)** | 17 Sep 2025 | 22 Sep 2025 | Supersedes 2/2017–CT(R). Exempt (nil-rated) goods |
| **19/2025–Central Tax (Rate)** | 31 Dec 2025 | **1 Feb 2026** | Amends 9/2025. **Omits Schedule VII entirely**; moves tobacco and pan masala to Schedule III, biris to Schedule II |

Parallel IGST and UTGST notifications were issued the same day. This benchmark
labels the **combined GST rate** (CGST + SGST, equivalently the IGST rate), so
the CGST figures in the notification are doubled throughout.

These notifications implement the decisions of the **56th GST Council meeting
(3 September 2025)**, commonly called *GST 2.0*.

---

## 2. Label space

Notification 9/2025 lays out seven schedules. Doubling the CGST rate gives the
combined GST rate that this benchmark uses as its label:

Schedule order and rates below are **read from the Gazette text**, not inferred.

| Schedule | CGST | **Combined GST (the label)** | Broad coverage |
|---|---|---|---|
| I | 2.5 % | **5 %** | Essential and mass-consumption goods |
| II | 9 % | **18 %** | Standard rate — the residual for most goods |
| III | 20 % | **40 %** | Demerit and luxury goods, incl. tobacco from 1 Feb 2026 |
| IV | 1.5 % | **3 %** | Precious metals (gold, silver, platinum) |
| V | 0.125 % | **0.25 %** | Rough / unworked precious stones |
| VI | 0.75 % | **1.5 %** | Cut and polished diamonds |
| ~~VII~~ | ~~14 %~~ | ~~28 %~~ | **Omitted 1 Feb 2026 by Notification 19/2025** |

Plus, from Notification 10/2025:

| Source | **Combined GST (the label)** | Coverage |
|---|---|---|
| Notification 10/2025 Schedule | **0 %** | Exempt goods |

### Permitted label values

```
0    0.25    1.5    3    5    18    40    UNANSWERABLE
```

### ⚠ Two slabs are abolished, on two different dates

| Slab | Abolished | By |
|---|---|---|
| **12 %** | 22 Sep 2025 | The 6 % CGST schedule of Notification 1/2017 has **no successor** in 9/2025. Items formerly at 12 % went predominantly to 5 %. |
| **28 %** | 1 Feb 2026 | Notification 19/2025 omits *"the Schedule VII – 14 %, and the entries relating thereto"*. |

**Neither `12` nor `28` is a valid label**, and `harness/schema.py` rejects both.

This is the benchmark's primary probe of stale parametric knowledge, and the
two dates make it sharper than a single cutoff would: a model emitting `12` is
reciting a table that died in September 2025, while one emitting `28` is
eighteen months stale in a different way. Both are reported as the
**stale-slab rate**, a first-class leaderboard column rather than something
folded into plain accuracy.

### Where the demerit goods went

Resolved — this was an open question in the draft of this file, and the
amending notification settles it:

| Goods | From 1 Feb 2026 |
|---|---|
| Pan masala (2106 90 20), unmanufactured tobacco (2401), cigars and cigarettes (2402), other manufactured tobacco (2403 other than biris), tobacco for inhalation (2404) | **Schedule III — 40 %** |
| **Biris** (2403 19 21, 2403 19 29) | **Schedule II — 18 %** |

Compensation cess on tobacco was simultaneously reduced to nil, and a separate
excise duty introduced, so the cess no longer sits on top of the GST rate.

Read directly from the Gazette, **Schedule VII contained only six entries** —
pan masala and the five tobacco headings — and nothing else. Omitting it
therefore stranded no other family.

### Aerated drinks and cement were never unsettled

Both were excluded from v1.0 on the assumption that they sat in Schedule VII
awaiting a transitional decision. Reading the notification shows they never did:

| Goods | Heading | Schedule | Rate | Pre-reform |
|---|---|---|---|---|
| All goods incl. aerated waters, containing added sugar or flavoured | 2202 10 | III | **40 %** | 28 % + cess |
| Carbonated beverages of fruit drink / with fruit juice | 2202 | III | **40 %** | 28 % + cess |
| Caffeinated beverages | 2202 99 90 | III | **40 %** | 28 % + cess |
| Other non-alcoholic beverages | 2202 91 00, 2202 99 90 | III | **40 %** | |
| Waters without added sugar; plant-based, soya and milk drinks; fruit-juice based (non-carbonated) | 2201, 2202 99, 2202 99 10/20/30 | I | **5 %** | |
| Portland, aluminous, slag and super-sulphate cement, incl. clinkers | 2523 | II | **18 %** | 28 % |

Neither family is touched by Notification 19/2025.

**Consequence for scope.** Every family excluded by `guideline.md` §4d now has
a settled rate, so none of them is excluded because the answer is unknown — the
exclusion is a scope choice throughout, and a costly one. Cement moved 28 % →
18 % and aerated drinks 28 % + cess → 40 %, so both are `rate-changed-2025`
examples *and* unusually sharp stale-slab probes: a model reciting the old
table answers 28 % for either, and 28 % is itself now an abolished slab. Biris
at 18 % against the rest of the tobacco family at 40 % is a similar
within-family split. These are among the best examples available, and they are
currently being filtered out.

---

## 3. Ordered lookup procedure

Annotators apply these steps **in order** and stop at the first that resolves.
Following a fixed procedure rather than judgement is what makes the
self-agreement check meaningful.

1. **Fix the essential character** of the good from the description alone.
   If this fails → `UNANSWERABLE`.
2. **Assign the HSN heading** (4-digit) using the Customs Tariff and the
   General Rules of Interpretation (GRI 1 → 3(a) → 3(b) → 3(c)).
3. **Check Notification 10/2025** (exemptions). If listed → `0`.
4. **Check Notification 9/2025** Schedules I–VII. First matching entry wins.
5. **Apply conditionality tests** (§4 of `../guideline.md`) — most importantly
   *pre-packaged and labelled*, which can move a food item between `0` and `5`.
6. **Residual.** A good not covered by any specific entry falls to the
   Schedule II standard rate → `18`.
7. If the rate turns on a fact the description does not state → `UNANSWERABLE`.

---

## 4. Primary sources

**The Gazette texts are archived in this repository**, under
[`primary/`](primary/), pinned by SHA-256 in `primary/MANIFEST.json`:

| File | Document | Pages |
|---|---|---|
| `09-2025-CTR.pdf` | Notification 9/2025–CT(R), rated goods | 52 |
| `10-2025-CTR.pdf` | Notification 10/2025–CT(R), exempt goods | 12 |
| `19-2025-CTR.pdf` | Notification 19/2025–CT(R), omitting Schedule VII | 2 |

```bash
make verify-sources    # checks hashes and re-reads the entries that must exist
```

CI runs it, so a corrupted or swapped file fails the build rather than quietly
changing what the dataset means. Verification is not only by hash: each
document carries content assertions — 9/2025 must contain "Motorcycles of
engine capacity exceeding 350 cc", 19/2025 must contain "Biris" and "shall be
omitted" — so the check confirms the file is the document it claims to be, not
merely unchanged.

| Also useful | Where |
|---|---|
| CBIC GST rate finder | <https://taxinformation.cbic.gov.in/> |
| CBIC GST rates landing page | <https://cbic-gst.gov.in/gst-goods-services-rates.html> |

Secondary commentary was used to establish the *structure* of the schedules
during drafting, and every structural claim has since been replaced by a
reading of the Gazette. No secondary source is permitted as the authority for
an individual example's label.

---

## Worked verification — heading 9608

A record of the method, and of the one entry checked end-to-end so far.

The Gazette text of Notification 9/2025 places, at **Schedule II (9 % CGST),
serial 626**:

> 9608 — Ball point pens; felt tipped and other porous-tipped pens and markers;
> fountain pens; stylograph pens and other pens; duplicating stylos; pen
> holders, pencil holders and similar holders; **parts (including caps and
> clips) of the foregoing articles**, other than those of heading 9609

So heading 9608, *including its parts*, is **18 %**. Heading 9609 appears in
the notification only inside that exclusion clause and has no rated entry of
its own, which is consistent with pencils and crayons having moved to nil.

**Consequence for the ruling in `first_pass.jsonl` on pen tips and balls:** the
authority placed them in Schedule III of Notification 1/2017, which was 18 %,
and they sit at 18 % today. **The rate did not move**, so that example must not
carry the `rate-changed-2025` tag — even though "12 %" appears in the ruling,
because that was the applicant's rejected contention.

## Verification checklist

- [x] Obtain the Gazette text of Notification 9/2025 and read the schedule
      structure from it rather than from commentary.
- [x] Confirm no 6 % CGST schedule exists — `12` really is unavailable.
- [x] Resolve Schedule VII and the tobacco transitional position.
      **Omitted from 1 Feb 2026 by Notification 19/2025**; tobacco to
      Schedule III (40 %), biris to Schedule II (18 %).
- [x] Check for amending notifications after 17 September 2025.
      Found 19/2025 (31 Dec 2025, in force 1 Feb 2026).
- [x] Confirm the position on aerated beverages and cement.
      **Both settled and neither was ever in Schedule VII**: aerated and
      carbonated drinks at Schedule III (40 %), cement at Schedule II (18 %).
- [x] Re-verify the rates asserted in the five worked examples of
      `../guideline.md`. **Four confirmed with schedule and serial recorded;
      WE-5 was wrong** — it claimed candles at 18 % when Schedule I serial 253
      rates them at 5 % — and has been rewritten around what the notification
      actually says.
- [x] Decide whether to bring cement, aerated drinks and tobacco into scope.
      **All three lifted.** Alcohol is the only remaining exclusion, and it is
      categorical rather than a scope choice.
- [x] Archive the Gazette PDFs into `primary/` with their SHA-256 hashes.
      Done, with `MANIFEST.json` and `python -m harness.verify_primary`, which
      CI runs. **Partly open:** the copies came from the taxo.online mirror,
      because the official CBIC portal serves an incomplete TLS certificate
      chain and could not be fetched over a verified connection. Bypassing
      verification for documents whose integrity is the whole point would be
      self-defeating, so the copies are pinned by hash and their content is
      asserted instead. Re-fetch from CBIC if their chain is ever fixed.

**Last checked against primary sources:** 2026-09-04, against the Gazette text
of Notification 9/2025 and 19/2025.
