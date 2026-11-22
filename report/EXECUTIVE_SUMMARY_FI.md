# Tutkimuksen ydinanti ja johtopäätökset (Selkokielinen yhteenveto)

**Kataguru Modular MoE Research Program (`v1.0.0`) — 2026-11-22**  
**Tekijä:** Jari Katajisto (Kataguru Research)  
*Täysi tekninen raportti ja mittausdata: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | [CONFOUNDS.md](../ledger/CONFOUNDS.md) | Rinnakkaisjulkaisu (Systems): [kataguru/moe-longcot-nondeterminism](https://github.com/kataguru/moe-longcot-nondeterminism) ([DOI: 10.5281/zenodo.24445305](https://doi.org/10.5281/zenodo.24445305))*

---

## Yhden lauseen ydin

Suureen, valmiiksi koulutettuun MoE-kielimalliin **voi asentaa uusia osaamisia modulaarisina "kasetteina" ilman että koko malli koulutetaan uudelleen ja ilman että vanha osaaminen hajoaa** — mutta vain tiukoilla arkkitehtuuri- ja koulutussäännöillä. Lisäksi osoitimme `sm_120` (RTX 5090) -ympäristössä, että upstreamissa raportoitu `vLLM` Marlin MoE -epädeterminismi ([`vllm#52525`](https://github.com/vllm-project/vllm/issues/52525#issuecomment-7082387518)) **ei ole "quality-neutral" pitkissä päättelyketjuissa (GPQA `44/198` flippauksia)** — ja että sekä **`VLLM_BATCH_INVARIANT=1`** (`TritonWNA16Experts`) että avoin korjaus-PR **`#52532`** (`MarlinExperts` säilytettynä) korjaavat julkisen AWQ-MoE:n **`0/20 → 20/20` bittitarkaksi**.

---

## Tutkimuksen 5 arvokkainta antia

### 1. `vLLM` Marlin MoE -epädeterminismi (`vllm#52525`) rikkoo pitkän CoT:n benchmark-vertailut — `VLLM_BATCH_INVARIANT=1` ja PR `#52532` palauttavat `20/20` bittitarkkuuden
- Upstream-ketju **[`vllm#52525`](https://github.com/vllm-project/vllm/issues/52525)** (korjaus-PR:t `#52532`, `#48032`; toistettu `sm_80/90/121`:llä) kuvasi `moe_align_block_size × Marlin` -epädeterminismin, mutta piti sitä lyhyillä GSM8K/IFEval-testeillä kohinaneutraalina (*"accuracy unaffected"*).
- **Oma kontribuutiomme (`sm_120` / RTX 5090, `vLLM 0.29.0`, `PyTorch 2.12.1+cu130`, `VLLM_MARLIN_USE_ATOMIC_ADD=False`):**
  - Pitkässä CoT:ssa (`GPQA-Diamond`, `N=198`) samat painot flippasivat **`44/198` (`22,2 %`) vastausta (`87/198` pass/fail)** `w=16`:ssa ja **`15/30` (`50 %`)** `TP=1, batch=1, --enforce-eager` -ajossa.
  - **Saman julkisen `Qwen3-Coder-30B-A3B-Instruct-AWQ`:n A/B/C-testi (`sm_120`, `batch=1`, `temp=0`):**
    - **Oletus (`MarlinExperts`):** **`0/20`** täsmäosumaa (**`7/20 = 35 %`** vastausflippauksia, `852,9 s`).
    - **`VLLM_BATCH_INVARIANT=1` (`TritonWNA16Experts`):** **`20/20` (`100 %`)** bittitarkat vastaukset (**`0/20`** flippiä, `7 067` merkkiin asti, `939,8 s`, `+10,2 %` latenssi).
    - **PR `#52532` (`MarlinExperts` säilytettynä, `VLLM_BATCH_INVARIANT=0`):** **`20/20` (`100 %`)** bittitarkat vastaukset (**`0/20`** flippiä, `6 079` merkkiin asti, `945,3 s`, `+10,8 %` latenssi); synkronoimattomassa CUDA-Event-kerneltestissä `0/20` eroa kaikilla `bs_m ∈ {8, 16, 32, 64}` (`+63,6–97,3 µs/kerros` esitäytössä, `0 µs` `M=1` -dekoodauksessa).
  - Tuoreella kemian kysymyksellä oletus-Marlin piti Token #0 argmaxin (`'I'`) samana (`0/4`), mutta `5/5` top-5-logprobit erosivat (rank-2 `'Let'` `1,000` nat) ja teksti haarautui kohdissa **sana 45 / sana 53** (`"5d⁵"`-hallusinaatio); sekä `VLLM_BATCH_INVARIANT=1` että PR `#52532` olivat `5/5` (Token #0) + `3/3` (512 tok) bittitarkkoja.
  - Testattu: `AWQ W4A16`, `NVFP4`-Marlin (`GPTQ` tunnettu upstreamissa; `MXFP4` testaamatta). `Dense NVFP4` (`13/13` valmistunutta paria), `OLMoE BF16` (`20/20`) ja `OLMoE FP8` (`20/20`) olivat 100 % bittitarkkoja. `0,68 s` `sm_120` -varmennus erotteli `sorted_token_ids`, `GEMM1` ja täyden `LAYER_OUT` (`20/20` LIVE vs. `0/20` FIXED, rivi-maksimin BF16 ULP `0,0625` kohdassa `ref_max = 12,75 / 11,50`).

### 2. Alkukerrokset (`L00–L05`) ovat mallin yhteinen kieli- ja syntaksikeskus — älä koske niihin (**Laki 5**)
- Kun kasettia koulutettiin kaikissa 40 kerroksessa, muut taidot hajosivat (`KL` 60-kertaistui, `bfcl +0,0146`, `stem +0,0088`). Pelkkä `L00–L05`-jäädytys poisti vahingon (`KL 0,008`) ja paransi kohdetaidon oppimista **`+40,7 %`**.
- Vastaavan `34/40`-kapasiteetin kontrolli (`L00–L33` auki, `L34–L39` jäässä → täysi FAIL `bfcl +0,0220`, `stem +0,0183`) todisti juurisyyksi juuri `L00–L05`-hubin. Viron kielen koe (`771` riviä × 2 epookkia, `base_nll = 0,314`) ylisovitti kaikilla kerrosrajoilla (`L0 ≈ L2 ≈ L6` — yhteensopiva joko datakonfoundin tai `schedule × 2-epookki` -ristiriidan kanssa; 1-epookkinen kontrolli `EXP-0013b` vaaditaan) → Laki 5 pidetään domeenirajattuna.

### 3. Pelkkä "eri ekspertit" ei vielä takaa, etteivät kasetit häiritse toisiaan (**Laki 7**)
- Erilliset eksperttinumerot poistavat slottikilpailun (`M[i][j] = 0 %`), järjestysherkkyyden (`0 %`) ja yhdistysajan (`0 s`).
- Täysi `3×3`-matriisi (`6/6` suunnattua paria, `L06–L39`) selittää `Union-3`-tulokset kriteerillä $\max_{j \neq i} \rho(j \to i) < 1,0$: **`Coder-12`** (`max_j = 0,1611×` → `98,11 %` PASS), **`Cyber-4`** (`max_j = 0,6014×` → `110,72 %` PASS), mutta **`Testing-QA-4`** (`max_j = 1,8723× > 1,0` → `61,46 %` FAIL P6). Piilotilojen Grassmann-vertailu falsifioitui (`0,2015 < 0,2292`). *Kandidaattiehto (`3` domeenia / `6` suunnattua paria), ei vielä yleispätevä laki.*

### 4. Koulutuksen ja servauksen tekniset ansat
- **`BF16`-pyöristysansa (`C-006`):** `BF16` ilman `FP32`-isäntäpainoja tappaa/vääristää pienet päivitykset (`cos = 0,15–0,35`). `FP32 + autocast + AdamW8bit` vastaa puhdasta `FP32`:ta 6 desimaalilla (toistettu `OLMoE-1B-7B`:llä).
- **Reitittimen `u`-rivien `FP32`-pakko & rank-riippuvainen evakuatio (`Laki 1`):** Evakuation vakiotermi (`α = 0,00`, kloonin `down_proj` nolla) on rank-riippuvainen (`rank1`: `noncode +0,00871`, `rank10`: `−0,00169 ≈ 0`) — tämä **vahvistaa** kaksitermisen evakuatio + kaksoislaskenta -dekomposition. `B1` FP64-neutraalisuunta `u` leikkasi `α=1,00`-shokin `10,9×` (`OLMoE`), `39×` (`32B`) ja `186×` (`Qwen3.5-Base`), mutta `BF16`-muunnos syö **`93,7 %`** `+0,003`-turvabudjetista (`bias_err = 0,00281`) → `u`-rivit pidettävä **`FP32`:ssa servessä (`1,3 MB`)**.
- **Kosinikäyrä vs. vakio-LR (`EXP-0016-T1 v3`):** `240` ajon LR-verhokäyrä osoitti merkkikäännöksen: kosini voittaa 4 epookin toistuvalla datalla, mutta **1 epookin ainutkertaisella datalla vakio-LR voittaa `1,32×` keskimääräisessä oppimisessa (`−1,7481` vs. `−1,3205` matchatulla `ProtKL ≈ 0,107`)** (`W4`:n `11,0×` on terminaalisen LR:n aikatauluartefakti → vain diagnostiikka). *Ristiriita:* tuotanto käyttää nyt `WSD`:tä, jonka T1 sijoitti vakio-LR:n alle → `OLMoE`-varmennus (`EXP-0016-T3`) vaaditaan ennen reseptimuutosta.
- **Unohduksen paikkaus:** Unohdusta paikataan mallin jo osaamalla matalan NLL:n *known*-ankkurilla (`base_nll <= 0,55`) — ei vaikealla uudella datalla.

### 5. Itsekriittinen fail-closed-portti, Laki 2 -ablaatio ja rehellinen tilastorajaus
- Koneellinen fail-closed-portti (`certificate.schema.json` Draft 2020-12 `v1.4`) pysäytti **`4/4` viallista kasettia** ja hyväksyi `3/3` kelvollista (`0/7` false-safe, `95 %` CP yläraja `34,8 %`), ja `validate_certificate.py` pakottaa 7 mutaatiotestiä (mukaan lukien `carries_neutral_u: true` ⇒ `router_dtype == "float32"` sekä symmetrinen `flip_rate`-tarkistus). `P2` (`Top-k Jaccard`) erotti `n=4` tapaukset mutta pudotettiin diagnostiikaksi, koska `J >= 0,99` ei toimi ja uusi kynnys on OOS-validoimaton.
- **Laki 2 & C1/C2-tarkistus:** Oma painottamaton L2 falsifioitui (`LOCO-11 R² = −0,1003`) ja pelkkä binäärinen `1[L00–L05 auki]` epäonnistui (`R² = −0,1231, TP = 0/5, p = 0,398`). Jatkuva `L05_continuous_only` saavutti **`R² = +0,5787, TP = 5/5, FN = 0/5`** (tarkka äärellisen populaation permutaatio: **`5/990 = 0,00505`** jatkuville 4-domeenivektoreille, **`1/110 = 0,00909`** järjestetylle parille, **`1/55 = 0,01818`** järjestämättömälle binääriparille; rajoittamattoman OLS:n `w_L639` on clamped-datassa ei-merkitsevä `−0,00792`, raakadatassa merkitsevä `−0,03891` → NNLS-nollaus on rajakiinnitys; turvallisten kasettien yliennuste `+0,00094 / +0,00336` raportoitu erikseen; päärajoitus `n_bad = 2` kasettia).
- **Monivertailurajaus:** Stable-item McNemar heikensi `LCB-Medium`-otsikkoa (`0,00781 → 0,03125`, `b=6, c=0`). Koko 12 testin perheessä yksikään ei selviä perhekohtaisesta korjauksesta (`p_min = 0,03125 > 0,00417`). `LCB-Medium` on post-hoc-primääri; konfirmatorinen tuore holdout (`EXP-0017`) vaaditaan ennen julkista koodausparannusväitettä.

---

## Julkaisupari (`v1.0.0`)

1. **Systems-julkaisu:** [kataguru/moe-longcot-nondeterminism](https://github.com/kataguru/moe-longcot-nondeterminism) ([DOI: 10.5281/zenodo.24445305](https://doi.org/10.5281/zenodo.24445305)) — Long-CoT benchmark-validiteetti, `sm_120` `vllm#52525` -varmennus, `VLLM_BATCH_INVARIANT=1` & PR `#52532` `20/20` A/B/C-korjaus ja Law 6.
2. **Methods-julkaisu (tämä repo):** [kataguru/functional-locality-moe](https://github.com/kataguru/functional-locality-moe) — Modulaarinen MoE-lokaalius, 3-arkkitehtuurin reproduktio, L1/L2/L5/L6/L7, fail-closed JSON-skeemasertifikaatit, kaikki negatiiviset tulokset sekä avoimet konfirmatoriset portit.
