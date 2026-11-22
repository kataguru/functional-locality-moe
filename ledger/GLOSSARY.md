# Käsitteet ja Metriikkamääritelmät (`GLOSSARY.md`)

> Jokainen tulos ja raportti viittaa näihin täsmällisiin määritelmiin. Diagnostiikkamittareita (`hit`, `rel_dW`, `gate_cos`) ei koskaan sekoiteta tavoitemittareihin (`ASST_d`, `upper 95% CI`, `pass@1`, `Bench 15`).

## 1. Kolme Lokaaliusakselia (Charterin selkäranka)
- **`L1` Reitityslokaalius (Routing Locality):** Muuttuuko, mitkä ekspertit valitaan kohdedomeenin $D$ ulkopuolella? Mitataan: `leak`, `top-10 Jaccard`, `e_10`-evakuatio, slot-varastusmatriisi `M[i][j]`.
- **`L2` Funktiolokaalius (Functional Locality):** Kun $D$:n ulkopuolinen tokeni yhä valitsee muuttuneen ekspertin, pysyykö ekspertin ulostulo ja mallin ennustejakauma samana? Mitataan: `expert-output Δ`, `ASST_d` (`upper 95% CI`), `KL(base || new)`.
- **`L3` Kompositiolokaalius (Composition Locality):** Muuttaako patruuna $A$ niitä olosuhteita (hidden-jakauma, `core_k10`, vapaat slotit), joissa patruuna $B$ kalibroitiin? Mitataan: pari-slot-varastusmatriisi `M[i][j]`, järjestysvarianssi, union-vuoto.

## 2. Reititys- ja Aineistometriikat (Diagnostiikkaa, EI optimointitavoitteita)
- **`any_hit` / `hit` (`%`):** Osuus tokeneista, joilla **vähintään yksi** kohdejoukon $S$ eksperteistä pääsee Top-10:een: $P(\exists e \in S : e \in \text{Top10}(t))$.
- **`touch` (token-kosketus, `%`):** Sama kuin `any_hit` korvattaville eksperteille (`Track B`): osuus ei-kooditokeneista, jotka koskettavat vähintään yhtä muuttunutta eksperttiä kerroksessa $\ell$. Raportoidaan aina `mean`, `max_layer` ja kumulatiivinen sekvenssitaso.
- **`E[slots]` (slot-osuus):** Odotettu määrä joukon $S$ eksperttejä Top-10:ssä per token: $\mathbb{E}[|S \cap \text{Top10}(t)|] \in [0, 10]$. Huom: `0,3–0,5` slottia voi vastata `30–50 %` token-kosketusta (`F-019`).
- **`leak` (`%`):** `any_hit` (tai `E[slots]`) kohdedomeenin $D$ **ulkopuolisilla** tokeneilla (`STEM`, `IFEval`, `BFCL`, `GSM8K`). Raportoidaan aina **sekä keskiarvo että pahimman kerroksen `max_layer` ja alkukerrosten `L0–5 max`**.
- **`Top-10 Jaccard`:** Parittainen reititys-identtisyys pohjamalliin nähden samalla tokenilla $t$: $J(t) = \frac{|\text{Top10}_{\text{new}}(t) \cap \text{Top10}_{\text{base}}(t)|}{|\text{Top10}_{\text{new}}(t) \cup \text{Top10}_{\text{base}}(t)|}$. **Protokollamuutos #2:** pudotettu ei-portittavaksi diagnostiikaksi (`diagnostics`) Law 6 `#k/#k+1`-rajapintaherkkyyden vuoksi (alkuperäinen `>= 0,99` ei päde; uudelleenkalibroitu kynnys on OOS-validoimaton).
- **`α` (`alpha`, Laki 1 kloonin ulostuloskaala):** Kloonatun uuden ekspertin `down_proj`-painomatriisin kerroin alustuksessa (`W_down_clone = α · W_down_parent`). `α = 0,00` nollaa kloonin ulostulon kokonaan ja eristää **puhtaan `e_k`-slotin evakuation**; `α = 1,00` lisää kloonin täyden kaksoislaskennan (`evakuatio + kaksoislaskenta`).
- **`ρ(j → i)` (Laki 7 ristiindominanssisuhde):** Suunnattu reititinmassan dominanssisuhde kasetista $j$ kasetin $i$ kohdedomeenilla $D_i$ modulaarikerroksilla (`L06–L39`): $\rho(j \to i) = \frac{\text{mass}(S_j \text{ on } D_i)}{\text{mass}(S_i \text{ on } D_i)}$. Laki 7 -kandidaattiehto kompositiosäilyvyydelle: $\max_{j \neq i} \rho(j \to i) < 1,0$.
- **`rel_dW` (`%`):** Suhteellinen painomuutos alustuksesta: $\|W_{\text{new}} - W_0\|_F / \|W_0\|_F$. Diagnostiikkaa (paljastaa BF16-jumiutumisen tai räjähdyksen), **ei** koulutustavoite (`F-012`).

## 3. Tavoitemittarit (Päätös- ja julkaisukriteerit)
- **`ASST_d` (natsia):** Parittainen muutos assistant-vastauksen Cross-Entropy NLL:ssä (`labels != -100`, `max_len >= 2048`): $\Delta_{\text{ASST}} = \text{NLL}_{\text{cand}}^{\text{asst}} - \text{NLL}_{\text{base}}^{\text{asst}}$. Negatiivinen = parempi.
- **`PROMPT_d` (natsia):** Sama käyttäjäpromptin tokeneille (erillinen diagnostiikka, ei saa sotkea `ASST_d`:hen).
- **`upper 95% CI` (sekvenssibootstrap):** Lasketaan per-sekvenssi keskiarvo $\Delta_i$, bootstrapataan sekvenssejä (`B = 2000`), raportoidaan 97,5. persentiili. Kova turvallisuusehto standardiportissa (`80k–120k` tok/domeeni): **`noncode ASST_d upper 95% CI <= +0,003`**.
- **`expert-output Δ`:** Normalisoitu router-painotettu ulostulodrift ei-koodilla: $\frac{\sum_t w_j(t)\|E_{\text{new}}(h_t) - E_{\text{old}}(h_t)\|^2}{\sum_t w_j(t)\|E_{\text{old}}(h_t)\|^2 + 10^{-6}}$.

## 4. Validointipino (`T0–T5`)
| Taso | Malli / Precisio | Tarkoitus | Budjetti |
| :---: | :--- | :--- | :---: |
| **T0** | CPU / synteettinen MoE | Router-matematiikka, dtype, manifestit, holdout-lukko | sekunteja |
| **T1** | Jäädytetty `CodeRestore-232E`, cachetetut `h`/Top-10 | Init-offset, salienssi, token-touch, ankkuri-local | `5–15 min` |
| **T2** | Slice-probe (4–8 kerrosta tai lyhyt 50-step halving) | LR-, ankkuri- ja datasekoitussuunta (ei julkaisupäätös) | `15–40 min` |
| **T3** | Täysi runko, parittainen cached NLL-portti (`80k–120k` tok) | `ASST_d` + `upper 95% CI`, `Top-10 Jaccard`, pitkä konteksti | `10–20 min` |
| **T4** | Täysi candidate, greedy (`DEV-40 LCB-Hard` + canaryt) | Käytösvalidointi ennen julkaisua (`10` BFCL/GPQA/IFEval/GSM8K) | `15–30 min` |
| **T5** | Täysi `Bench 15` (`NVFP4 TP=2`, 2 siementä, McNemar) | Virallinen julkaisusertifikaatti (`HOLDOUT-40`) | `1–2 h` |
