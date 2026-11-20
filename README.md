# kokoro-tonal-kit 🗣️

A clean, language-agnostic starter toolkit for adapting **Kokoro-82M / StyleTTS2** to **tonal and low-resource languages** (Igbo, Twi, Hausa, Ewe, Yorùbá, Lingala, Vietnamese, etc.) on modern **PyTorch 2.6+** and **Google Colab (T4 / L4 GPUs)**.

Extracted from real-world lessons while building **[Kokoro Yorùbá](https://github.com/Akintunde102/kokoro-yoruba)** and **[`oro-g2p`](https://github.com/Akintunde102/oro-g2p)**.

---

## What's Inside

| File | Purpose |
| :--- | :--- |
| **`patch_styletts2.py`** | Universal idempotent patcher fixing PyTorch 2.6 `weights_only=False`, `Modules/slmadv.py` `batch=1` crashes, `Modules/istftnet.py` 1D/3D shape crashes, `spectral_norm` encoder warmup, Stage 2 `no_grad` VRAM burn, and NaN stability. |
| **`vocab_template.py`** | Packs custom `(sound + tone)` symbols into Kokoro-82M's **63 vacant token slots** inside `0..177` without resizing any embedding tensors. |
| **`export_release.py`** | Strips `optimizer` and GAN critics (`mpd`, `msd`, `wd`) to shrink a ~2.1 GB training checkpoint to a ~500 MB release file. |
| **`configs/stage2_recommended_notes.yml`** | Key Stage 2 hyperparameters (`lambda_diff=1.0`, `lambda_sty=1.5`, `lambda_F0=4.5`, `bert_lr=4e-5`) to avoid flat prosody. |

---

## 5-Step Blueprint for Any Tonal Language

1. **Pack `Sound + Tone` Atomically (`vocab_template.py`):**
   Never emit tone marks as separate trailing tokens after vowels — the duration predictor will stretch or drop trailing tone symbols. Glue `vowel + tone` (and `syllabic_nasal + tone`) into single atomic slots inside Kokoro's 63 free slots.
2. **Curate Short, Silence-Trimmed, Speaker-Pure Clips:**
   Trim silence (`top_db=35`), keep utterances between `1.2s – 5.8s`, and assign every voice its own integer `speaker_id` (`clip.wav|ipa_string|speaker_id`).
3. **Clone `StyleTTS2` & Patch:**
   ```bash
   git clone https://github.com/yl4579/StyleTTS2.git
   python patch_styletts2.py --styletts2_dir ./StyleTTS2
   ```
4. **Train Stage 1 → Stage 2 (Keep Diffusion Enabled!):**
   Ensure `lambda_diff >= 1.0`, `lambda_sty >= 1.0`, `lambda_F0 >= 3.0–4.5`, and `bert_lr >= 4e-5`.
5. **Export & Serve:**
   ```bash
   python export_release.py --ckpt epoch_2nd_00014.pth --config config_stage2.yml --out kokoro_lang.pth
   ```
   Use punctuation chunking and `alpha=0.0` during inference for steady vocal cords.

---

## Ecosystem
- **Reference Implementation (Yorùbá TTS):** [Akintunde102/kokoro-yoruba](https://github.com/Akintunde102/kokoro-yoruba)
- **Yorùbá G2P Engine:** [Akintunde102/oro-g2p](https://github.com/Akintunde102/oro-g2p) (`pip install oro-g2p`)
