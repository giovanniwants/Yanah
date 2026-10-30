# Yanah

**Private, local AI for macOS.** 
Run an 80-billion parameter intelligence on your own hardware. No cloud, no subscriptions, no telemetry.

[Download Yanah v0.2.0 (macOS)](https://github.com/giovanniwants/Yanah/releases/latest) • [Read the Architecture](https://yanahchat.com/architecture) • [Website](https://yanahchat.com)

---

## The "Impossible" 

A few months ago, running a frontier-class 80B parameter model required a data center. Today, Yanah runs it on an 8GB MacBook. 

How? We wrote a custom MLX inference engine (`moe_prefetch`) that exploits the physical structure of Mixture-of-Experts (MoE) models. By applying surgical 8-bit/4-bit/2-bit quantization and streaming the inactive experts directly from the SSD via zero-copy Unified Memory DMA, Yanah only loads the 3B parameters it needs for each specific word.

| Hardware | Model | Mode | Speed |
| :--- | :--- | :--- | :--- |
| **64GB M1 Max** | Qwen3-Next-80B-A3B | Deep Mode (SSD Stream) | ~15-20 tok/s |
| **64GB M1 Max** | LiquidAI LFM2-24B-A2B | Quick Mode (In RAM) | ~18+ tok/s |
| **8GB M1/M3** | Qwen3-Next-80B-A3B | Deep Mode (SSD Stream) | ~5-9 tok/s |
| **8GB M1/M3** | LiquidAI LFM2-24B-A2B | Quick Mode (SSD Stream) | ~9-18 tok/s |

---

## Features (v0.2.0)

*   **100% Offline:** Turn off your Wi-Fi. Yanah keeps working.
*   **Multimodal Context:** Drop in a PDF, an Excel spreadsheet, a CSV, or an image. Yanah processes it locally using PyMuPDF, `calamine`, and Qwen3-VL/Moondream2.
*   **Proprietary Local Memory:** Yanah remembers your conversations using a fast local SQLite index and plain Markdown files. Your data never leaves your Mac.
*   **Built-in Personas:** 8 distinct personas included out of the box.
*   **Signed & Notarized:** The `.dmg` is fully signed and notarized by Apple. No scary security warnings.

---

## How it works

Traditional SSD streaming thrashes because it waits for the GPU to ask for data. Yanah's `moe_prefetch` engine uses a **two-layer GPU router lookahead** to predict which experts are needed *before* the GPU asks. This hides the NVMe latency and allows consumer hardware to run models far larger than their physical RAM.

The app itself is a hybrid: a native macOS PyWebView wrapper controlling a custom Python orchestration layer, with dual sidecars for Quick Mode (LFM2) and Vision (Qwen3-VL / Moondream2).

---

## Installation

1.  Download the latest `Yanah-0.2.0.dmg` from the [Releases page](https://github.com/giovanniwants/Yanah/releases/latest).
2.  Open the DMG and drag **Yanah** to your Applications folder.
3.  Launch Yanah. 
4.  On first launch, Yanah will detect your Mac's RAM and download the appropriate models (from ~13GB to ~45GB depending on your hardware). 

*Requires macOS on Apple Silicon (M1/M2/M3/M4).*

---

## Repository Structure

*   `moe_prefetch/`: The custom MLX SSD-streaming inference engine.
*   `start_yanah.py`: The macOS app launcher and supervisor.
*   `yanah_shell.py`: The chat orchestration and memory system.
*   `mlx_server.py`: The 80B Deep Mode engine.
*   `lfm2_server.py`: The 24B Quick Mode engine.
*   `download_models.py`: Resumable model downloader.
*   `quantize_2bit.py`: The custom 8/4/2 hybrid quantization script.
*   `Yanah.spec`: PyInstaller build configuration.
*   `personas/`: The built-in AI personas.
*   `vision_server.py`: Local vision and PDF extraction sidecar.

---

## License & Source

Yanah is **Source Available** under the Functional Source License (FSL). 
We believe in transparency. You can audit the code, verify your privacy, and see exactly how the engine works. The source is available to prevent vendor lock-in and to allow developers to learn from our SSD-streaming techniques. 

*(Note: The FSL restricts using this code to build a competing commercial SaaS product. For personal and non-competing commercial use, it is free.)*

---

## Contributing

Yanah is built on the shoulders of giants. We stand on the breakthroughs of `llama.cpp`, `mlx`, `Swiftlet`, `sushi`, and `SlotStream`. 

If you want to contribute to the `moe_prefetch` engine, open an issue or submit a pull request. We are particularly interested in optimizations for the 16GB-24GB tier, and adding support for new MoE architectures.

**Let's build a private, sovereign alternative to the cloud.**
