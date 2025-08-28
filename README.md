# DeepLabCut Extended Tracking Branch

This branch builds on the upstream [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) project to explore enhanced tracking capabilities.

## Core Extensions

1. **Velocity-gated ellipse tracker** – The original ellipse tracker is augmented with a speed-space gate that filters unlikely associations based on motion, improving identity preservation when subjects move rapidly.
2. **RFID-assisted long-term tracking** – Radio-frequency identification tags are incorporated to maintain subject identities across long experiments, allowing the system to re-identify individuals after occlusions or extended absences.

## Relationship to DeepLabCut

These extensions are developed on top of the official DeepLabCut codebase and retain compatibility with its APIs and workflow. For comprehensive documentation, tutorials, and the full feature set, please refer to the [DeepLabCut repository](https://github.com/DeepLabCut/DeepLabCut).
